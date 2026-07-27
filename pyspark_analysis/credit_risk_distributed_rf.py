# Databricks notebook source
# ---------------------------------------------------------------------------
# Credit Risk Classification — Distributed Random Forest (PySpark / MLlib)
#
# Originally developed and run on a Databricks single-node cluster
# (Runtime 12.2 LTS, Spark 3.3.2). The dbfs:/FileStore paths below are
# Databricks-native; to run in a plain local/standalone Spark session,
# replace them with a local filesystem path to HPCI_Cleaned_Data.csv
# (see data/sample/credit_risk_sample_200.csv for a 200-row preview,
# or download the full dataset per the root README).
# ---------------------------------------------------------------------------
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sk_auc, cohen_kappa_score, precision_recall_curve
import pandas as pd
import math
import os

# Ensure /dbfs/FileStore exists
os.makedirs("/dbfs/FileStore", exist_ok=True)

# Start Spark Session
spark = SparkSession.builder.appName("CreditRiskClassification").getOrCreate()

# Load the cleaned dataset from DBFS
df=spark.read.csv("dbfs:/FileStore/HPCI_Cleaned_Data.csv", header=True, inferSchema=True, nullValue="NA")

# Confirm schema of the dataset
df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC # Class Balancing & Train-Test Split (Manual & Controlled)
# MAGIC Ensure class balance in training set and evaluate model realistically with imbalanced test set.

# COMMAND ----------

# Check the class distribution
class_counts = df.groupBy("Risk").count().collect()
print("Original class distribution:")
for row in class_counts:
    print(f"Class {row[0]}: {row[1]} samples")

# Identify class labels
classes = [row[0] for row in class_counts]

train_dfs = []
test_dfs = []

for class_value in classes:
    class_df = df.filter(col("Risk") == class_value)
    class_count = class_df.count()
   
    # Finding the minimum class count
    min_class_count = min([row[1] for row in class_counts])
   
    # Undersample majority class
    if class_count > min_class_count:
        # Undersample to match minority class
        class_df = class_df.orderBy(rand(seed=42)).limit(min_class_count)
        class_count = min_class_count
   
     # Assign row numbers for reproducible split
    class_df = class_df.withColumn("row_num", F.row_number().over(Window.partitionBy("Risk").orderBy(rand(seed=42))))
   
    # Calculate split point (We are using 70% for training)
    train_count = int(class_count * 0.7)
   
    # Split into train and test
    train_class_df = class_df.filter(col("row_num") <= train_count).drop("row_num")
    test_class_df = class_df.filter(col("row_num") > train_count).drop("row_num")
   
    train_dfs.append(train_class_df)
    test_dfs.append(test_class_df)

# Merge all classes back
train_df = train_dfs[0]
for i in range(1, len(train_dfs)):
    train_df = train_df.union(train_dfs[i])

test_df = test_dfs[0]
for i in range(1, len(test_dfs)):
    test_df = test_df.union(test_dfs[i])

# Final shuffle
train_df = train_df.orderBy(rand(seed=42))
test_df = test_df.orderBy(rand(seed=42))

# Verifing the class distribution in train and test sets
print("\nTraining set class distribution:")
train_df.groupBy("Risk").count().show()

print("\nTest set class distribution:")
test_df.groupBy("Risk").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Assembly & Label Encoding
# MAGIC Convert features to vector format and categorical target to numeric label.

# COMMAND ----------


# Define feature columns
feature_cols = [col for col in train_df.columns if col != "Risk"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Apply assembler
train_data = assembler.transform(train_df).select("features", "Risk")
test_data = assembler.transform(test_df).select("features", "Risk")

# Convert Risk to label index
label_indexer = StringIndexer(inputCol="Risk", outputCol="label")
label_indexer_model = label_indexer.fit(train_data)
train_data = label_indexer_model.transform(train_data)
test_data = label_indexer_model.transform(test_data)

# Hardening check: the confusion-matrix-derived metrics further below compare the
# raw "Risk" column directly against the classifier's "prediction" column, on the
# assumption that StringIndexer's 0/1 label ordering matches Risk's own 0/1 coding.
# Assert that explicitly rather than relying on it implicitly.
print("StringIndexer label mapping (index -> original Risk value):", label_indexer_model.labels)
assert label_indexer_model.labels[0] == "0" and label_indexer_model.labels[1] == "1", (
    "StringIndexer label order does not match raw Risk coding — "
    "downstream Risk == prediction comparisons would need remapping."
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Random Forest Model
# MAGIC Train a non-cross-validated Random Forest model with R-tuned hyperparameters.

# COMMAND ----------

# Random Forest Model
rf_model = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
    numTrees=1000,
    featureSubsetStrategy="8",  # Mimics mtry = 8 in R
    seed=42
)

rf_fitted = rf_model.fit(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Evaluate Model – AUC & Accuracy
# MAGIC Quantify model's ROC AUC and raw classification accuracy.

# COMMAND ----------

# Predict on test set
predictions = rf_fitted.transform(test_data)

# AUC evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC")
auc = evaluator.evaluate(predictions)
print(f"AUC: {auc:.4f}")

# Accuracy
accuracy = predictions.filter(predictions.Risk == predictions.prediction).count() / predictions.count()
print(f"Accuracy: {accuracy:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Confusion Matrix Metrics – Precision, Recall, F1, Specificity
# MAGIC Provide deeper insight beyond accuracy using clinical/classification evaluation metrics.
# MAGIC

# COMMAND ----------


# Derive confusion matrix values
preds = predictions.withColumn("TP", when((col("Risk") == 1) & (col("prediction") == 1), 1).otherwise(0)) \
                   .withColumn("TN", when((col("Risk") == 0) & (col("prediction") == 0), 1).otherwise(0)) \
                   .withColumn("FP", when((col("Risk") == 0) & (col("prediction") == 1), 1).otherwise(0)) \
                   .withColumn("FN", when((col("Risk") == 1) & (col("prediction") == 0), 1).otherwise(0))

# Aggregate
agg = preds.selectExpr("sum(TP) as TP", "sum(TN) as TN", "sum(FP) as FP", "sum(FN) as FN").collect()[0]
TP, TN, FP, FN = agg["TP"], agg["TN"], agg["FP"], agg["FN"]

# Compute metrics
precision = TP / (TP + FP)
recall = TP / (TP + FN)
specificity = TN / (TN + FP)
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Visualise ROC Curve
# MAGIC Visualize model performance with respect to trade-off between TPR and FPR

# COMMAND ----------


# Convert Spark predictions to pandas
roc_df = predictions.select("probability", "Risk").toPandas()
roc_df["probability"] = roc_df["probability"].apply(lambda x: float(x[1]))

# Compute ROC curve
fpr, tpr, _ = roc_curve(roc_df["Risk"], roc_df["probability"])
roc_auc = sk_auc(fpr, tpr)

# Plot
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})", color='darkorange')
plt.plot([0,1], [0,1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tuned Random Forest (PySpark)")
plt.legend(loc="lower right")
plt.grid()
plt.savefig("/tmp/roc_curve.png")
dbutils.fs.cp("file:/tmp/roc_curve.png", "dbfs:/FileStore/roc_curve.png")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top 10 most important features 

# COMMAND ----------


# Get feature importances from the trained model
importances = rf_fitted.featureImportances

# Pair feature names with their importance scores
feature_importance_df = pd.DataFrame(
    list(zip(feature_cols, importances)),
    columns=["Feature", "Importance"]
)

# Sort and display top 10
top_10_features = feature_importance_df.sort_values(by="Importance", ascending=False).head(10)
print(top_10_features)

top_10_features.to_csv("/tmp/top_10_feature_importances.csv", index=False)
dbutils.fs.cp("file:/tmp/top_10_feature_importances.csv", "dbfs:/FileStore/top_10_feature_importances.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Plot (Bar Chart)

# COMMAND ----------

top_10_features.plot(kind="barh", x="Feature", y="Importance", legend=False, color="steelblue")
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("/tmp/feature_importance_plot.png")
dbutils.fs.cp("file:/tmp/feature_importance_plot.png", "dbfs:/FileStore/feature_importance_plot.png")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion Matrix

# COMMAND ----------

conf_matrix = preds.groupBy("Risk", "prediction").count().orderBy("Risk", "prediction")
conf_matrix.show()

# Convert to Pandas for export
conf_matrix_pd = conf_matrix.toPandas()

# Add percentage of total
total = conf_matrix_pd["count"].sum()
conf_matrix_pd["percentage"] = (conf_matrix_pd["count"] / total * 100).round(2)

# Save to temporary location and move to DBFS
conf_matrix_pd.to_csv("/tmp/confusion_matrix.csv", index=False)
dbutils.fs.cp("file:/tmp/confusion_matrix.csv", "dbfs:/FileStore/confusion_matrix.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Balanced Accuracy

# COMMAND ----------

balanced_accuracy = (recall + specificity) / 2
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Matthews Correlation Coefficient (MCC)

# COMMAND ----------


mcc_numerator = (TP * TN) - (FP * FN)
mcc_denominator = math.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))
mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Precision-Recall (PR) Curve

# COMMAND ----------


precision_vals, recall_vals, _ = precision_recall_curve(roc_df["Risk"], roc_df["probability"])
plt.plot(recall_vals, precision_vals, color="green", lw=2)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid(True)
plt.savefig("/tmp/precision_recall_curve.png")
dbutils.fs.cp("file:/tmp/precision_recall_curve.png", "dbfs:/FileStore/precision_recall_curve.png")
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Kappa Statistic (Cohen’s Kappa)

# COMMAND ----------


# Kappa requires true labels and predictions
y_true = roc_df["Risk"]
y_pred = predictions.select("prediction").toPandas()

kappa = cohen_kappa_score(y_true, y_pred)
print(f"Cohen's Kappa: {kappa:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gini Coefficient

# COMMAND ----------

gini = 2 * auc - 1
print(f"Gini Coefficient: {gini:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-Class Metrics Table

# COMMAND ----------

# Format all percentage metrics (excluding AUC, Gini, MCC, Kappa)
accuracy_pct = accuracy * 100
precision_pct = precision * 100
recall_pct = recall * 100
specificity_pct = specificity * 100
f1_pct = f1 * 100
balanced_accuracy_pct = balanced_accuracy * 100

# Print formatted metrics
print(f"""
| Metric              | Value     |
|---------------------|-----------|
| AUC                 | {auc:.4f} |
| Gini Coefficient    | {gini:.4f} |
| Accuracy            | {accuracy_pct:.2f}% |
| Precision           | {precision_pct:.2f}% |
| Recall (Sensitivity)| {recall_pct:.2f}% |
| Specificity         | {specificity_pct:.2f}% |
| F1 Score            | {f1_pct:.2f}% |
| Balanced Accuracy   | {balanced_accuracy_pct:.2f}% |
| MCC                 | {mcc:.4f} |
| Kappa               | {kappa:.4f} |
""")

# Save metrics to CSV with percentage values
metrics_dict = {
    "AUC": [round(auc, 4)],
    "Gini": [round(gini, 4)],
    "Accuracy (%)": [round(accuracy_pct, 2)],
    "Precision (%)": [round(precision_pct, 2)],
    "Recall (Sensitivity) (%)": [round(recall_pct, 2)],
    "Specificity (%)": [round(specificity_pct, 2)],
    "F1 Score (%)": [round(f1_pct, 2)],
    "Balanced Accuracy (%)": [round(balanced_accuracy_pct, 2)],
    "MCC": [round(mcc, 4)],
    "Kappa": [round(kappa, 4)]
}

# Create DataFrame and save to FileStore
metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv("/tmp/evaluation_metrics.csv", index=False)
dbutils.fs.cp("file:/tmp/evaluation_metrics.csv", "dbfs:/FileStore/evaluation_metrics.csv")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Model for Deployment & Export Predictions

# COMMAND ----------


# Get probability of class 1 from vector
get_prob = udf(lambda v: float(v[1]), DoubleType())
predictions_cleaned = predictions.withColumn("prob_class_1", get_prob(col("probability")))

# Drop complex fields
predictions_cleaned = predictions_cleaned.select("Risk", "prediction", "prob_class_1")

# Save predictions
predictions_cleaned.write.mode("overwrite").csv("dbfs:/FileStore/credit_predictions.csv", header=True)

# Save to CSV
predictions_cleaned.toPandas().to_csv("/dbfs/FileStore/credit_predictions.csv", index=False)