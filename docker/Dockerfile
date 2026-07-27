# Reproducible PySpark environment for pyspark_analysis/credit_risk_distributed_rf.py
#
# Build:  docker build -t credit-risk-pyspark -f docker/Dockerfile .
# Run:    docker run -it --rm -v $(pwd):/workspace -w /workspace credit-risk-pyspark \
#           spark-submit pyspark_analysis/credit_risk_distributed_rf.py

FROM eclipse-temurin:11-jre-jammy

ENV SPARK_VERSION=3.3.2 \
    HADOOP_VERSION=3 \
    SPARK_HOME=/opt/spark \
    PATH=/opt/spark/bin:$PATH \
    PYSPARK_PYTHON=python3

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" \
    | tar -xz -C /opt && \
    mv "/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}"

WORKDIR /workspace
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["bash"]
