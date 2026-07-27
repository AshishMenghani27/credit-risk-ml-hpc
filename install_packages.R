# R dependencies for r_analysis/credit_risk_random_forest.Rmd
#
# Note: the original analysis session's exact package versions were not
# recorded (no renv/packrat lockfile was generated at the time), so this is
# a plain install list rather than a version-pinned lockfile. For a fully
# reproducible environment, run renv::init() after installing these once,
# then renv::snapshot() to generate your own renv.lock.

packages <- c(
  "rmarkdown",
  "knitr",
  "ROCR",
  "rpart",
  "cluster",
  "sampling",
  "vcd",
  "caret",
  "corrplot",
  "ggplot2",
  "factoextra",
  "tidyverse",
  "pROC",
  "pdp",
  "dplyr",
  "gridExtra",
  "randomForest"
)

installed <- rownames(installed.packages())
to_install <- setdiff(packages, installed)

if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
} else {
  message("All required packages already installed.")
}
