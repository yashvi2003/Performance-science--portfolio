# ==============================================================================
# R Cross-Check Script: Model 1 (Readiness) Linear Mixed-Effects Model
# ------------------------------------------------------------------------------
# Refits pre-registered Model 1 using lme4 / lmerTest in R to verify numerical
# equivalence with the Python statsmodels MixedLM implementation.
# ==============================================================================

# 1. Load required libraries
if (!requireNamespace("lme4", quietly = TRUE)) {
  install.packages("lme4", repos = "https://cloud.r-project.org")
}
if (!requireNamespace("lmerTest", quietly = TRUE)) {
  install.packages("lmerTest", repos = "https://cloud.r-project.org")
}

library(lme4)
library(lmerTest)

# 2. Load the exported Currie-Hoffmann centered dataset
data_path <- "data/processed/p3_model1_dataset.csv"
if (!file.exists(data_path)) {
  stop(paste("Data file not found at:", data_path))
}

df <- read.csv(data_path)
cat(sprintf("Loaded dataset: %d rows across %d athletes\n\n", nrow(df), length(unique(df$athlete_id))))

# 3. Fit Model 1 via REML
# Formula: Readiness ~ zSleep(within) + Sleep(mean) + zRHR(within) + RHR(mean) + zLoad(within) + Load(mean) + Weekend + (1 | Athlete)
m1_r <- lmer(
  readiness ~ z_dev_prior_sleep_hours + mean_prior_sleep_hours +
              z_dev_prior_rhr + mean_prior_rhr +
              z_dev_prior_day_strain + mean_prior_day_strain +
              is_weekend + (1 | athlete_id),
  data = df,
  REML = TRUE
)

# 4. Display Model Summary, Optimizer & Convergence Info
cat("================================================================================\n")
cat("                       R lme4 / lmerTest Model 1 Output\n")
cat("================================================================================\n")
cat(sprintf("Optimizer used: %s\n", m1_r@optinfo$optimizer))
cat(sprintf("Convergence code: %d\n", m1_r@optinfo$conv$opt))
if (length(m1_r@optinfo$warnings) > 0) {
  cat("Optimizer warnings:\n")
  print(m1_r@optinfo$warnings)
} else {
  cat("Optimizer warnings: None (clean convergence)\n\n")
}
print(summary(m1_r))

# 5. Extract 95% Wald Confidence Intervals
cat("\nComputing 95% Wald Confidence Intervals (Normal Approximation)...\n")
conf_ints <- confint(m1_r, method = "Wald")
print(conf_ints)

# 6. Load Python results programmatically for direct side-by-side comparison
py_results_path <- "data/processed/p3_model1_py_results.csv"
if (!file.exists(py_results_path)) {
  stop(paste("Python results file not found at:", py_results_path))
}
py_res <- read.csv(py_results_path)

# Map term names between R and Python
fe_r <- fixef(m1_r)
se_r <- sqrt(diag(vcov(m1_r)))

comparison_df <- data.frame(
  Term = character(),
  R_Beta = numeric(),
  Py_Beta = numeric(),
  Delta_Beta = numeric(),
  R_SE = numeric(),
  Py_SE = numeric(),
  Delta_SE = numeric(),
  stringsAsFactors = FALSE
)

for (r_name in names(fe_r)) {
  py_name <- ifelse(r_name == "(Intercept)", "Intercept", r_name)
  py_row <- py_res[py_res$term == py_name, ]
  if (nrow(py_row) > 0) {
    r_b <- as.numeric(fe_r[r_name])
    py_b <- as.numeric(py_row$coef)
    r_se_val <- as.numeric(se_r[r_name])
    py_se_val <- as.numeric(py_row$std_err)
    comparison_df <- rbind(comparison_df, data.frame(
      Term = r_name,
      R_Beta = round(r_b, 4),
      Py_Beta = round(py_b, 4),
      Delta_Beta = round(abs(r_b - py_b), 6),
      R_SE = round(r_se_val, 4),
      Py_SE = round(py_se_val, 4),
      Delta_SE = round(abs(r_se_val - py_se_val), 6)
    ))
  }
}

cat("\n================================================================================\n")
cat("            Parameter Comparison: R lme4 vs Python statsmodels\n")
cat("================================================================================\n")
print(comparison_df, row.names = FALSE)

py_sleep_row <- py_res[py_res$term == "z_dev_prior_sleep_hours", ]
cat("\n[Verification Summary]:\n")
cat(sprintf("  Acute Sleep Deviation (z_dev_prior_sleep_hours):\n"))
cat(sprintf("    R lme4:          beta = +%.8f, SE = %.8f, 95%% CI = [%.8f, %.8f]\n",
            fe_r["z_dev_prior_sleep_hours"], se_r["z_dev_prior_sleep_hours"],
            conf_ints["z_dev_prior_sleep_hours", 1], conf_ints["z_dev_prior_sleep_hours", 2]))
cat(sprintf("    Py statsmodels:  beta = +%.8f, SE = %.8f, 95%% CI = [%.8f, %.8f]\n",
            py_sleep_row$coef, py_sleep_row$std_err, py_sleep_row$ci_lower, py_sleep_row$ci_upper))
delta_sleep_beta <- abs(fe_r["z_dev_prior_sleep_hours"] - py_sleep_row$coef)
cat(sprintf("    Numerical Diff:  |Delta beta| = %.8f (Direct subtraction delta)\n", delta_sleep_beta))
