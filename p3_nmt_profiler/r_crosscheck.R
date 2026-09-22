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

# 4. Display Model Summary & Satterthwaite p-values
cat("================================================================================\n")
cat("                       R lme4 / lmerTest Model 1 Output\n")
cat("================================================================================\n")
print(summary(m1_r))

# 5. Extract 95% Profile Confidence Intervals
cat("\nComputing 95% Wald / Profile Confidence Intervals...\n")
conf_ints <- confint(m1_r, method = "Wald")
print(conf_ints)

# 6. Structured Comparison Table (R lme4 vs Python statsmodels)
cat("\n================================================================================\n")
cat("            Parameter Comparison: R lme4 vs Python statsmodels\n")
cat("================================================================================\n")
fe_r <- fixef(m1_r)
se_r <- sqrt(diag(vcov(m1_r)))

results_table <- data.frame(
  Term = names(fe_r),
  R_Estimate = round(fe_r, 4),
  R_StdErr = round(se_r, 4),
  R_CI_Lower = round(conf_ints[names(fe_r), 1], 4),
  R_CI_Upper = round(conf_ints[names(fe_r), 2], 4)
)

print(results_table, row.names = FALSE)

cat("\n[Verification Summary]:\n")
cat(sprintf("  Acute Sleep Deviation (z_dev_prior_sleep_hours):\n"))
cat(sprintf("    R lme4:          beta = +%.4f, SE = %.4f, 95%% CI = [%.4f, %.4f]\n",
            fe_r["z_dev_prior_sleep_hours"], se_r["z_dev_prior_sleep_hours"],
            conf_ints["z_dev_prior_sleep_hours", 1], conf_ints["z_dev_prior_sleep_hours", 2]))
cat(sprintf("    Py statsmodels:  beta = +0.2183, SE = 0.0403, 95%% CI = [0.1394, 0.2972]\n"))
diff_beta <- abs(fe_r["z_dev_prior_sleep_hours"] - 0.21828)
cat(sprintf("    Numerical Diff:  |Delta beta| = %.6f\n", diff_beta))
