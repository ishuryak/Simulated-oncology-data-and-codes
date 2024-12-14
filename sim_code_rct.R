
library(corrplot)
library(psych)
library(grf)
library(caret)
library(ggplot2)
library(fastshap)

# Set seed for reproducibility
set.seed(123)

# Number of patients
n <- 5000

# Simulate radiation oncology patient data

# Age: Normal distribution with mean 60 and standard deviation 10
Age <- rnorm(n, mean = 60, sd = 10)

# Sex: Bernoulli distribution (0 = female, 1 = male) with probability 0.5
Sex <- rbinom(n, 1, 0.5)

# Calendar_year: Normal distribution with mean 2010 and standard deviation 5
Calendar_year <- rnorm(n, mean = 2010, sd = 2)

# Stage: Categorical (1-4) with probabilities 20%, 20%, 40%, 20%
Stage <- sample(1:4, n, replace = TRUE, prob = c(0.2, 0.2, 0.4, 0.2))

# Smoking: Bernoulli distribution with probability 0.3 for males (Sex=1) and 0.1 for females (Sex=0)
Smoking <- ifelse(Sex == 1, rbinom(n, 1, 0.3), rbinom(n, 1, 0.1))

# BED (radiotherapy biologically effective dose): Normal distribution with mean 65 and standard deviation 3
BED <- rnorm(n, mean = 65, sd = 2)

# Chemotherapy:
# It is a random vector from the Bernoulli distribution with length n. 
# Here we assume a randomized trial situation with a constant 50% probability for each 
#  patient to get Chemotherapy. This mimics a randomized clinical trial.

# Generate the Chemotherapy vector
Chemotherapy <- rbinom(n, 1, 0.5)


#######
# Survival_time: Normal distribution with mean depending on treatments (Chemotherapy, BED) 
# and other features (Stage, Sex, Smoking, Calendar_year)

# Treatment effect for Chemotherapy
# Here we assume that the effect varies as function of age.
# It is larger for younger ages <=55, and smaller for older ages
Coef_Chemotherapy <- ifelse(Age<=55, 4, 2)

# This is the overall population average treatment effect
mean(Coef_Chemotherapy)
# 2.6132

# Treatment effect for radiotherapy BED
Coef_BED <- 1

# Implementing these effects on survival time
Survival_time <- pmax(rnorm(n, mean = 6 + Coef_Chemotherapy * Chemotherapy + Coef_BED * BED / 10 
                            - 1 * Stage - 0.2 * Sex - 0.5 * Smoking + 0.04 * (Calendar_year - 2000) / 10, sd = 7), 0.01)

summary(Survival_time)

cor(Survival_time, Chemotherapy)
cor(Survival_time, BED)


# Survival_status: Bernoulli distribution with probability 0.7 (allows censoring)
Survival_status <- rbinom(n, 1, 0.7)

# Combine simulated data into a data frame
oncology_data <- data.frame(
  Age = Age,
  Sex = Sex,
  Calendar_year = Calendar_year,
  Stage = Stage,
  Smoking = Smoking,
  BED = BED,
  Chemotherapy = Chemotherapy,
  Survival_time = Survival_time,
  Survival_status = Survival_status
)

# View the first few rows
head(oncology_data)

# Summary statistics
summary(oncology_data)

# export to a file
write.csv(oncology_data, "oncology_data.csv", row.names = FALSE)


# Plot Pearson correlation matrix of features
varscor_oncology_data_pearson <- corr.test(oncology_data, method = "pearson", adjust = "bonf", alpha = .05, ci = FALSE)
varscor_oncology_data_pearson_p <- varscor_oncology_data_pearson$p

# export to a file
write.csv(varscor_oncology_data_pearson_p, file = "varscor_oncology_data_pearson_p.csv", row.names = TRUE)
write.csv(varscor_oncology_data_pearson$r, file = "varscor_oncology_data_pearson_r.csv", row.names = TRUE)

# Export correlation matrix as PNG
png("correlation_matrix_pearson.png", width = 1000, height = 1000, res = 200)
corrplot(varscor_oncology_data_pearson$r, p.mat = varscor_oncology_data_pearson$p, method = 'circle', 
         tl.col = "black", type = "upper", sig.level = 0.05, pch.cex = 0.8, cl.cex = 0.8, tl.cex = 0.8, 
         insig = 'pch', pch = 19, pch.col = "white", diag = FALSE, font = 1,
         mar = c(0,0,1,0))
dev.off()


# Create training and testing sets by splitting data
set.seed(99)
index <- createDataPartition(oncology_data$Survival_status, p=0.75, list=FALSE)
trainSet <- oncology_data[index,]
testSet <- oncology_data[-index,]
nrow(trainSet)
nrow(testSet)


# Implement causal survival forest to estimate Chemotherapy treatment effect

# separate variable types
trainSet_X <- as.data.frame(subset(trainSet, select = -c(Survival_time, Survival_status, Chemotherapy)))
trainSet_W <- trainSet$Chemotherapy
trainSet_times <- trainSet$Survival_time
trainSet_events <- trainSet$Survival_status


testSet_X <- as.data.frame(subset(testSet, select = -c(Survival_time, Survival_status, Chemotherapy)))
testSet_W <- testSet$Chemotherapy
testSet_times <- testSet$Survival_time
testSet_events <- testSet$Survival_status

# select time horizons
n_trees_val <- 3000
horizons <- c(1*6, 2*6, 3*6, 4*6, 5*6, 6*6, 7*6, 8*6, 9*6, 10*6)
results_RMST <- data.frame(horizon_sel = integer(), ATE_estimate_train_RMST = numeric(), ATE_se_train_RMST = numeric(),
                           ATE_estimate_test_RMST = numeric(), ATE_se_test_RMST = numeric())

for (horizon in horizons) {
  csf_model_RMST <- causal_survival_forest(X = trainSet_X, Y = trainSet_times, W = trainSet_W,
                                           D = trainSet_events, 
                                           num.trees = n_trees_val, target = "RMST", 
                                           horizon = horizon, seed = 123)
  
  ate_train_RMST <- average_treatment_effect(csf_model_RMST)
  csf_pred_test_RMST <- predict(csf_model_RMST, testSet_X, estimate.variance = TRUE)
  ate_h_RMST_test <- mean(csf_pred_test_RMST$predictions)
  ate_h_RMST_test_sd <- mean(sqrt(csf_pred_test_RMST$variance.estimates))
  
  results_RMST <- rbind(results_RMST, data.frame(horizon_sel = horizon, 
                                                 ATE_estimate_train_RMST = ate_train_RMST[1], ATE_se_train_RMST = ate_train_RMST[2],
                                                 ATE_estimate_test_RMST = ate_h_RMST_test, ATE_se_test_RMST = ate_h_RMST_test_sd))
}

print(results_RMST)
write.csv(results_RMST, "train_and_test_ATE_RMST.csv", row.names = FALSE)


############
# Plot the results

# Convert horizons to years for better interpretation
results_RMST$horizon_years <- results_RMST$horizon_sel / 12


# Create the plot
rmst_plot <- ggplot(results_RMST, aes(x = horizon_years)) +
  geom_pointrange(aes(y = ATE_estimate_train_RMST, 
                      ymin = ATE_estimate_train_RMST - ATE_se_train_RMST,
                      ymax = ATE_estimate_train_RMST + ATE_se_train_RMST,
                      color = "Training"),
                  size = 0.8,
                  shape = 16,
                  position = position_nudge(x = -0.05),
                  fatten = 2) +
  geom_pointrange(aes(y = ATE_estimate_test_RMST,
                      ymin = ATE_estimate_test_RMST - ATE_se_test_RMST,
                      ymax = ATE_estimate_test_RMST + ATE_se_test_RMST,
                      color = "Testing"),
                  size = 0.8,
                  shape = 17,
                  position = position_nudge(x = 0.05),
                  fatten = 2) +
  scale_color_manual(values = c("Training" = "blue", "Testing" = "red")) +
  labs(x = "Time after treatment (Years)",
       y = "Average Treatment Effect (RMST, months)",
       color = "Dataset",
       title = "Estimated Treatment Effects",
       subtitle = "With Standard Errors") +
  theme_bw() +
  theme(
    text = element_text(size = 12),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.title = element_text(size = 12),
    legend.key.size = unit(0.5, "cm"),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    plot.subtitle = element_text(size = 12, hjust = 0.5),
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10, "pt")
  )

# Save the plot as a 1000x1000 pixel PNG
ggsave("rmst_results.png", rmst_plot, width = 5, height = 5, dpi = 200)


#############

# calculate SHAP values for causal forest at selected horizon
horizon_sel <- 3*12
forest_sel <- causal_survival_forest(X = trainSet_X, Y = trainSet_times, W = trainSet_W, 
                                     D = trainSet_events, num.trees = n_trees_val, 
                                     target = "RMST", horizon = horizon_sel, 
                                     tune.parameters = "all", seed=1234)


forest_sel_preds_train <- predict(forest_sel, trainSet_X, estimate.variance = TRUE)
forest_sel_preds_test <- predict(forest_sel, testSet_X, estimate.variance = TRUE)

forest_sel_preds_train_df <- cbind(trainSet_X, forest_sel_preds_train)
forest_sel_preds_test_df <- cbind(testSet_X, forest_sel_preds_test)

write.csv(forest_sel_preds_train_df, file = "sim_forest_sel_preds_train_df.csv", row.names = FALSE)
write.csv(forest_sel_preds_test_df, file = "sim_forest_sel_preds_test_df.csv", row.names = FALSE)



# Compute SHAP values
pfun <- function(object, newdata) {
  predict(object, newdata = newdata, estimate.variance = TRUE)$predictions
}

# these are approximate SHAP values calculated using a Monte Carlo method
# choose number of Monte Carlo repeats
nsim_shap <- 200

shap <- fastshap::explain(forest_sel, X = trainSet_X, pred_wrapper = pfun, nsim = nsim_shap)
colnames(shap) <- paste0(colnames(shap), "_SHAP")
shap_vals <- cbind(trainSet_X, shap, forest_sel_preds_train)

# Calculate the average prediction for all samples
average_prediction <- mean(shap_vals$predictions)

# Normalize the SHAP values so the Monte Carlo random error is removed as much as possible
# Step 1: Sum all the columns ending in _SHAP
shap_cols <- grep("_SHAP$", names(shap_vals), value = TRUE)
shap_vals$sum_SHAP <- rowSums(shap_vals[, shap_cols])

# Normalize each _SHAP column
for (col in shap_cols) {
  norm_col <- paste0(col, "_norm_SHAP")
  shap_vals[[norm_col]] <- (shap_vals[[col]] / shap_vals$sum_SHAP) * (shap_vals$predictions - average_prediction)
}

# Create a sum of the normalized SHAP columns
norm_shap_cols <- grep("_norm_SHAP$", names(shap_vals), value = TRUE)
shap_vals$sum_norm_SHAP <- rowSums(shap_vals[, norm_shap_cols])

# Check if sum of normalized SHAP columns equals the difference between predictions and average prediction
all.equal(shap_vals$sum_norm_SHAP, shap_vals$predictions - average_prediction)

write.csv(shap_vals, file = "sim_shap_vals_raw.csv", row.names = FALSE)


# Identify columns to remove
cols_to_remove <- grep("_SHAP$", names(shap_vals), value = TRUE)

# Identify columns to keep (those ending with _SHAP_norm_SHAP)
cols_to_keep <- grep("_SHAP_norm_SHAP$", names(shap_vals), value = TRUE)

# Remove the _SHAP_norm_SHAP columns from the list of columns to remove
cols_to_remove <- setdiff(cols_to_remove, cols_to_keep)

# Remove the identified columns
shap_vals <- shap_vals[, !(names(shap_vals) %in% cols_to_remove)]

# Identify columns ending with _SHAP_norm_SHAP
cols_to_rename <- grep("_SHAP_norm_SHAP$", names(shap_vals), value = TRUE)

# Create new names
new_names <- sub("_SHAP_norm_SHAP$", "_SHAP", cols_to_rename)

# Rename the columns
names(shap_vals)[names(shap_vals) %in% cols_to_rename] <- new_names

str(shap_vals)

# Identify all columns ending with _SHAP
shap_cols <- grep("_SHAP$", names(shap_vals), value = TRUE)

# Calculate the sum of all _SHAP columns
shap_sum <- rowSums(shap_vals[, shap_cols])

# Compare with the predictions column: theoretically they should correspond
is_equal <- all.equal(shap_sum, shap_vals$predictions, tolerance = 1e-6)

if (isTRUE(is_equal)) {
  print("The sum of all _SHAP columns equals the predictions column.")
} else {
  print("The sum of all _SHAP columns does NOT equal the predictions column.")
  
  # Calculate and print the maximum difference
  max_diff <- max(abs(shap_sum - shap_vals$predictions))
  print(paste("Maximum difference:", max_diff))
  
  # Optionally, you can print summary statistics of the differences 
  diff_summary <- summary(shap_sum - shap_vals$predictions)
  print("Summary of differences:")
  print(diff_summary)
}


write.csv(shap_vals, file = "sim_shap_vals.csv", row.names = FALSE)

# plot age effects for interest

library(devEMF)


# Select columns ending with _SHAP
shap_cols <- grep("_SHAP$", names(shap_vals), value = TRUE)

# Calculate median values for these _SHAP columns
medians <- apply(shap_vals[shap_cols], 2, median)

# Calculate median of absolute values for these _SHAP columns
abs_medians <- apply(abs(shap_vals[shap_cols]), 2, median)

# Combine results into a data frame
shap_vals_medians <- data.frame(medians, abs_medians)
write.csv(shap_vals_medians, file = "sim_shap_vals_medians.csv", row.names = TRUE)

# Sort shap_vals_medians by abs_medians in descending order
shap_vals_medians <- shap_vals_medians[order(-shap_vals_medians$abs_medians), ]

# Save the top 10
shap_vals_medians_top_10 <- head(shap_vals_medians, 10)
write.csv(shap_vals_medians_top_10, file = "sim_shap_vals_medians_top_10.csv", row.names = TRUE)

# Plot correlations between top SHAP values
std_devs <- apply(shap_vals, 2, sd)
cols <- names(std_devs)[std_devs > 0]
selected_shap_vals <- shap_vals[, cols]

varscor_selected_shap_vals <- corr.test(selected_shap_vals, method = "spearman", adjust = "bonf", alpha = .05, ci = FALSE)
varscor_selected_shap_vals_p <- varscor_selected_shap_vals$p

write.csv(varscor_selected_shap_vals_p, file = "varscor_selected_shap_vals_spearman_p.csv", row.names = TRUE)
write.csv(varscor_selected_shap_vals$r, file = "varscor_selected_shap_vals_spearman_r.csv", row.names = TRUE)

# Plot Spearman correlation matrix
emf("correlation_matrix_spearman_SHAP.emf", width = 7, height = 7, bg = "transparent", fg = "black", pointsize = 8, family = "Arial", coordDPI = 600)

corrplot(varscor_selected_shap_vals$r, p.mat = varscor_selected_shap_vals$p, method = 'circle', tl.col = "black", type = "upper", sig.level = 0.05, pch.cex = 0.6, cl.cex = 1, tl.cex = 1, insig = 'pch', pch = 19, pch.col = "white", diag = FALSE, font = 1)

dev.off()

# Plot Pearson correlation matrix
varscor_selected_shap_vals_pearson <- corr.test(selected_shap_vals, method = "pearson", adjust = "bonf", alpha = .05, ci = FALSE)
varscor_selected_shap_vals_pearson_p <- varscor_selected_shap_vals_pearson$p

write.csv(varscor_selected_shap_vals_pearson_p, file = "varscor_selected_shap_vals_pearson_p.csv", row.names = TRUE)
write.csv(varscor_selected_shap_vals_pearson$r, file = "varscor_selected_shap_vals_pearson_r.csv", row.names = TRUE)

emf("correlation_matrix_pearson_SHAP.emf", width = 7, height = 7, bg = "transparent", fg = "black", pointsize = 8, family = "Arial", coordDPI = 600)

corrplot(varscor_selected_shap_vals_pearson$r, p.mat = varscor_selected_shap_vals_pearson$p, method = 'circle', tl.col = "black", type = "upper", sig.level = 0.05, pch.cex = 0.6, cl.cex = 1, tl.cex = 1, insig = 'pch', pch = 19, pch.col = "white", diag = FALSE, font = 1)

dev.off()


# Create dummy data for the red line legend entry
dummy_data <- data.frame(Age = selected_shap_vals$Age[1], 
                         predictions = selected_shap_vals$predictions[1])

emf("treatment_effects_vs_Age.emf", width = 7, height = 7, bg = "transparent", fg = "black", pointsize = 8, family = "Arial", coordDPI = 600)

ggplot(selected_shap_vals, aes(x = Age, y = predictions)) +
  geom_point(aes(color = "Treatment effect estimates"), alpha = 0.4) +
  geom_smooth(aes(color = "LOESS fit"), method = "loess", se = FALSE, span = 0.95, linewidth = 1) +
  annotate("segment", x = 20, xend = 55, y = 4, yend = 4, color = "red", linewidth = 1) +
  annotate("segment", x = 55, xend = max(selected_shap_vals$Age), y = 2, yend = 2, color = "red", linewidth = 1) +
  # Create dummy data with two points for the red line
  geom_line(data = data.frame(
    Age = c(20, 21),
    predictions = c(0, 0),
    group = 1
  ), aes(color = "True treatment effect", group = group)) +
  labs(x = "Age (years)", y = "Treatment effect (RMST, months)") +
  theme(
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),  # Larger axis numbers
    legend.position = "top"
  ) +
  xlim(20, 100) +
  ylim(-0.5, 6) +
  scale_color_manual(
    values = c(
      "Treatment effect estimates" = "gray",
      "LOESS fit" = "blue",
      "True treatment effect" = "red"
    ),
    guide = guide_legend(
      override.aes = list(
        linetype = c("blank", "solid", "solid"),
        shape = c(16, NA, NA),
        size = c(2, 1, 1),  # Adjust point and line sizes in legend
        alpha = c(0.4, 1, 1)
      )
    )
  ) +
  theme(legend.title = element_blank())

dev.off()


# graph the SHAPs

# correlation matrix of features and SHAPs
# remove causal effects - only keep features and corresponding SHAP values
SHAP_data_1 <- as.data.frame(subset(shap_vals,
                                    select=-c(predictions, variance.estimates)))

# identify feature columns
features <- names(SHAP_data_1)[!grepl("_SHAP$", names(SHAP_data_1))]

# causal effects vector
causal_effects <- shap_vals$predictions

for (feature in features) {
  # Create a filename for each feature
  emf_filename <- paste0(feature, "_SHAP_plot.emf")
  
  # Open the EMF device
  emf(emf_filename, width = 10, height = 6)  # Adjust width and height as needed
  
  # Create the plot
  p <- ggplot(SHAP_data_1, aes(x = .data[[feature]], y = .data[[paste0(feature, "_SHAP")]])) +
    geom_point(aes(color = causal_effects, shape = factor(Sex)), size = 3) +
    scale_shape_manual(values = c(16, 15)) +
    scale_color_gradient2(low = "blue", mid = "white", high = "red", midpoint = median(causal_effects)) +
    labs(title = paste("SHAP values for", feature),
         x = feature,
         y = paste0(feature, "_SHAP"),
         color = "Causal Effects",
         shape = "Sex") +
    theme_minimal() +
    theme(
      axis.text.x = element_text(size = 14, face = "bold"),  # Change size and make x-axis text bold
      axis.text.y = element_text(size = 14, face = "bold"),  # Change size and make y-axis text bold
      axis.title.x = element_text(size = 16, face = "bold"), # Make x-axis title bold and larger
      axis.title.y = element_text(size = 16, face = "bold")  # Make y-axis title bold and larger
    ) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +  # Rotate x-axis labels if needed
    coord_cartesian(ylim = c(-3, 3))
  
  # Print the plot to the EMF device
  print(p)
  
  # Close the EMF device to save the file
  dev.off()
}




