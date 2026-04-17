# =============================================================================
# ASSIGNMENT 1: CUSTOMER CHURN PREDICTION
# Data Science Methods
# =============================================================================


# =============================================================================
# 1. SETUP & PACKAGE LOADING
# =============================================================================
library(tidyverse)    # Core Tidyverse (dplyr, ggplot2, tidyr, readr)
library(ROCR)         # For Gini/AUC calculation
prediction  <- ROCR::prediction
performance <- ROCR::performance

library(psych)        # For describe()
library(fastDummies)  # For One-Hot Encoding
library(robustHD)     # For Winsorizing
library(e1071)        # For skewness/kurtosis, and SVM
library(patchwork)    # For combining plots
library(scales)       # For axis formatting
library(randomForest) # For Random Forest
library(MASS)         # For Stepwise Regression
library(neuralnet)    # For ANN
library(rpart)        # For CART
library(partykit)     # For CART visualization
library(ipred)        # For Bagging
library(gbm)          # For Boosting (Requires Churn to be 0/1 numeric)
library(ggplot2)
library(caret)
library(corrplot)
library(rpart.plot)
# Set random seed for reproducibility (used for data split later)
set.seed(42)


# =============================================================================
#2. DATA IMPORT & INITIAL EXPLORATION
# =============================================================================

# 2. DATA IMPORT & INITIAL INSPECTION -----------------------------------------
# NOTE: Update the file path as necessary
data <- read.csv(
  "/Users/nhubui/Documents/SEM 1B/Data Science for MADS/Assignment/data assignment 1.csv",
  na.strings = c("", "NULL")
)

#Initial Data Summary
summary(data)
describe(data)
# =============================================================================
# 3.1 CHECK MISSING VALUES
# =============================================================================
missing_summary <- data.frame(
  Variable = names(data),
  Missing_Count = colSums(is.na(data)),
  Missing_Percent = round(colSums(is.na(data)) / nrow(data) * 100, 4)
)

missing_summary <- missing_summary %>%
  arrange(desc(Missing_Count))


print(missing_summary, row.names = FALSE)
# Define continuous variables for outlier analysis
continuous_vars <- c("Age", "Income", "Relation_length", "Contract_length",
                     "Home_age", "Electricity_usage", "Gas_usage")

# =============================================================================
# 3. OUTLIER DETECTION & TREATMENT
# =============================================================================
# IQR FUNCTION
detect_outliers_iqr <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR_val
  upper_bound <- Q3 + 1.5 * IQR_val
  return(x < lower_bound | x > upper_bound)
}

# Create Outlier Summary Table
outlier_summary <- data.frame(
  Variable = continuous_vars,
  Pct_Outliers_IQR = sapply(data[continuous_vars], function(x) round(100 * sum(detect_outliers_iqr(x)) / length(x), 2))
)
print("--- 3. Outlier Detection Summary (IQR %) ---")
print(outlier_summary)

# 3.1. WINDSORIZING (P1/P99) for Age, Home_age, Relation_length
vars_to_winsorize <- c("Age", "Relation_length", "Home_age", "Income", "Electricity_usage", "Gas_usage")

for (var_name in vars_to_winsorize) {
  lower_limit <- quantile(data[[var_name]], 0.01, na.rm = TRUE)
  upper_limit <- quantile(data[[var_name]], 0.99, na.rm = TRUE)
  new_col_name <- paste0(var_name, "_wins")
  
  data[[new_col_name]] <- robustHD::winsorize(
    data[[var_name]], 
    min.value = lower_limit, 
    max.value = upper_limit
  )
}
  

# 3.3. BINARIZATION for Contract_length (0 = Flexible)
data <- data %>%
  mutate(
    flexible_contract = as.factor(if_else(Contract_length == 0, 1, 0)) # 1 = Flexible, 0 = Fixed
  )

# 3.4. CLEAN CATEGORICAL NAMES BEFORE OHE 
data <- data %>%
  mutate(
    Province = gsub(" |-|\\.", "_", Province),
    Home_label = gsub(" |-|\\.", "_", Home_label),
    
    # CRITICAL FIX FOR CHI-SQ: Ensure original categorical variables are factors
    Start_channel = as.factor(Start_channel),
    Home_label = as.factor(Home_label),
    Province = as.factor(Province),
    Gender = as.factor(Gender),
    Email_list = as.factor(Email_list)
  )

# 3.5. One-hot encoding

# One-Hot Encoding for Categorical Variables
cat_cols_to_encode <- intersect(c("Start_channel", "Home_label", "Province"), colnames(data))
data <- fastDummies::dummy_cols(data,
                                select_columns = cat_cols_to_encode,
                                remove_first_dummy = TRUE, 
                                remove_selected_columns = FALSE) # KEEP ORIGINAL COLUMNS FOR CHI-SQ


# Prepare Churn target variable and numeric Churn for correlation
data$Churn <- factor(data$Churn)
data$Churn_num <- as.numeric(as.character(data$Churn))

# Define processed continuous variables for 5.1.1 and 5.3.1
continuous_processed_vars <- c("Age_wins", "Relation_length_wins",
                               "Home_age_wins", "Income_wins",
                               "Electricity_usage_wins", "Gas_usage_wins")

# Define key categorical variables for 5.1.2 and 5.3.2
categorical_vars <- c("Gender", "Email_list", "flexible_contract", 
                      "Start_channel", "Home_label", "Province")

# =============================================================================
# 4. DESCRIPTIVE & BIVARIATE ANALYSIS
# =============================================================================
# Continuous/Ratio Variables: 
data_processed_cont <- data[ , continuous_processed_vars]
print(describe(data_processed_cont))

# Categorical/Dummy Variables:
# Overall Churn Rate (Baseline)
overall_churn_rate <- mean(data$Churn_num)
cat(paste("Overall Churn Rate (Baseline):", round(overall_churn_rate * 100, 2), "%\n"))

# Frequencies for key categorical variables (Gender, Email_list, Contract, Channel, Home_label, Province)
categorical_vars_list <- c("Gender", "Email_list", "flexible_contract", "Start_channel", "Home_label", "Province")
plot_list_bar <- list()

for (var_name in categorical_vars_list) {
  
  # A. Print Frequency Table
  freq_table <- table(data[[var_name]]) / nrow(data) * 100
  print(paste("\nFrequency Table (%) for:", var_name))
  print(freq_table)
  
  # B. Generate Bivariate Bar Plot (Distribution by Churn Status)
  df_plot <- data %>% 
    group_by(!!sym(var_name), Churn) %>% 
    summarise(Count = n(), .groups = 'drop') %>%
    # Calculate the proportion of Churn/Non-Churn within the total observation count
    group_by(!!sym(var_name)) %>% 
    mutate(Proportion = Count / sum(Count)) %>%
    ungroup()
  
  p <- ggplot(df_plot, aes(x = !!sym(var_name), y = Proportion, fill = Churn)) +
    # Use position="fill" to stack and position="dodge" to compare side-by-side
    geom_bar(stat = "identity", position = "fill") +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = paste("Churn Proportion by", var_name),
      x = var_name,
      y = "Proportion",
      fill = "Churn Status (0=Stay, 1=Churn)"
    ) +
    scale_fill_manual(values = c("0" = "grey", "1" = "pink")) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  plot_list_bar[[var_name]] <- p
}
print("\n--- BAR CHART VISUALIZATIONS (5.1.2 / 5.3.2) ---")
# Print bar chart
print(plot_list_bar[["flexible_contract"]])
print(plot_list_bar[["Gender"]]) 
print(plot_list_bar[["Home_label"]]) 
print(plot_list_bar[["Province"]])
print(plot_list_bar[["Start_channel"]])
print(plot_list_bar[["Email_list"]]) 

ggplot(data, aes(x = factor(Churn), y = Income_wins)) +
  geom_boxplot(fill = "pink")

ggplot(data, aes(x = factor(Churn), y = Electricity_usage_wins)) +
  geom_boxplot(fill = "pink")

ggplot(data, aes(x = factor(Churn), y = Gas_usage_wins)) +
  geom_boxplot(fill = "pink")

# Skewness improvement table
# Define all 6 variables for comparison (Original and Processed)
comparison_list <- list(
  c("Age", "Age_wins"),
  c("Relation_length", "Relation_length_wins"),
  c("Home_age", "Home_age_wins"),
  c("Income", "Income_wins"),
  c("Electricity_usage", "Electricity_usage_wins"),
  c("Gas_usage", "Gas_usage_wins")
)

skewness_report <- data.frame()

for (pair in comparison_list) {
  orig_col <- pair[1]
  proc_col <- pair[2]
  
  orig_skew <- e1071::skewness(data[[orig_col]], na.rm = TRUE)
  proc_skew <- e1071::skewness(data[[proc_col]], na.rm = TRUE)
  
  # Append to report table
  skewness_report <- skewness_report %>%
    bind_rows(data.frame(
      Variable = orig_col,
      Skewness_Original = round(orig_skew, 2),
      Skewness_Processed = round(proc_skew, 2),
      Change_Pct = round(abs(proc_skew - orig_skew) / abs(orig_skew) * 100, 2)
    ))
}

print("--- 5.2 Skewness Comparison (Effectiveness of Outlier Treatment) ---")
print(skewness_report)


# =============================================================================
# 5.3 BIVARIATE ANALYSIS
# =============================================================================
# 5.3.1 Continuous Variables vs Churn (Correlation Matrix)
corr_matrix <- cor(data[ , continuous_processed_vars], use = "complete.obs")
print(corr_matrix)
# Visualize Correlation Matrix
corrplot(corr_matrix, 
         method = "circle", 
         type = "upper", 
         order = "hclust")

# Statistical tests
t.test(Age ~ Churn, data = data)
t.test(Income_wins ~ Churn, data = data)
t.test(Relation_length_wins ~ Churn, data = data)
t.test(Electricity_usage_wins ~ Churn, data = data)
t.test(Gas_usage_wins ~ Churn, data = data)
t.test(Home_age_wins ~ Churn, data = data)

#5.3.2 Categorical/Dummy Variables:
chisq.test(table(data$Gender, data$Churn))
chisq.test(table(data$Start_channel, data$Churn))
chisq.test(table(data$Email_list, data$Churn))
chisq.test(table(data$Province, data$Churn))
chisq.test(table(data$flexible_contract, data$Churn))
chisq.test(table(data$Home_label, data$Churn))

# =============================================================================
# 5. TRAIN-VALIDATION SPLIT (75%/25%)
# =============================================================================
# OUT-OF-SAMPLE VALIDATION SETUP -------------------------------------------
# Remove redundant original continuous columns and ID
cols_to_remove <- c("Income", "Contract_length", "Age", "Relation_length", 
                    "Home_age", "Electricity_usage", "Gas_usage", "Customer_ID",
                    "Start_channel", "Home_label", "Province", "Churn_num")

data <- data[ , -which(names(data) %in% cols_to_remove)]

# 75% estimation sample, 25% validation sample
data$estimation_sample <- rbinom(nrow(data), 1, 0.75)
estimation_dataset <- subset(data, estimation_sample == 1)
validation_dataset  <- subset(data, estimation_sample == 0)
 
# =============================================================================
# 6. PREDICTIVE MODELING
# =============================================================================

# ------------------------------------------------------------------
# 6.1 Baseline Logistic Regression (hypothesis-driven variables only)
# ------------------------------------------------------------------
Logistic_regression1 <- glm(
  Churn ~ Income_wins + Relation_length_wins + flexible_contract + 
    Gas_usage_wins + Electricity_usage_wins,
  family = binomial,
  data = estimation_dataset
)

summary(Logistic_regression1)

predictions_model1 <- predict(Logistic_regression1, 
                              newdata = validation_dataset, 
                              type = "response")

# Hit-rate
predicted_class1 <- ifelse(predictions_model1 > 0.5, 1, 0)
hit_table1 <- table(validation_dataset$Churn, predicted_class1,
                    dnn = c("Observed", "Predicted"))
hit_rate1 <- sum(diag(hit_table1)) / sum(hit_table1)
cat("Baseline Hit-rate:", round(hit_rate1, 4), "\n")

# Top-decile lift
decile1 <- ntile(predictions_model1, 10)
decile_table1 <- table(validation_dataset$Churn, decile1)
top_decile_lift1 <- (decile_table1[2,10] / sum(decile_table1[,10])) / mean(as.numeric(as.character(validation_dataset$Churn)))
cat("Baseline Top-decile lift:", round(top_decile_lift1, 3), "\n")

# GINI COEFFICIENT (AUC) ---------------------------------------------------
pred_obj1 <- prediction(predictions_model1, validation_dataset$Churn)
gini1 <- 2 * performance(pred_obj1, "auc")@y.values[[1]] - 1
cat("Baseline Gini:", round(gini1, 4), "\n")

# ------------------------------------------------------------------
# 6.2 Step-wise Logistic
# ------------------------------------------------------------------
# Estimate full model and null model on estimation sample
Logistic_regression_full <- glm(Churn ~ . - estimation_sample,
                                data = estimation_dataset,
                                family = binomial)

Logistic_regression_both <- stepAIC(Logistic_regression_full, 
                                    direction="both", 
                                    trace = FALSE)
summary(Logistic_regression_both)

# 4. Backward với BIC 
n_est <- nrow(estimation_dataset)
Logistic_regression_backward_BIC <- stepAIC(Logistic_regression_full,
                                            direction = "backward",
                                            trace = FALSE,
                                            k = log(n_est)) # k = log(n) → BIC
summary(Logistic_regression_backward_BIC)
# Choose Logistic_regression_both
step_model_final <- Logistic_regression_both
cat("\nFinal selected step-wise model (both directions):\n")
print(summary(step_model_final))
cat("Number of variables in final model:", length(coef(step_model_final)) - 1, "\n\n")
# Prediction on validation set
predictions_step <- predict(step_model_final,
                            newdata = validation_dataset,
                            type = "response")

# Hit-rate
pred_class_step <- ifelse(predictions_step > 0.5, 1, 0)
hit_table_step <- table(validation_dataset$Churn, pred_class_step,
                        dnn = c("Observed", "Predicted"))
hit_rate_step <- sum(diag(hit_table_step)) / sum(hit_table_step)
cat("Step-wise Hit-rate:", round(hit_rate_step, 4), "\n")

# Top-decile lift
decile_step <- ntile(predictions_step, 10)
decile_table_step <- table(validation_dataset$Churn, decile_step)
top10_rate_step <- decile_table_step[2,10] / sum(decile_table_step[,10])
overall_rate <- mean(as.numeric(as.character(validation_dataset$Churn)))
top_decile_lift_step <- top10_rate_step / overall_rate
cat("Step-wise Top-decile lift:", round(top_decile_lift_step, 3), "\n")

# Gini
pred_obj_step <- prediction(predictions_step, validation_dataset$Churn)
gini_step <- 2 * performance(pred_obj_step, "auc")@y.values[[1]] - 1
cat("Step-wise Gini:", round(gini_step, 4), "\n\n")

# ------------------------------------------------------------------
# 6.3 CART Tree
# ------------------------------------------------------------------
cart_control <- rpart.control(
  minsplit = 50,
  minbucket = 20,
  cp = 0.0005,      
  maxdepth = 12,  
  xval = 10
)
cart_model <- rpart(Churn ~ ., 
                    data = estimation_dataset,
                    method = "class",
                    control = cart_control,
                    parms = list(split = "gini"))

# Visualize
rpart.plot(cart_model, main = "CART Decision Tree for Customer Churn", 
           extra = 104, box.palette = "RdYlGn", shadow.col = "gray", nn = TRUE)

# Predict 
predictions_cart <- predict(cart_model, validation_dataset, type = "prob")[,2]

# 1. HIT-RATE
predicted_class <- ifelse(predictions_cart > 0.5, 1, 0)
conf_matrix <- table(Observed = validation_dataset$Churn, Predicted = predicted_class)
hit_rate_cart <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("CART Hit-rate:", round(hit_rate_cart, 4), "\n")

# Top-decile lift
decile <- ntile(predictions_cart, 10)
top10_rate <- mean(validation_dataset$Churn[decile == 10] == 1)
overall_rate <- mean(validation_dataset$Churn == 1)
top_decile_lift <- top10_rate / overall_rate
cat("CART Top-decile lift:", round(top_decile_lift, 3), "\n")

# Gini
pred_rocr <- prediction(predictions_cart, validation_dataset$Churn)
auc_val <- performance(pred_rocr, "auc")@y.values[[1]]
gini_cart <- 2 * auc_val - 1
cat("CART Gini:", round(gini_cart, 4), "\n")

# ------------------------------------------------------------------
# 6.4 BAGGING
# ------------------------------------------------------------------
bagging_model <- bagging(Churn ~ .,
                         data = estimation_dataset,
                         nbagg = 500, 
                         coob = TRUE,
                         control = rpart.control(minsplit = 5, cp = 0))
print(bagging_model)

#calculate variable importance...
pred.imp <- varImp(bagging_model)
pred.imp

# Plot variable importance
barplot(pred.imp$Overall, names.arg = row.names(pred.imp))

# Predictions on validation set (probability of Churn = 1)
predictions_bagging <- predict(bagging_model, newdata = validation_dataset, type = "prob")[,2]

# 1. Hit-rate (Accuracy)
predicted_class_bag <- ifelse(predictions_bagging > 0.5, "1", "0")
hit_table_bag <- table(Observed = validation_dataset$Churn, 
                       Predicted = predicted_class_bag,
                       dnn = c("Observed", "Predicted"))
hit_rate_bag <- sum(diag(hit_table_bag)) / sum(hit_table_bag)
cat("\nBagging Hit-rate:", round(hit_rate_bag, 4), "\n")

# 2. Top-decile lift
decile_bag <- ntile(predictions_bagging, 10)
decile_table_bag <- table(validation_dataset$Churn, decile_bag)
top10_churn_rate <- decile_table_bag[2, 10] / sum(decile_table_bag[, 10])
overall_churn_rate <- mean(as.numeric(as.character(validation_dataset$Churn)))
top_decile_lift_bag <- top10_churn_rate / overall_churn_rate
cat("Bagging Top-decile lift:", round(top_decile_lift_bag, 3), "\n")

# 3. Gini coefficient
pred_obj_bag <- prediction(predictions_bagging, validation_dataset$Churn)
auc_bag <- performance(pred_obj_bag, "auc")@y.values[[1]]
gini_bag <- 2 * auc_bag - 1
cat("Bagging Gini:", round(gini_bag, 4), "\n\n")


# ------------------------------------------------------------------
# 6.5 BOOSTING
# ------------------------------------------------------------------
est_gbm <- estimation_dataset
est_gbm$Churn <- as.numeric(as.character(est_gbm$Churn))

boosting_model <- gbm(Churn ~ ., 
                      data = est_gbm,
                      distribution = "bernoulli",
                      n.trees = 10000,
                      interaction.depth = 4,
                      shrinkage = 0.01,
                      cv.folds = 5)

boosting_model

best_iter <- gbm.perf(boosting_model, method = "cv")
summary(boosting_model, n.trees = best_iter)

# In variable importance + plot
cat("\n--- Top 10 Most Important Variables (GBM) ---\n")
var_importance <- summary(boosting_model, n.trees = best_iter, plotit = FALSE)
print(head(var_importance, 10))

summary(boosting_model, n.trees = best_iter, 
        main = "Variable Importance - Gradient Boosting Model")

predictions_boosting <- predict(boosting_model, newdata = validation_dataset, 
                                n.trees = best_iter, type = "response")

# 1. Hit-rate (Accuracy)
predicted_class_gbm <- ifelse(predictions_boosting > 0.5, "1", "0")
hit_table_gbm <- table(Observed = validation_dataset$Churn, 
                       Predicted = predicted_class_gbm,
                       dnn = c("Observed", "Predicted"))
hit_rate_gbm <- sum(diag(hit_table_gbm)) / sum(hit_table_gbm)
cat("\nGBM Hit-rate:", round(hit_rate_gbm, 4), "\n")

# 2. Top-decile lift
decile_gbm <- ntile(predictions_boosting, 10)
top10_churn_rate <- mean(validation_dataset$Churn[decile_gbm == 10] == "1")
overall_churn_rate <- mean(validation_dataset$Churn == "1")
top_decile_lift_gbm <- top10_churn_rate / overall_churn_rate
cat("GBM Top-decile lift:", round(top_decile_lift_gbm, 3), "\n")

# 3. Gini coefficient
pred_obj_gbm <- prediction(predictions_boosting, validation_dataset$Churn)
auc_gbm <- performance(pred_obj_gbm, "auc")@y.values[[1]]
gini_gbm <- 2 * auc_gbm - 1
cat("GBM Gini:", round(gini_gbm, 4), "\n\n")

# ------------------------------------------------------------------
# 6.6 RANDOM FOREST
# ------------------------------------------------------------------
rf_model <- randomForest(Churn ~ ., 
                         data = estimation_dataset,
                         ntree = 800,
                         importance = TRUE)

#OOB error (out-of-bag)
cat("Random Forest OOB error rate:", round(rf_model$err.rate[nrow(rf_model$err.rate), "OOB"], 4), "\n")

#Top 15 Most Important Variables (Random Forest) 
importance_rf <- importance(rf_model, type = 2)  # type=2 = MeanDecreaseGini
print(head(importance_rf[order(importance_rf, decreasing = TRUE), , drop = FALSE], 15))

varImpPlot(rf_model, 
           type = 2, 
           main = "Variable Importance – Random Forest (Mean Decrease Gini)",
           n.var = 15,
           col = "steelblue")

# Prediction on validation set
predictions_rf <- predict(rf_model, 
                          newdata = validation_dataset, 
                          type = "prob")[, "1"] 

# 1. Hit-rate (Accuracy)
predicted_class_rf <- ifelse(predictions_rf > 0.5, "1", "0")
hit_table_rf <- table(Observed = validation_dataset$Churn,
                      Predicted = predicted_class_rf,
                      dnn = c("Observed", "Predicted"))
hit_rate_rf <- sum(diag(hit_table_rf)) / sum(hit_table_rf)
cat("\nRandom Forest Hit-rate:", round(hit_rate_rf, 4), "\n")

# 2. Top-decile lift
decile_rf <- ntile(predictions_rf, 10)
top10_churn_rate_rf <- mean(validation_dataset$Churn[decile_rf == 10] == "1")
overall_churn_rate <- mean(validation_dataset$Churn == "1")
top_decile_lift_rf <- top10_churn_rate_rf / overall_churn_rate
cat("Random Forest Top-decile lift:", round(top_decile_lift_rf, 3), "\n")

# 3. Gini coefficient
pred_obj_rf <- prediction(predictions_rf, validation_dataset$Churn)
auc_rf <- performance(pred_obj_rf, "auc")@y.values[[1]]
gini_rf <- 2 * auc_rf - 1
cat("Random Forest Gini:", round(gini_rf, 4), "\n\n")

# ------------------------------------------------------------------
# 6.7 SUPPORT VECTOR MACHINE
# ------------------------------------------------------------------
cat("=== 6.7 Support Vector Machine (Radial) ===\n")
# Scale for continuous 
scaled_est <- estimation_dataset
scaled_val <- validation_dataset

cont_vars <- c("Age_wins", "Relation_length_wins", "Home_age_wins",
               "Income_wins", "Electricity_usage_wins", "Gas_usage_wins")

for(v in cont_vars) {
  mu <- mean(scaled_est[[v]], na.rm = TRUE)
  sigma <- sd(scaled_est[[v]], na.rm = TRUE)
  scaled_est[[v]] <- (scaled_est[[v]] - mu) / sigma
  scaled_val[[v]] <- (scaled_val[[v]] - mu) / sigma
}

svm_model <- svm(Churn ~ ., 
                 data = scaled_est,
                 type = "C-classification",
                 kernel = "radial", 
                 cost = 10,
                 probability = TRUE)
summary(svm_model)

# Prediction on validation set
pred_svm_obj <- predict(svm_model, scaled_val, probability = TRUE)
predictions_svm <- attr(pred_svm_obj, "probabilities")[,"1"]

# Evaluation
predicted_class_svm <- ifelse(predictions_svm > 0.5, 1, 0)
# 1. Hit-rate (Accuracy)
hit_rate_svm <- mean(predicted_class_svm == validation_dataset$Churn)
cat("SVM Hit-rate:", round(hit_rate_svm, 4), "\n")
# 2. Top-decile lift
decile_svm <- ntile(predictions_svm, 10)
top10_churn_rate <- mean(validation_dataset$Churn[decile_svm == 10] == 1)
overall_churn_rate <- mean(validation_dataset$Churn == 1)
top_decile_lift_svm <- top10_churn_rate / overall_churn_rate
cat("SVM Top-decile lift:", round(top_decile_lift_svm, 3), "\n")
# 3. Gini coefficient
pred_obj <- prediction(predictions_svm, validation_dataset$Churn)
auc <- performance(pred_obj, "auc")@y.values[[1]]
gini_svm <- 2 * auc - 1
cat("SVM Gini:", round(gini_svm, 4), "\n\n")

# ------------------------------------------------------------------
# 6.7 ARTIFICIAL NEURAL NETWORKS
# ------------------------------------------------------------------
set.seed(42)

# Scaling for continuous variable
cont_vars <- c("Age_wins", "Relation_length_wins", "Home_age_wins",
               "Income_wins", "Electricity_usage_wins", "Gas_usage_wins")

ann_train <- estimation_dataset
ann_valid <- validation_dataset

for (v in cont_vars) {
  mu    <- mean(ann_train[[v]], na.rm = TRUE)
  sigma <- sd(ann_train[[v]],   na.rm = TRUE)
  ann_train[[v]] <- (ann_train[[v]] - mu) / sigma
  ann_valid[[v]] <- (ann_valid[[v]] - mu) / sigma
}

# Conver Churn into 0/1
ann_train$Churn01 <- as.numeric(as.character(ann_train$Churn))
ann_valid$Churn01  <- as.numeric(as.character(ann_valid$Churn))

# Convert dummy (as factor) into numeric 0/1
ann_train <- ann_train %>% mutate(across(where(is.factor), ~ as.numeric(.) - 1))
ann_valid <- ann_valid %>% mutate(across(where(is.factor), ~ as.numeric(.) - 1))

# Create formula
ann_formula <- as.formula(paste("Churn01 ~", 
                                paste(setdiff(names(ann_train), 
                                              c("Churn", "Churn01", "estimation_sample")), 
                                      collapse = " + ")))
# Run neuralnet
ann_model <- neuralnet(ann_formula,
                       data = ann_train,
                       hidden = c(10),
                       linear.output = FALSE,
                       threshold = 0.01,
                       stepmax = 1e6,
                       rep = 3,
                       lifesign = "minimal",
                       algorithm = "rprop+")

cat("ANN trained successfully!\n")
summary(ann_model)
plot(ann_model)

# Prediction on validate 
pred_raw <- predict(ann_model, ann_valid)
if (is.list(pred_raw)) {
  predictions_ann <- pred_raw[[which.min(ann_model$result.matrix[1, ])]]
} else {
  predictions_ann <- pred_raw
}

head(pred_raw)


# 1. Hit-rate (Accuracy)
predicted_class_ann <- ifelse(predictions_ann > 0.5, "1", "0")
hit_table_ann <- table(Observed = validation_dataset$Churn,
                       Predicted = predicted_class_ann,
                       dnn = c("Observed", "Predicted"))
hit_rate_ann <- sum(diag(hit_table_ann)) / sum(hit_table_ann)
cat("ANN Hit-rate:", round(hit_rate_ann, 4), "\n")

# 2. Top-decile lift
decile_ann <- ntile(predictions_ann, 10)
top10_churn_rate_ann <- mean(validation_dataset$Churn[decile_ann == 10] == "1")
overall_churn_rate   <- mean(validation_dataset$Churn == "1")
top_decile_lift_ann  <- top10_churn_rate_ann / overall_churn_rate
cat("ANN Top-decile lift:", round(top_decile_lift_ann, 3), "\n")

# 3. Gini coefficient
pred_obj_ann <- prediction(predictions_ann, validation_dataset$Churn)
auc_ann <- performance(pred_obj_ann, "auc")@y.values[[1]]
gini_ann <- 2 * auc_ann - 1
cat("ANN Gini:", round(gini_ann, 4), "\n\n")


# =============================================================================
# 7. FINAL PERFORMANCE COMPARISON TABLE
# =============================================================================
final_comparison <- data.frame(
  Model = c("Baseline Logistic", "Step-wise Logistic", "CART", "Bagging",
            "Gradient Boosting", "Random Forest", "SVM (Radial)", "Neural Network"),
  Gini = round(c(gini1, gini_step, gini_cart, gini_bag, gini_gbm, gini_rf, gini_svm, gini_ann), 4),
  Top_Decile_Lift = round(c(top_decile_lift1, top_decile_lift_step, top_decile_lift,
                            top_decile_lift_bag, top_decile_lift_gbm, top_decile_lift_rf,
                            top_decile_lift_svm, top_decile_lift_ann), 3),
  Hit_Rate = round(c(hit_rate1, hit_rate_step, hit_rate_cart, hit_rate_bag,
                     hit_rate_gbm, hit_rate_rf, hit_rate_svm, hit_rate_ann), 4)
)

# Ranking
final_comparison <- final_comparison[order(-final_comparison$Gini), ]
final_comparison$Rank <- 1:nrow(final_comparison)
final_comparison <- final_comparison[, c("Rank", "Model", "Gini", "Top_Decile_Lift", "Hit_Rate")]

cat("\n=== FINAL MODEL PERFORMANCE COMPARISON ===\n")
print(final_comparison, row.names = FALSE)


# =============================================================================
# 8.ROC CURVES + LIFT CHARTS 
# =============================================================================

library(gridExtra)

pred_list <- list(
  "Baseline Logistic" = predictions_model1,
  "Step-wise Logistic" = predictions_step,
  "CART"               = predictions_cart,
  "Random Forest"      = predictions_rf,
  "SVM (Radial)"       = predictions_svm,
  "Neural Network"     = predictions_ann
)

# --- 2. Create data for ROC Curve ---
roc_data <- data.frame()
for(model_name in names(pred_list)) {
  pred <- prediction(pred_list[[model_name]], validation_dataset$Churn)
  perf <- performance(pred, "tpr", "fpr")
  roc_data <- rbind(roc_data, data.frame(
    FPR = unlist(perf@x.values),
    TPR = unlist(perf@y.values),
    Model = model_name
  ))
}

# Plot ROC Curve
p_roc <- ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  labs(title = "Figure 8.1: ROC Curves – Model Comparison",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold")) +
  guides(color = guide_legend(nrow = 2))

# Create data for Lift Chart ---
lift_data <- data.frame()
for(model_name in names(pred_list)) {
  scores <- pred_list[[model_name]]
  decile <- ntile(scores, 10)
  lift_table <- tapply(validation_dataset$Churn == "1", decile, mean)
  cum_lift <- cumsum(lift_table * table(decile)) / cumsum(table(decile))
  random_lift <- seq(0.1, 1, 0.1)
  lift_data <- rbind(lift_data, data.frame(
    Decile = 1:10,
    Cumulative_Gain = cum_lift * 100,
    Random = random_lift * 100,
    Model = model_name
  ))
}

# Lift Chart
p_lift <- ggplot(lift_data, aes(x = Decile)) +
  geom_line(aes(y = Cumulative_Gain, color = Model), size = 1.3) +
  geom_line(aes(y = Random), color = "gray60", linetype = "dashed", size = 1) +
  scale_x_reverse(breaks = 1:10) +
  labs(title = "Figure 8.2: Cumulative Lift Charts – Top-decile Analysis",
       subtitle = "Higher curve = better concentration of churners in top deciles",
       x = "Decile (10 = highest predicted risk)", 
       y = "% of Total Churners Captured") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom", plot.title = element_text(face = "bold")) +
  guides(color = guide_legend(nrow = 2))

grid.arrange(p_roc, p_lift, ncol = 2)