library(dplyr)
library(xgboost)
library(caret)

# Load data (assumes the CSV files are in your working directory)
DataDictionary <- read.csv("DataDictionary.csv")
test <- read.csv("East Regional Games to predict.csv")
train <- read.csv("games_2022.csv")
region_groups <- read.csv("Team Region Groups.csv")

# Data Wrangling
stats <- train |> 
  group_by(team) |> 
  summarize(
    FG2_percentage = mean(FGM_2 / FGA_2, na.rm = TRUE),
    FG3_percentage = mean(FGM_3 / FGA_3, na.rm = TRUE),
    FT_percentage = mean(FTM / FTA, na.rm = TRUE), 
    AST = mean(AST, na.rm = TRUE), 
    BLK = mean(BLK, na.rm = TRUE), 
    STL = mean(STL, na.rm = TRUE), 
    TOV = mean(TOV, na.rm = TRUE), 
    TOV_team = mean(TOV_team, na.rm = TRUE), 
    DREB = mean(DREB, na.rm = TRUE), 
    OREB = mean(OREB, na.rm = TRUE), 
    F_tech = mean(F_tech, na.rm = TRUE),
    F_personal = mean(F_personal, na.rm = TRUE), 
    average_score = mean(team_score[is.na(OT_length_min_tot)], na.rm = TRUE),
    largest_lead = mean(largest_lead), 
    notD1_incomplete = any(notD1_incomplete), 
    attendance = mean(attendance, na.rm = TRUE)
  )

test <- test |> 
  left_join(
    stats |> rename_with(~ paste0(.x, "_home"), -team), 
    by = join_by(team_home == team)
  ) |> 
  left_join(
    stats |> rename_with(~ paste0(.x, "_away"), -team), 
    by = join_by(team_away == team)
  ) |> 
  dplyr::select(
    FG2_percentage_home, 
    FG2_percentage_away, 
    FG3_percentage_home,
    FG3_percentage_away,
    FT_percentage_home,
    FT_percentage_away,
    AST_home,
    AST_away,
    BLK_home,
    BLK_away,
    STL_home,
    STL_away,
    TOV_home,
    TOV_away,
    TOV_team_home,
    TOV_team_away,
    DREB_home,
    DREB_away,
    OREB_home,
    OREB_away,
    F_tech_home,
    F_tech_away,
    F_personal_home,
    F_personal_away,
    average_score_home,
    average_score_away,
    largest_lead_home,
    largest_lead_away,
    notD1_incomplete_home,
    notD1_incomplete_away,
    attendance_home,
    attendance_away,
    home_away_NS,
    rest_days_Home,
    rest_days_Away,
    travel_dist_Home,
    travel_dist_Away
  ) |> 
  rename(
    rest_days_home = rest_days_Home,
    rest_days_away = rest_days_Away,
    travel_dist_home = travel_dist_Home,
    travel_dist_away = travel_dist_Away
  )

cummean_if <- function(x, condition) {
  n <- length(x)
  out <- numeric(n)
  for (i in seq_len(n)) {
    if (i == 1) {
      out[i] <- NA
    } else {
      valid <- condition[1:(i - 1)]
      if (sum(valid) == 0) {
        out[i] <- NA
      } else {
        out[i] <- mean(x[1:(i - 1)][valid], na.rm = TRUE)
      }
    }
  }
  out
}

cumany <- function(x) {
  out <- logical(length(x))
  acc <- FALSE
  for (i in seq_along(x)) {
    acc <- acc | x[i]
    out[i] <- acc
  }
  out
}

train_pregame <- train %>%
  arrange(team, game_date) %>%   
  group_by(team) %>%
  mutate(
    FG2_percentage = lag(cummean(FGM_2 / FGA_2)),
    FG3_percentage = lag(cummean(FGM_3 / FGA_3)),
    FT_percentage  = lag(cummean(FTM / FTA)),
    AST         = lag(cummean(AST)),
    BLK         = lag(cummean(BLK)),
    STL         = lag(cummean(STL)),
    TOV         = lag(cummean(TOV)),
    TOV_team    = lag(cummean(TOV_team)),
    DREB        = lag(cummean(DREB)),
    OREB        = lag(cummean(OREB)),
    F_tech      = lag(cummean(F_tech)),
    F_personal  = lag(cummean(F_personal)),
    largest_lead = lag(cummean(largest_lead)),
    attendance  = lag(cummean(attendance)),
    average_score = cummean_if(team_score, is.na(OT_length_min_tot)),
    notD1_incomplete = lag(cumany(notD1_incomplete))
  ) %>%
  dplyr::select(
    game_id,
    game_date,
    team,
    FG2_percentage, 
    FG3_percentage,
    FT_percentage,
    AST,
    BLK,
    STL,
    TOV,
    TOV_team,
    DREB,
    OREB,
    F_tech,
    F_personal,
    average_score,
    largest_lead,
    notD1_incomplete,
    attendance,
    home_away_NS,
    rest_days,
    travel_dist,
    home_away, 
    team_score,
    opponent_team_score
  ) |> 
  ungroup()

train_home <- train_pregame %>%
  filter(home_away == "home") %>%
  dplyr::select(-home_away) 

train_away <- train_pregame %>%
  filter(home_away == "away") %>%
  dplyr::select(-home_away)

merged_games <- inner_join(
  train_home,
  train_away,
  by = c("game_id", "game_date"),
  suffix = c("_home", "_away")
) |>
  mutate(
    home_win = if_else(team_score_home > opponent_team_score_home, 1, 0)
  ) |> 
  dplyr::select(
    FG2_percentage_home, 
    FG2_percentage_away, 
    FG3_percentage_home,
    FG3_percentage_away,
    FT_percentage_home,
    FT_percentage_away,
    AST_home,
    AST_away,
    BLK_home,
    BLK_away,
    STL_home,
    STL_away,
    TOV_home,
    TOV_away,
    TOV_team_home,
    TOV_team_away,
    DREB_home,
    DREB_away,
    OREB_home,
    OREB_away,
    F_tech_home,
    F_tech_away,
    F_personal_home,
    F_personal_away,
    average_score_home,
    average_score_away,
    largest_lead_home,
    largest_lead_away,
    notD1_incomplete_home,
    notD1_incomplete_away,
    attendance_home,
    attendance_away,
    home_away_NS_home,
    rest_days_home,
    rest_days_away,
    travel_dist_home,
    travel_dist_away,
    home_win
  ) |> 
  rename(
    home_away_NS = home_away_NS_home
  )

# Machine Learning

# For training data:
train_features <- merged_games %>%
  # Convert logicals (and any factors) to numeric if necessary
  mutate(across(c(notD1_incomplete_home, notD1_incomplete_away), ~ as.numeric(.)),
         home_away_NS = as.numeric(home_away_NS)) %>%
  # Remove the target variable from features
  dplyr::select(-home_win)

# Convert to matrix (xgboost expects a numeric matrix)
train_matrix <- as.matrix(train_features)

# Extract the target variable
train_label <- merged_games$home_win

# Create the xgboost DMatrix for training
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)

# For test data:
test_features <- test %>%
  mutate(across(c(notD1_incomplete_home, notD1_incomplete_away), ~ as.numeric(.)),
         home_away_NS = as.numeric(home_away_NS))

test_matrix <- as.matrix(test_features)
dtest <- xgb.DMatrix(data = test_matrix)

# Define the parameters for XGBoost
params <- list(
  objective = "binary:logistic",  # binary classification
  eval_metric = "auc",            # evaluation metric: Area Under the Curve
  max_depth = 6,                  # maximum depth of trees
  eta = 0.1,                      # learning rate
  subsample = 0.8,                # subsample ratio of the training instance
  colsample_bytree = 0.8          # subsample ratio of columns when constructing each tree
)

# ------------------------------------------------------------------
# First: Use xgb.cv to determine the best number of boosting rounds
cv_results <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 100,              # maximum number of boosting rounds
  nfold = 5,                  # 5-fold cross-validation
  early_stopping_rounds = 10, # stop if no improvement for 10 rounds
  verbose = 1
)

# Retrieve the best number of rounds from cross-validation
best_nrounds <- cv_results$best_iteration
cat("Best number of rounds from xgb.cv:", best_nrounds, "\n")

# ------------------------------------------------------------------
# Next: Manual 5-Fold Cross-Validation to report confusion matrices
set.seed(123)
folds <- createFolds(train_label, k = 5, list = TRUE, returnTrain = FALSE)
# This vector will hold the out-of-fold predictions for an overall validation performance
val_preds_all <- rep(NA, length(train_label))

cat("\nManual Cross-Validation Results:\n")
for (fold in seq_along(folds)) {
  cat(sprintf("\n--- Fold %d ---\n", fold))
  val_idx <- folds[[fold]]
  train_idx <- setdiff(seq_along(train_label), val_idx)
  
  # Create fold-specific DMatrix objects
  dtrain_fold <- xgb.DMatrix(data = train_matrix[train_idx, ], label = train_label[train_idx])
  dval_fold   <- xgb.DMatrix(data = train_matrix[val_idx, ], label = train_label[val_idx])
  
  watchlist <- list(train = dtrain_fold, eval = dval_fold)
  
  # Train on the current fold using best_nrounds from xgb.cv
  model_fold <- xgb.train(
    params = params,
    data = dtrain_fold,
    nrounds = best_nrounds,
    watchlist = watchlist,
    verbose = 0
  )
  
  # Predict on training fold and validation fold
  train_pred_prob <- predict(model_fold, dtrain_fold)
  val_pred_prob   <- predict(model_fold, dval_fold)
  
  # Convert probabilities to binary classes (threshold = 0.5)
  train_pred_class <- ifelse(train_pred_prob > 0.5, 1, 0)
  val_pred_class   <- ifelse(val_pred_prob > 0.5, 1, 0)
  
  # Compute and print confusion matrices
  cm_train <- confusionMatrix(as.factor(train_pred_class), as.factor(train_label[train_idx]))
  cm_val   <- confusionMatrix(as.factor(val_pred_class), as.factor(train_label[val_idx]))
  
  cat("Training Confusion Matrix:\n")
  print(cm_train)
  
  cat("Validation Confusion Matrix:\n")
  print(cm_val)
  
  # Store out-of-fold validation predictions
  val_preds_all[val_idx] <- val_pred_prob
}

# Overall Validation Performance (aggregated across folds)
overall_val_class <- ifelse(val_preds_all > 0.5, 1, 0)
overall_cm <- confusionMatrix(as.factor(overall_val_class), as.factor(train_label))
cat("\nOverall Validation Confusion Matrix (aggregated across folds):\n")
print(overall_cm)

# ------------------------------------------------------------------
# Now, train the final model on the full training data using best_nrounds
final_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = best_nrounds,
  verbose = 1
)

# Predict on the full training set (this is in-sample; it may be overoptimistic)
train_predictions <- predict(final_model, dtrain)
train_predicted_classes <- ifelse(train_predictions > 0.5, 1, 0)
conf_matrix <- confusionMatrix(as.factor(train_predicted_classes), as.factor(train_label))
cat("\nConfusion Matrix on Full Training Data (Final Model):\n")
print(conf_matrix)

# ------------------------------------------------------------------
# Predict on the test set using the full model
predictions <- predict(final_model, dtest)
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# View the first few predictions
cat("\nTest Set Predicted Probabilities:\n")
print((predictions))
cat("\nTest Set Predicted Classes:\n")
print((predicted_classes))

# ------------------------------------------------------------------
# Feature Importance
feature_names <- colnames(train_matrix)
importance_matrix <- xgb.importance(feature_names, model = final_model)
print(importance_matrix)
xgb.plot.importance(importance_matrix)
