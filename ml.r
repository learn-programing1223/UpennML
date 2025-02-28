###############################################################################
# Integrated Model Training & Overfitting/Calibration Validation Script
###############################################################################

# 0) Libraries, Seed, and Data Reading
library(tidyverse)
library(caret)
library(xgboost)
library(rBayesianOptimization)  # for Bayesian hyperparameter tuning
library(ggplot2)
library(gridExtra)

set.seed(123)

# Define file paths (adjust these paths as needed)
train_raw_path <- "games_2022.csv"
test_raw_path  <- "East Regional Games to predict.csv"

# Read training and test files (using check.names=TRUE for valid column names)
train_raw <- read.csv(train_raw_path, stringsAsFactors = FALSE, check.names = TRUE)
test_raw  <- read.csv(test_raw_path, stringsAsFactors = FALSE, check.names = TRUE)

###############################################################################
# 1) Define the Whole-History Rating (WHR) Functions
###############################################################################
expected_home_win_prob <- function(rh, ra, home_field_advantage = 32, c = 0.01) {
  1 / (1 + exp(-((rh + home_field_advantage) - ra) * c))
}

update_ratings_for_game <- function(rh, ra, outcome_home,
                                    k = 20,
                                    home_field_advantage = 32,
                                    c = 0.01,
                                    margin_of_victory = 1) {
  p_home <- expected_home_win_prob(rh, ra, home_field_advantage, c)
  error_home <- outcome_home - p_home
  error_away <- (1 - outcome_home) - (1 - p_home)
  list(rh = rh + k * error_home * margin_of_victory,
       ra = ra + k * error_away * margin_of_victory)
}

whr_pass <- function(ratings, df_games, k, home_field_advantage, c, margin_func = NULL) {
  for (i in seq_len(nrow(df_games))) {
    rowi <- df_games[i, ]
    home_team <- rowi[["team_home"]]
    away_team <- rowi[["team_away"]]
    outcome_home <- rowi[["home_win"]]
    margin_val <- if (!is.null(margin_func)) margin_func(rowi) else 1
    old_rh <- ratings[home_team]
    old_ra <- ratings[away_team]
    updated <- update_ratings_for_game(
      rh = old_rh,
      ra = old_ra,
      outcome_home = outcome_home,
      k = k,
      home_field_advantage = home_field_advantage,
      c = c,
      margin_of_victory = margin_val
    )
    ratings[home_team] <- updated$rh
    ratings[away_team] <- updated$ra
  }
  ratings
}

run_whole_history_rating <- function(df_games,
                                     init_rating = 1500,
                                     k = 20,
                                     home_field_advantage = 32,
                                     c = 0.01,
                                     margin_func = NULL,
                                     max_iter = 50,
                                     tol = 0.0005) {
  all_teams <- unique(c(df_games$team_home, df_games$team_away))
  ratings <- rep(init_rating, length(all_teams))
  names(ratings) <- all_teams
  
  for (iter in seq_len(max_iter)) {
    old_ratings <- ratings
    ratings <- whr_pass(ratings, df_games, k, home_field_advantage, c, margin_func)
    avg_change <- mean(abs(ratings - old_ratings), na.rm = TRUE)
    cat(sprintf("Iteration %d: avg rating change = %.4f\n", iter, avg_change))
    if (avg_change < tol) {
      cat(sprintf("Converged at iteration %d\n", iter))
      break
    }
  }
  ratings
}

###############################################################################
# 2) Process and Aggregate the Training Data
###############################################################################
# Remove rows missing team identifier
train_raw <- train_raw %>% filter(!is.na(team))

# Aggregate training data by game_id (one row per game)
train <- train_raw %>% 
  group_by(game_id) %>% 
  summarize(
    team_home  = first(team[home_away == "home"]),
    team_away  = first(team[home_away == "away"]),
    score_home = first(team_score[home_away == "home"]),
    score_away = first(opponent_team_score[home_away == "home"]),
    home_win   = if_else(first(team_score[home_away == "home"]) > first(opponent_team_score[home_away == "home"]), 1, 0),
    
    FG2_percentage_home = mean(FGM_2[home_away == "home"] / FGA_2[home_away == "home"], na.rm = TRUE),
    FG2_percentage_away = mean(FGM_2[home_away == "away"] / FGA_2[home_away == "away"], na.rm = TRUE),
    FG3_percentage_home = mean(FGM_3[home_away == "home"] / FGA_3[home_away == "home"], na.rm = TRUE),
    FG3_percentage_away = mean(FGM_3[home_away == "away"] / FGA_3[home_away == "away"], na.rm = TRUE),
    FT_percentage_home  = mean(FTM[home_away == "home"] / FTA[home_away == "home"], na.rm = TRUE),
    FT_percentage_away  = mean(FTM[home_away == "away"] / FTA[home_away == "away"], na.rm = TRUE),
    
    AST_home = mean(AST[home_away == "home"], na.rm = TRUE),
    AST_away = mean(AST[home_away == "away"], na.rm = TRUE),
    BLK_home = mean(BLK[home_away == "home"], na.rm = TRUE),
    BLK_away = mean(BLK[home_away == "away"], na.rm = TRUE),
    STL_home = mean(STL[home_away == "home"], na.rm = TRUE),
    STL_away = mean(STL[home_away == "away"], na.rm = TRUE),
    TOV_home = mean(TOV[home_away == "home"], na.rm = TRUE),
    TOV_away = mean(TOV[home_away == "away"], na.rm = TRUE),
    DREB_home = mean(DREB[home_away == "home"], na.rm = TRUE),
    DREB_away = mean(DREB[home_away == "away"], na.rm = TRUE),
    OREB_home = mean(OREB[home_away == "home"], na.rm = TRUE),
    OREB_away = mean(OREB[home_away == "away"], na.rm = TRUE),
    
    home_away_NS = first(home_away_NS[home_away == "home"]),
    rest_days_home = first(rest_days[home_away == "home"]),
    rest_days_away = first(rest_days[home_away == "away"]),
    travel_dist_home = first(travel_dist[home_away == "home"]),
    travel_dist_away = first(travel_dist[home_away == "away"]),
    
    notD1_incomplete_home = as.numeric(any(notD1_incomplete[home_away == "home"])),
    notD1_incomplete_away = as.numeric(any(notD1_incomplete[home_away == "away"]))
  ) %>% ungroup()

cat("\nAggregated train data sample:\n")
print(head(train))

###############################################################################
# 2a) Build Team-Level Averages for Imputation
###############################################################################
team_stats <- train_raw %>% 
  group_by(team) %>% 
  summarize(
    FG2_pct = mean(FGM_2 / FGA_2, na.rm = TRUE),
    FG3_pct = mean(FGM_3 / FGA_3, na.rm = TRUE),
    FT_pct  = mean(FTM / FTA, na.rm = TRUE),
    AST_avg = mean(AST, na.rm = TRUE),
    BLK_avg = mean(BLK, na.rm = TRUE),
    STL_avg = mean(STL, na.rm = TRUE),
    TOV_avg = mean(TOV, na.rm = TRUE),
    DREB_avg = mean(DREB, na.rm = TRUE),
    OREB_avg = mean(OREB, na.rm = TRUE),
    notD1 = as.numeric(any(notD1_incomplete)),
    off_eff_avg = mean(team_score / (TOV + 1) * (1 + AST / 100), na.rm = TRUE),
    def_eff_avg = mean((BLK + STL) / pmax(TOV, 1), na.rm = TRUE)
  ) %>% ungroup()

###############################################################################
# 3) Compute WHR (Team Ratings)
###############################################################################
my_margin_func <- function(row) {
  diff_abs <- abs(row$score_home - row$score_away)
  1 + log(1 + diff_abs)
}

set.seed(123)
whr_ratings <- run_whole_history_rating(
  df_games = train,
  init_rating = 1500,
  k = 20,
  home_field_advantage = 32,
  c = 0.01,
  margin_func = my_margin_func,
  max_iter = 50,
  tol = 0.0005
)

cat("\nSample of final WHR ratings:\n")
print(head(whr_ratings))

###############################################################################
# 4) Feature Engineering for Training Data
###############################################################################
# Join team_stats (for efficiency metrics) for both home and away teams
train_fe <- train %>% 
  left_join(team_stats %>% select(team, off_eff_avg, def_eff_avg), by = c("team_home" = "team")) %>%
  rename(off_eff_home = off_eff_avg, def_eff_home = def_eff_avg) %>%
  left_join(team_stats %>% select(team, off_eff_avg, def_eff_avg), by = c("team_away" = "team")) %>%
  rename(off_eff_away = off_eff_avg, def_eff_away = def_eff_avg) %>%
  mutate(
    rating_home = whr_ratings[team_home],
    rating_away = whr_ratings[team_away],
    diff_rating = rating_home - rating_away,
    
    diff_FG2 = FG2_percentage_home - FG2_percentage_away,
    diff_FG3 = FG3_percentage_home - FG3_percentage_away,
    diff_FT  = FT_percentage_home - FT_percentage_away,
    
    ratio_AST = AST_home / pmax(AST_away, 1),
    ratio_BLK = BLK_home / pmax(BLK_away, 1),
    ratio_STL = STL_home / pmax(STL_away, 1),
    ratio_REB = (DREB_home + OREB_home) / pmax(DREB_away + OREB_away, 1),
    
    home_away_NS = as.numeric(home_away_NS)
  )

# Select 34 features for modeling
train_features <- train_fe %>% 
  select(
    rating_home, rating_away, diff_rating,
    FG2_percentage_home, FG2_percentage_away, diff_FG2,
    FG3_percentage_home, FG3_percentage_away, diff_FG3,
    FT_percentage_home, FT_percentage_away, diff_FT,
    ratio_AST, ratio_BLK, ratio_STL, ratio_REB,
    off_eff_home, off_eff_away, def_eff_home, def_eff_away,
    home_away_NS, rest_days_home, rest_days_away,
    travel_dist_home, travel_dist_away,
    notD1_incomplete_home, notD1_incomplete_away
  )

train_label <- factor(train_fe$home_win, levels = c(0, 1), labels = c("Loss", "Win"))

train_model_df <- na.omit(data.frame(train_features, home_win = train_label))
cat("\nTraining data dimensions after omitting NAs:\n")
print(dim(train_model_df))

###############################################################################
# 5) Split Data and Tune XGBoost via Bayesian Optimization
###############################################################################
# Create design matrix and DMatrix
train_matrix <- model.matrix(home_win ~ . - 1, data = train_model_df)
train_label_num <- ifelse(train_model_df$home_win == "Win", 1, 0)
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label_num)

cv_folds <- 5

opt_func <- function(max_depth, min_child_weight, subsample, colsample_bytree, eta, gamma) {
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "error",
    max_depth = as.integer(max_depth),
    min_child_weight = min_child_weight,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    eta = eta,
    gamma = gamma
  )
  cv <- xgb.cv(params = params,
               data = dtrain,
               nrounds = 200,
               nfold = cv_folds,
               stratified = TRUE,
               verbose = 0,
               early_stopping_rounds = 10)
  best_error <- min(cv$evaluation_log$test_error_mean)
  # We maximize negative error (accuracy = 1 - error)
  list(Score = -best_error, Pred = cv$evaluation_log$test_error_mean)
}

bounds <- list(
  max_depth = c(3L, 9L),
  min_child_weight = c(1, 10),
  subsample = c(0.7, 1),
  colsample_bytree = c(0.7, 1),
  eta = c(0.01, 0.1),
  gamma = c(0, 2)
)

set.seed(123)
opt_results <- BayesianOptimization(opt_func,
                                    bounds = bounds,
                                    init_points = 5,
                                    n_iter = 10,
                                    acq = "ucb",
                                    kappa = 2.576,
                                    verbose = TRUE)

best_params <- opt_results$Best_Par
cat("\nBest parameters from Bayesian Optimization:\n")
print(best_params)

final_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "error",
  max_depth = as.integer(best_params["max_depth"]),
  min_child_weight = best_params["min_child_weight"],
  subsample = best_params["subsample"],
  colsample_bytree = best_params["colsample_bytree"],
  eta = best_params["eta"],
  gamma = best_params["gamma"]
)

set.seed(123)
xgb_final <- xgb.train(
  params = final_params,
  data = dtrain,
  nrounds = 200,
  verbose = 0
)

# Evaluate final model with caret split
set.seed(123)
trainIndex <- createDataPartition(train_model_df$home_win, p = 0.8, list = FALSE)
train_split <- train_model_df[trainIndex, ]
valid_split <- train_model_df[-trainIndex, ]

train_matrix_split <- model.matrix(home_win ~ . - 1, data = train_split)
train_label_split <- ifelse(train_split$home_win == "Win", 1, 0)
dtrain_split <- xgb.DMatrix(data = train_matrix_split, label = train_label_split)

valid_matrix_split <- model.matrix(home_win ~ . - 1, data = valid_split)
valid_label_split <- ifelse(valid_split$home_win == "Win", 1, 0)
dvalid_split <- xgb.DMatrix(data = valid_matrix_split, label = valid_label_split)

train_pred <- predict(xgb_final, dtrain_split)
train_pred_class <- factor(ifelse(train_pred > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))
train_cm <- confusionMatrix(train_pred_class, train_split$home_win)
cat("\nFinal Training Accuracy:", round(train_cm$overall["Accuracy"] * 100, 2), "%\n")

valid_pred <- predict(xgb_final, dvalid_split)
valid_pred_class <- factor(ifelse(valid_pred > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))
valid_cm <- confusionMatrix(valid_pred_class, valid_split$home_win)
cat("\nFinal Validation Accuracy:", round(valid_cm$overall["Accuracy"] * 100, 2), "%\n")

###############################################################################
# 6) Prepare Test Data – Feature Engineering and Imputation
###############################################################################
test <- test_raw %>%
  rename(
    rest_days_home = rest_days_Home,
    rest_days_away = rest_days_Away,
    travel_dist_home = travel_dist_Home,
    travel_dist_away = travel_dist_Away
  )

test_fe <- test %>%
  left_join(team_stats %>% select(team, FG2_pct, FG3_pct, FT_pct, AST_avg, BLK_avg, STL_avg, TOV_avg, DREB_avg, OREB_avg, notD1, off_eff_avg, def_eff_avg),
            by = c("team_home" = "team")) %>%
  rename(
    FG2_percentage_home = FG2_pct,
    FG3_percentage_home = FG3_pct,
    FT_percentage_home = FT_pct,
    AST_home = AST_avg,
    BLK_home = BLK_avg,
    STL_home = STL_avg,
    TOV_home = TOV_avg,
    DREB_home = DREB_avg,
    OREB_home = OREB_avg,
    notD1_incomplete_home = notD1,
    off_eff_home = off_eff_avg,
    def_eff_home = def_eff_avg
  ) %>%
  left_join(team_stats %>% select(team, FG2_pct, FG3_pct, FT_pct, AST_avg, BLK_avg, STL_avg, TOV_avg, DREB_avg, OREB_avg, notD1, off_eff_avg, def_eff_avg),
            by = c("team_away" = "team")) %>%
  rename(
    FG2_percentage_away = FG2_pct,
    FG3_percentage_away = FG3_pct,
    FT_percentage_away = FT_pct,
    AST_away = AST_avg,
    BLK_away = BLK_avg,
    STL_away = STL_avg,
    TOV_away = TOV_avg,
    DREB_away = DREB_avg,
    OREB_away = OREB_avg,
    notD1_incomplete_away = notD1,
    off_eff_away = off_eff_avg,
    def_eff_away = def_eff_avg
  ) %>%
  mutate(
    diff_FG2 = FG2_percentage_home - FG2_percentage_away,
    diff_FG3 = FG3_percentage_home - FG3_percentage_away,
    diff_FT  = FT_percentage_home - FT_percentage_away,
    ratio_AST = AST_home / pmax(AST_away, 1),
    ratio_BLK = BLK_home / pmax(BLK_away, 1),
    ratio_STL = STL_home / pmax(STL_away, 1),
    ratio_REB = (DREB_home + OREB_home) / pmax(DREB_away + OREB_away, 1),
    rating_home = whr_ratings[team_home],
    rating_away = whr_ratings[team_away],
    diff_rating = rating_home - rating_away,
    home_away_NS = as.numeric(home_away_NS)
  )

test_features <- test_fe %>% 
  select(
    rating_home, rating_away, diff_rating,
    FG2_percentage_home, FG2_percentage_away, diff_FG2,
    FG3_percentage_home, FG3_percentage_away, diff_FG3,
    FT_percentage_home, FT_percentage_away, diff_FT,
    ratio_AST, ratio_BLK, ratio_STL, ratio_REB,
    off_eff_home, off_eff_away, def_eff_home, def_eff_away,
    home_away_NS, rest_days_home, rest_days_away,
    travel_dist_home, travel_dist_away,
    notD1_incomplete_home, notD1_incomplete_away
  )

cat("\nTest data after feature engineering:\n")
print(head(test_features))

###############################################################################
# 7) Predict on Test Data
###############################################################################
test_pred_prob <- predict(xgb_final, newdata = xgb.DMatrix(data = model.matrix(~ . - 1, data = test_features)))
test_pred_class <- factor(ifelse(test_pred_prob > 0.5, "Win", "Loss"), levels = c("Loss", "Win"))

test_results <- test_fe %>%
  mutate(
    predicted_class_home_win = test_pred_class,
    predicted_prob_home_win = test_pred_prob
  )

cat("\nFinal Test Predictions:\n")
print(test_results %>% 
        select(team_home, team_away, rest_days_home, rest_days_away,
               travel_dist_home, travel_dist_away,
               predicted_class_home_win, predicted_prob_home_win))
cat("\nSCRIPT COMPLETE.\n")

###############################################################################
# 8) Model Validation: Overfitting, Learning Curves, and Calibration
###############################################################################

## 8a) Cross-Validation Analysis Function
run_cv_analysis <- function(train_model_df, k = 5, seed = 123) {
  set.seed(seed)
  folds <- createFolds(train_model_df$home_win, k = k, list = TRUE, returnTrain = FALSE)
  
  train_acc <- numeric(k)
  valid_acc <- numeric(k)
  train_brier <- numeric(k)
  valid_brier <- numeric(k)
  all_preds <- data.frame()
  
  for (i in 1:k) {
    cat(sprintf("Processing fold %d of %d...\n", i, k))
    fold_test <- train_model_df[folds[[i]], ]
    fold_train <- train_model_df[-folds[[i]], ]
    
    train_matrix <- model.matrix(home_win ~ . - 1, data = fold_train)
    train_label_num <- ifelse(fold_train$home_win == "Win", 1, 0)
    dtrain <- xgb.DMatrix(data = train_matrix, label = train_label_num)
    
    valid_matrix <- model.matrix(home_win ~ . - 1, data = fold_test)
    valid_label_num <- ifelse(fold_test$home_win == "Win", 1, 0)
    dvalid <- xgb.DMatrix(data = valid_matrix, label = valid_label_num)
    
    params <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "error",
      max_depth = final_params$max_depth,
      min_child_weight = final_params$min_child_weight,
      subsample = final_params$subsample,
      colsample_bytree = final_params$colsample_bytree,
      eta = final_params$eta,
      gamma = final_params$gamma
    )
    
    watchlist <- list(train = dtrain, valid = dvalid)
    fold_model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = 200,
      watchlist = watchlist,
      verbose = 0,
      early_stopping_rounds = 10
    )
    
    train_pred <- predict(fold_model, dtrain)
    train_pred_class <- ifelse(train_pred > 0.5, "Win", "Loss")
    train_pred_class <- factor(train_pred_class, levels = c("Loss", "Win"))
    
    valid_pred <- predict(fold_model, dvalid)
    valid_pred_class <- ifelse(valid_pred > 0.5, "Win", "Loss")
    valid_pred_class <- factor(valid_pred_class, levels = c("Loss", "Win"))
    
    train_acc[i] <- confusionMatrix(train_pred_class, fold_train$home_win)$overall["Accuracy"]
    valid_acc[i] <- confusionMatrix(valid_pred_class, fold_test$home_win)$overall["Accuracy"]
    
    train_brier[i] <- mean((train_pred - train_label_num)^2)
    valid_brier[i] <- mean((valid_pred - valid_label_num)^2)
    
    fold_preds <- data.frame(
      fold = i,
      set = "validation",
      pred_prob = valid_pred,
      actual = valid_label_num
    )
    all_preds <- rbind(all_preds, fold_preds)
  }
  
  cv_results <- data.frame(
    fold = 1:k,
    train_accuracy = train_acc,
    valid_accuracy = valid_acc,
    train_brier = train_brier,
    valid_brier = valid_brier
  )
  
  cv_summary <- data.frame(
    metric = c("Training Accuracy", "Validation Accuracy", "Training Brier", "Validation Brier"),
    mean = c(mean(train_acc), mean(valid_acc), mean(train_brier), mean(valid_brier)),
    sd = c(sd(train_acc), sd(valid_acc), sd(train_brier), sd(valid_brier)),
    min = c(min(train_acc), min(valid_acc), min(train_brier), min(valid_brier)),
    max = c(max(train_acc), max(valid_acc), max(train_brier), max(valid_brier))
  )
  
  overfitting_gap <- mean(train_acc) - mean(valid_acc)
  
  list(
    results = cv_results,
    summary = cv_summary,
    overfitting_gap = overfitting_gap,
    all_predictions = all_preds
  )
}

## 8b) Learning Curves Function
generate_learning_curves <- function(train_model_df, train_sizes = seq(0.1, 0.9, by = 0.1), seed = 123) {
  set.seed(seed)
  trainIndex <- createDataPartition(train_model_df$home_win, p = 0.8, list = FALSE)
  train_data <- train_model_df[trainIndex, ]
  valid_data <- train_model_df[-trainIndex, ]
  
  valid_matrix <- model.matrix(home_win ~ . - 1, data = valid_data)
  valid_label_num <- ifelse(valid_data$home_win == "Win", 1, 0)
  dvalid <- xgb.DMatrix(data = valid_matrix, label = valid_label_num)
  
  sizes <- floor(train_sizes * nrow(train_data))
  train_scores <- numeric(length(sizes))
  valid_scores <- numeric(length(sizes))
  
  for (i in seq_along(sizes)) {
    cat(sprintf("Training with %d examples (%.1f%%)...\n", sizes[i], train_sizes[i] * 100))
    train_indices <- sample(nrow(train_data), sizes[i])
    train_subset <- train_data[train_indices, ]
    
    train_matrix <- model.matrix(home_win ~ . - 1, data = train_subset)
    train_label_num <- ifelse(train_subset$home_win == "Win", 1, 0)
    dtrain <- xgb.DMatrix(data = train_matrix, label = train_label_num)
    
    params <- list(
      booster = "gbtree",
      objective = "binary:logistic",
      eval_metric = "error",
      max_depth = final_params$max_depth,
      min_child_weight = final_params$min_child_weight,
      subsample = final_params$subsample,
      colsample_bytree = final_params$colsample_bytree,
      eta = final_params$eta,
      gamma = final_params$gamma
    )
    
    model <- xgb.train(
      params = params,
      data = dtrain,
      nrounds = 100,
      verbose = 0
    )
    
    train_pred <- predict(model, dtrain)
    train_pred_class <- ifelse(train_pred > 0.5, "Win", "Loss")
    train_pred_class <- factor(train_pred_class, levels = c("Loss", "Win"))
    
    valid_pred <- predict(model, dvalid)
    valid_pred_class <- ifelse(valid_pred > 0.5, "Win", "Loss")
    valid_pred_class <- factor(valid_pred_class, levels = c("Loss", "Win"))
    
    train_scores[i] <- confusionMatrix(train_pred_class, train_subset$home_win)$overall["Accuracy"]
    valid_scores[i] <- confusionMatrix(valid_pred_class, valid_data$home_win)$overall["Accuracy"]
  }
  
  learning_curve_data <- data.frame(
    size = sizes,
    percentage = train_sizes * 100,
    train_accuracy = train_scores,
    valid_accuracy = valid_scores
  )
  
  learning_curve_data
}

## 8c) Reliability Diagram and Brier Score
create_reliability_diagram <- function(predictions, actual_outcomes, n_bins = 10) {
  bin_breaks <- seq(0, 1, length.out = n_bins + 1)
  binned_preds <- cut(predictions, breaks = bin_breaks, include.lowest = TRUE)
  
  reliability_data <- data.frame(
    bin_midpoint = numeric(0),
    observed_freq = numeric(0),
    n_samples = numeric(0)
  )
  
  for (bin in levels(binned_preds)) {
    bin_indices <- which(binned_preds == bin)
    if (length(bin_indices) > 0) {
      bin_range <- as.numeric(str_extract_all(bin, "[-+]?[0-9]*\\.?[0-9]+")[[1]])
      bin_midpoint <- mean(bin_range)
      observed_freq <- mean(actual_outcomes[bin_indices])
      n_samples <- length(bin_indices)
      
      reliability_data <- rbind(reliability_data, data.frame(
        bin_midpoint = bin_midpoint,
        observed_freq = observed_freq,
        n_samples = n_samples
      ))
    }
  }
  
  reliability_data
}

calculate_brier_score <- function(predictions, actual_outcomes) {
  mean((predictions - actual_outcomes)^2)
}

## 8d) Validation Function to run all checks
validate_model <- function(train_model_df, final_model, test_data, final_params) {
  cat("Running cross-validation analysis...\n")
  cv_results <- run_cv_analysis(train_model_df, k = 5)
  
  cat("\nOverfitting check:\n")
  cat("Average training accuracy:", round(mean(cv_results$results$train_accuracy) * 100, 2), "%\n")
  cat("Average validation accuracy:", round(mean(cv_results$results$valid_accuracy) * 100, 2), "%\n")
  cat("Gap (training - validation):", round(cv_results$overfitting_gap * 100, 2), "% points\n")
  if (cv_results$overfitting_gap > 0.05) {
    cat("WARNING: Potential overfitting detected. Gap > 5% points.\n")
  } else {
    cat("Good: Minimal overfitting detected.\n")
  }
  
  cat("\nGenerating learning curves...\n")
  learning_curves <- generate_learning_curves(train_model_df)
  final_gap <- abs(tail(learning_curves$train_accuracy, 1) - tail(learning_curves$valid_accuracy, 1))
  cat("Final gap in learning curves:", round(final_gap * 100, 2), "% points\n")
  if (final_gap > 0.05) {
    cat("WARNING: Learning curves have not converged.\n")
  } else {
    cat("Good: Learning curves have converged.\n")
  }
  
  cat("\nAnalyzing probability calibration...\n")
  set.seed(123)
  validation_split <- train_model_df[createDataPartition(train_model_df$home_win, p = 0.2, list = FALSE), ]
  valid_matrix <- model.matrix(home_win ~ . - 1, data = validation_split)
  valid_label_num <- ifelse(validation_split$home_win == "Win", 1, 0)
  dvalid <- xgb.DMatrix(data = valid_matrix, label = valid_label_num)
  
  valid_preds <- predict(final_model, dvalid)
  reliability_data <- create_reliability_diagram(valid_preds, valid_label_num)
  brier_score <- calculate_brier_score(valid_preds, valid_label_num)
  
  cat("Brier score:", round(brier_score, 4), "\n")
  if (brier_score > 0.25) {
    cat("WARNING: Brier score > 0.25 indicates poor calibration.\n")
  } else {
    cat("Good: Brier score indicates decent calibration.\n")
  }
  
  list(
    cv_results = cv_results,
    learning_curves = learning_curves,
    reliability_data = reliability_data,
    brier_score = brier_score
  )
}

## 8e) Plotting Function for Validation Results
plot_validation_results <- function(results) {
  cv_plot <- ggplot(results$cv_results$results, aes(x = fold)) +
    geom_line(aes(y = train_accuracy, color = "Training"), size = 1) +
    geom_line(aes(y = valid_accuracy, color = "Validation"), size = 1) +
    labs(title = "Cross-Validation Results", x = "Fold", y = "Accuracy") +
    scale_color_manual(values = c("Training" = "blue", "Validation" = "red")) +
    theme_minimal() +
    theme(legend.title = element_blank())
  
  lc_data_long <- pivot_longer(results$learning_curves, 
                              cols = c(train_accuracy, valid_accuracy),
                              names_to = "dataset", values_to = "accuracy")
  
  lc_plot <- ggplot(lc_data_long, aes(x = percentage, y = accuracy, color = dataset)) +
    geom_line(size = 1) +
    geom_point(size = 3) +
    labs(title = "Learning Curves", x = "Training Set Size (%)", y = "Accuracy") +
    scale_color_manual(values = c("train_accuracy" = "blue", "valid_accuracy" = "red"),
                       labels = c("train_accuracy" = "Training", "valid_accuracy" = "Validation")) +
    theme_minimal() +
    theme(legend.title = element_blank())
  
  rel_plot <- ggplot(results$reliability_data, aes(x = bin_midpoint, y = observed_freq)) +
    geom_point(aes(size = n_samples), color = "darkblue", alpha = 0.7) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Reliability Diagram", 
         subtitle = paste("Brier Score:", round(results$brier_score, 4)),
         x = "Predicted Probability", y = "Observed Frequency") +
    scale_size_continuous(name = "# Samples") +
    coord_equal() +
    theme_minimal()
  
  grid.arrange(cv_plot, lc_plot, rel_plot, ncol = 2)
  
  list(
    cv_plot = cv_plot,
    lc_plot = lc_plot,
    rel_plot = rel_plot
  )
}

###############################################################################
# 8) Run Model Validation Analysis
###############################################################################
validation_results <- validate_model(train_model_df, xgb_final, test_fe, final_params)
plots <- plot_validation_results(validation_results)

cat("\nValidation analysis complete.\n")
