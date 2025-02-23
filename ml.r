###############################################################################
# 0) Libraries, Seed, and Data Reading
###############################################################################
library(tidyverse)
library(caret)
library(xgboost)

set.seed(123)

# Define file paths (adjust as needed)
train_raw_path <- "games_2022.csv"
test_raw_path  <- "East Regional Games to predict.csv"

# Read data (ensuring valid R column names)
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
# 2a) Build Team-Level Averages for Imputation (for missing advanced stats)
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
    notD1 = as.numeric(any(notD1_incomplete))
  ) %>% ungroup()

# Compute team-level efficiency averages from aggregated training games
team_eff_home <- train %>%
  group_by(team_home) %>%
  summarize(
    off_eff_home = mean(score_home / (TOV_home + 1) * (1 + AST_home / 100), na.rm = TRUE),
    def_eff_home = mean((BLK_home + STL_home) / pmax(TOV_home, 1), na.rm = TRUE)
  ) %>% ungroup() %>% rename(team = team_home)

team_eff_away <- train %>%
  group_by(team_away) %>%
  summarize(
    off_eff_away = mean(score_away / (TOV_away + 1) * (1 + AST_away / 100), na.rm = TRUE),
    def_eff_away = mean((BLK_away + STL_away) / pmax(TOV_away, 1), na.rm = TRUE)
  ) %>% ungroup() %>% rename(team = team_away)

###############################################################################
# 3) Compute WHR (Team Ratings) on Training Data
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
train_fe <- train %>%
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
    ratio_REB = (DREB_home + OREB_home) / pmax(DREB_away + OREB_away, 1)
  ) %>%
  # Compute efficiency features using actual game data
  mutate(
    off_eff_home = (score_home / (TOV_home + 1)) * (1 + AST_home / 100),
    off_eff_away = (score_away / (TOV_away + 1)) * (1 + AST_away / 100),
    def_eff_home = (BLK_home + STL_home) / pmax(TOV_home, 1),
    def_eff_away = (BLK_away + STL_away) / pmax(TOV_away, 1)
  ) %>%
  mutate(
    across(c(notD1_incomplete_home, notD1_incomplete_away), as.numeric),
    home_away_NS = as.numeric(home_away_NS)
  )

# Select training features (34 columns)
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
# 5) Split Data and Train XGBoost Model (via caret)
###############################################################################
set.seed(123)
trainIndex <- createDataPartition(train_model_df$home_win, p = 0.8, list = FALSE)
train_split <- train_model_df[trainIndex, ]
valid_split <- train_model_df[-trainIndex, ]

train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  savePredictions = "final",
  verboseIter = FALSE
)

# Use a tuning grid that previously yielded high accuracy
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(3, 6),
  eta = c(0.01, 0.05),
  gamma = c(0, 1),
  colsample_bytree = c(0.8, 1.0),
  min_child_weight = c(1, 5),
  subsample = c(0.8, 1.0)
)

set.seed(123)
xgb_tuned <- train(
  home_win ~ .,
  data = train_split,
  method = "xgbTree",
  metric = "Accuracy",
  trControl = train_control,
  tuneGrid = xgb_grid
)

cat("\nBest Tuning Parameters:\n")
print(xgb_tuned$bestTune)

cat("\nTraining Results:\n")
print(xgb_tuned)

# Evaluate performance on training and validation splits
train_pred_class <- predict(xgb_tuned, newdata = train_split, type = "raw")
train_cm <- confusionMatrix(train_pred_class, train_split$home_win)
cat("\nTraining Accuracy:", round(train_cm$overall["Accuracy"] * 100, 2), "%\n")

valid_pred_class <- predict(xgb_tuned, newdata = valid_split, type = "raw")
valid_cm <- confusionMatrix(valid_pred_class, valid_split$home_win)
cat("\nValidation Accuracy:", round(valid_cm$overall["Accuracy"] * 100, 2), "%\n")

###############################################################################
# 6) Prepare Test Data – Feature Engineering and Imputation
###############################################################################
# The test file ("East Regional Games to predict.csv") has different columns:
# game_id, description, team_home, team_away, seed_home, seed_away,
# home_away_NS, rest_days_Home, rest_days_Away, travel_dist_Home, travel_dist_Away, WINNING_%
# First, rename columns to match our training names.
test <- test_raw %>%
  rename(
    rest_days_home = rest_days_Home,
    rest_days_away = rest_days_Away,
    travel_dist_home = travel_dist_Home,
    travel_dist_away = travel_dist_Away
  )

# For test data, advanced shooting stats and game scores are not available.
# We impute missing shooting stats by joining team_stats (for FG2, FG3, FT, etc.)
test_fe <- test %>%
  left_join(team_stats, by = c("team_home" = "team")) %>%
  rename_at(vars(FG2_pct, FG3_pct, FT_pct, AST_avg, BLK_avg, STL_avg, TOV_avg, DREB_avg, OREB_avg, notD1),
            ~ paste0(., "_home")) %>%
  left_join(team_stats, by = c("team_away" = "team")) %>%
  rename_at(vars(FG2_pct, FG3_pct, FT_pct, AST_avg, BLK_avg, STL_avg, TOV_avg, DREB_avg, OREB_avg, notD1),
            ~ paste0(., "_away")) %>%
  rename(
    FG2_percentage_home = FG2_pct_home,
    FG3_percentage_home = FG3_pct_home,
    FT_percentage_home  = FT_pct_home,
    AST_home = AST_avg_home,
    BLK_home = BLK_avg_home,
    STL_home = STL_avg_home,
    TOV_home = TOV_avg_home,
    DREB_home = DREB_avg_home,
    OREB_home = OREB_avg_home,
    notD1_incomplete_home = notD1_home,
    
    FG2_percentage_away = FG2_pct_away,
    FG3_percentage_away = FG3_pct_away,
    FT_percentage_away  = FT_pct_away,
    AST_away = AST_avg_away,
    BLK_away = BLK_avg_away,
    STL_away = STL_avg_away,
    TOV_away = TOV_avg_away,
    DREB_away = DREB_avg_away,
    OREB_away = OREB_avg_away,
    notD1_incomplete_away = notD1_away
  ) %>%
  # Compute differences and ratios for shooting stats
  mutate(
    diff_FG2 = FG2_percentage_home - FG2_percentage_away,
    diff_FG3 = FG3_percentage_home - FG3_percentage_away,
    diff_FT  = FT_percentage_home - FT_percentage_away,
    ratio_AST = AST_home / pmax(AST_away, 1),
    ratio_BLK = BLK_home / pmax(BLK_away, 1),
    ratio_STL = STL_home / pmax(STL_away, 1),
    ratio_REB = (DREB_home + OREB_home) / pmax(DREB_away + OREB_away, 1)
  ) %>%
  # Impute WHR ratings using the training WHR values
  mutate(
    rating_home = whr_ratings[team_home],
    rating_away = whr_ratings[team_away],
    diff_rating = rating_home - rating_away,
    home_away_NS = as.numeric(home_away_NS)
  )

# Impute efficiency features for test data using team-level averages computed on training:
# Join the precomputed team efficiencies for home and away.
test_fe <- test_fe %>%
  left_join(team_eff_home, by = c("team_home" = "team")) %>%
  left_join(team_eff_away, by = c("team_away" = "team"))

# For test games, we now have off_eff_home, def_eff_home, off_eff_away, def_eff_away imputed.
# Define test_features using the same columns as used in training.
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
test_pred_class <- predict(xgb_tuned, newdata = test_features, type = "raw")
test_pred_prob  <- predict(xgb_tuned, newdata = test_features, type = "prob")

# Append predictions to the test feature set
test_results <- test_fe %>%
  mutate(
    predicted_class_home_win = test_pred_class,
    predicted_prob_home_win  = test_pred_prob[,"Win"]
  )

cat("\nFinal Test Predictions:\n")
print(
  test_results %>% 
    select(team_home, team_away, rest_days_home, rest_days_away,
           travel_dist_home, travel_dist_away,
           predicted_class_home_win, predicted_prob_home_win)
)

cat("\nSCRIPT COMPLETE.\n")
