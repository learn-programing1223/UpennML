All Code Under Whole Rating Branch

NCAA Tournament Predictor
This repository contains an end-to-end workflow for building and validating a machine learning model to predict NCAA basketball game outcomes. It includes:

Data Reading & Cleaning

Whole-History Rating (WHR) calculation

Feature Engineering

Bayesian Optimization of an XGBoost Model

Model Validation (Cross-Validation, Learning Curves, Calibration)

Seeding for Unseeded Regions

Table of Contents
Prerequisites

File Descriptions

Running the Script

Explanation of Key Steps

Results and Outputs

Troubleshooting

License

Prerequisites
R version 4.0 or above (Recommended)

The following R packages (install via install.packages("<package-name>")):

tidyverse

caret

xgboost

rBayesianOptimization

ggplot2

gridExtra

Basic knowledge of how to run R scripts from the command line or within RStudio.

File Descriptions
games_2022.csv
Raw training dataset containing game-level information for the 2022 season.
Columns include:

team (Team Name)

team_score (Team’s final score)

opponent_team_score (Opponent’s final score)

home_away (Indicator of whether the team was home or away)

... plus additional stats like FGM_2, FGA_2, etc.

East Regional Games to predict.csv
Test set containing games to predict in the East Region. The script will output predictions (home win probabilities) for these games.

Script (R code)
The main code that:

Reads the data

Computes Whole-History Ratings

Engineers features (offensive/defensive efficiency, rating differences, etc.)

Trains and optimizes an XGBoost model using Bayesian Optimization

Validates the model via cross-validation, learning curves, reliability diagrams

Predicts outcomes on the test data

Assigns seeds to unseeded regions based on WHR ratings

Running the Script
Clone or Download this repository.

Place your CSV files (games_2022.csv and East Regional Games to predict.csv) in the same directory as the script.

Open R or RStudio and set your working directory to the location of this script and data files.

Install any missing packages:

r
Copy
install.packages(c("tidyverse", "caret", "xgboost", "rBayesianOptimization", "ggplot2", "gridExtra"))
Run the script:

r
Copy
source("path/to/your_script.R")
or

r
Copy
Rscript your_script.R
The script will print progress and results to your console.

Explanation of Key Steps
Whole-History Rating (WHR)

Computes an Elo-like rating for each team based on all their matches.

A margin-of-victory function (my_margin_func) further refines rating adjustments.

Feature Engineering

Merges WHR ratings, shooting percentages, turnovers, rebounds, etc. into a single row per game.

Constructs ratios and differences (e.g., diff_FG2, diff_FT, ratio_AST) for modeling.

Bayesian Optimization

Utilizes rBayesianOptimization to search hyperparameter space (max_depth, learning_rate, etc.).

The best combination of hyperparameters is used to build the final XGBoost model.

Model Validation

Cross-Validation: Uses 5-fold CV to estimate accuracy and detect overfitting.

Learning Curves: Monitors how the model’s performance changes with more training data.

Calibration: Uses a reliability diagram and Brier score to see how well predicted probabilities match actual outcomes.

Prediction & Seeding

Generates home-win predictions on the test set (East Regional Games to predict.csv).

Seeds unseeded regions (North, South, West) by sorting teams based on their WHR rating.

Results and Outputs
Model Accuracy

Printed to the console for both training and validation sets.

Test Predictions

Shows predicted_class_home_win and predicted_prob_home_win for the test games.

Validation Plots

Cross-Validation results

Learning curves

Reliability diagram

Seeding Information

Prints a table with seeds assigned to each team in the North, South, and West regions.

Troubleshooting
Package Not Found
Make sure you have installed all required packages and you have the latest version of R.

File Not Found
Verify that games_2022.csv and East Regional Games to predict.csv are in your working directory.

Bayesian Optimization Errors
Ensure the rBayesianOptimization package is installed correctly and updated.

Memory or Computation Time
Bayesian optimization and cross-validation can be intensive. Reduce the number of trials or folds if runtime is too long.

