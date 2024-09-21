# Import necessary libraries
import pandas as pd
import pickle
import numpy as np
import os

# Get file paths, so full path doesn't need to be explict 
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, '../data/football_data_cleaned.csv')

# Load the saved model and scaler
with open(os.path.join(base_dir, '../models/xgboost_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(base_dir, '../models/scaler_xgboost.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Load historical data to estimate features
data = pd.read_csv(data_file)

# Function to calculate team averages (same as before)
def calculate_team_averages(team_name, data):
    home_column = f'HomeTeam_{team_name}'
    away_column = f'AwayTeam_{team_name}'
    
    home_games = data[data[home_column] == True]
    away_games = data[data[away_column] == True]
    
    avg_goals_home = home_games['FTHG'].mean()  
    avg_goals_away = away_games['FTAG'].mean()  
    avg_goals = (avg_goals_home + avg_goals_away) / 2
    
    avg_shots_home = home_games['HS'].mean()    
    avg_shots_away = away_games['AS'].mean()    
    avg_shots = (avg_shots_home + avg_shots_away) / 2
    
    avg_shots_on_target_home = home_games['HST'].mean()  
    avg_shots_on_target_away = away_games['AST'].mean()  
    avg_shots_on_target = (avg_shots_on_target_home + avg_shots_on_target_away) / 2
    
    avg_accuracy = avg_shots_on_target / avg_shots if avg_shots > 0 else 0
    
    return avg_goals, avg_accuracy

# Predict the result for a specific match
home_team = 'West Ham'
away_team = 'Chelsea'

# Estimate features for the upcoming match
home_avg_goals, home_avg_accuracy = calculate_team_averages(home_team, data)
away_avg_goals, away_avg_accuracy = calculate_team_averages(away_team, data)

goal_diff = home_avg_goals - away_avg_goals
total_goals = home_avg_goals + away_avg_goals

# Create the feature array for prediction
features = np.array([[goal_diff, home_avg_accuracy, away_avg_accuracy, total_goals]])

# Scale the features
features_scaled = scaler.transform(features)

# Predict the result using XGBoost
prediction = model.predict(features_scaled)
prediction_proba = model.predict_proba(features_scaled)

# Map prediction to result
result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
predicted_result = result_map[prediction[0]]

# Output the prediction
print(f"Predicted result for {home_team} vs {away_team} using XGBoost: {predicted_result}")
print(f"Prediction probabilities: {prediction_proba}")
