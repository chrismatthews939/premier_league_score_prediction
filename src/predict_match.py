# Import necessary libraries
import pandas as pd
import pickle
import numpy as np
import os

# Get file paths, so full path doesn't need to be explict 
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, '../data/football_data_cleaned.csv')

# Load the saved model and scaler
with open(os.path.join(base_dir, '../models/logistic_regression_model.pkl'), 'rb') as f:
    model = pickle.load(f)

with open(os.path.join(base_dir, '../models/scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Load historical data to estimate features
data = pd.read_csv(data_file)

# Set the teams for prediction
home_team = 'Chelsea'
away_team = 'Man City'

# Function to calculate team averages based on new column structure
def calculate_team_averages(team_name, data):
    # Identify home and away columns for the team
    home_column = f'HomeTeam_{team_name}'
    away_column = f'AwayTeam_{team_name}'
    
    # Filter home and away games for the team
    home_games = data[data[home_column] == True]
    away_games = data[data[away_column] == True]
    
    # Calculate average goals and shooting accuracy for home and away games
    avg_goals_home = home_games['FTHG'].mean()  # Home goals scored
    avg_goals_away = away_games['FTAG'].mean()  # Away goals scored
    avg_goals = (avg_goals_home + avg_goals_away) / 2
    
    avg_shots_home = home_games['HS'].mean()    # Home shots
    avg_shots_away = away_games['AS'].mean()    # Away shots
    avg_shots = (avg_shots_home + avg_shots_away) / 2
    
    avg_shots_on_target_home = home_games['HST'].mean()  # Home shots on target
    avg_shots_on_target_away = away_games['AST'].mean()  # Away shots on target
    avg_shots_on_target = (avg_shots_on_target_home + avg_shots_on_target_away) / 2
    
    # Calculate shooting accuracy
    avg_accuracy = avg_shots_on_target / avg_shots if avg_shots > 0 else 0
    
    return avg_goals, avg_accuracy

# Get average stats for each team
home_avg_goals, home_avg_accuracy = calculate_team_averages(home_team, data)
away_avg_goals, away_avg_accuracy = calculate_team_averages(away_team, data)

# Calculate the features for the match
goal_diff = home_avg_goals - away_avg_goals
total_goals = home_avg_goals + away_avg_goals

# Create the feature array for prediction
features = np.array([[goal_diff, home_avg_accuracy, away_avg_accuracy, total_goals]])

# Scale the features
features_scaled = scaler.transform(features)

# Predict the result
prediction = model.predict(features_scaled)
prediction_proba = model.predict_proba(features_scaled)

# Map prediction to result
result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
predicted_result = result_map[prediction[0]]

# Output the prediction
print(f"Predicted result for {home_team} vs {away_team}: {predicted_result}")
print(f"Prediction probabilities: {prediction_proba}")
