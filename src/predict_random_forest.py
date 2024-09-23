import pickle
import pandas as pd
import os

# Get file paths, so full path doesn't need to be explict 
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, '../data/football_data_cleaned.csv')

# Load the saved model and scaler
with open(os.path.join(base_dir, '../models/rf_home_model.pkl'), 'rb') as f:
    rf_home = pickle.load(f)

with open(os.path.join(base_dir, '../models/rf_away_model.pkl'), 'rb') as f:
    rf_away = pickle.load(f)

# Function to predict score for a given match
def predict_match_score(home_team, away_team, goal_diff, home_acc, away_acc, total_goals):
    # Create a DataFrame for the match features
    match_data = pd.DataFrame({
        'Goal_Diff': [goal_diff],
        'Home_Accuracy': [home_acc],
        'Away_Accuracy': [away_acc],
        'Total_Goals': [total_goals]
    })
    
    # Predict home and away goals
    home_goals_pred = rf_home.predict(match_data)[0]
    away_goals_pred = rf_away.predict(match_data)[0]
    
    return round(home_goals_pred), round(away_goals_pred)

# Example prediction for Manchester United vs. Chelsea
home_team = 'Manchester United'
away_team = 'Chelsea'
goal_diff = 1.2  # Estimate based on past performance
home_acc = 0.78  # Estimate home team's accuracy
away_acc = 0.73  # Estimate away team's accuracy
total_goals = 2.8  # Estimated total goals in the match

home_score, away_score = predict_match_score(home_team, away_team, goal_diff, home_acc, away_acc, total_goals)

print(f"Predicted score for {home_team} vs. {away_team}: {home_team} {home_score} - {away_team} {away_score}")
