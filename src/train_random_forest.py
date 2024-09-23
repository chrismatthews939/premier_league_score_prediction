# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os

# Get file paths, so full path doesn't need to be explict 
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, '../data/football_data_cleaned.csv')

# Load the dataset (use the cleaned and feature-engineered data)
data = pd.read_csv(data_file)

# Features (X) - selecting relevant columns (you may add more relevant features based on the dataset)
X = data[['Goal_Diff', 'Home_Accuracy', 'Away_Accuracy', 'Total_Goals']]  # Example feature set

# Target variables: Home goals (FTHG) and Away goals (FTAG)
y_home = data['FTHG']  # Full-time Home Goals
y_away = data['FTAG']  # Full-time Away Goals

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_home_train, y_home_test = train_test_split(X, y_home, test_size=0.2, random_state=42)
_, _, y_away_train, y_away_test = train_test_split(X, y_away, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor with regularization to avoid overfitting
rf_home = RandomForestRegressor(
    n_estimators=50,  # fewer trees to prevent overfitting
    max_depth=5,  # limit depth of trees
    min_samples_split=10,  # more samples required to split nodes
    min_samples_leaf=5,  # leaf nodes must contain at least 5 samples
    random_state=42
)

# Train the model on the training data (home team goals)
rf_home.fit(X_train, y_home_train)

# Cross-validation to evaluate model performance
cv_scores_home = cross_val_score(rf_home, X_train, y_home_train, cv=5, scoring='neg_mean_absolute_error')

# Predict home goals using the test set
y_home_pred = rf_home.predict(X_test)

# Evaluate the model using mean absolute error (MAE) and R-squared (R2)
mae_home = mean_absolute_error(y_home_test, y_home_pred)
r2_home = r2_score(y_home_test, y_home_pred)

print(f"Home Goals - Mean Absolute Error (MAE): {mae_home:.2f}")
print(f"Home Goals - R-squared (R2): {r2_home:.2f}")
print(f"Home Goals - Cross-validation MAE: {-cv_scores_home.mean():.2f}")

# Initialize the Random Forest Regressor for predicting away goals
rf_away = RandomForestRegressor(
    n_estimators=50,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)

# Train the model on the training data (away team goals)
rf_away.fit(X_train, y_away_train)

# Cross-validation to evaluate model performance
cv_scores_away = cross_val_score(rf_away, X_train, y_away_train, cv=5, scoring='neg_mean_absolute_error')

# Predict away goals using the test set
y_away_pred = rf_away.predict(X_test)

# Evaluate the model using MAE and R-squared
mae_away = mean_absolute_error(y_away_test, y_away_pred)
r2_away = r2_score(y_away_test, y_away_pred)

print(f"Away Goals - Mean Absolute Error (MAE): {mae_away:.2f}")
print(f"Away Goals - R-squared (R2): {r2_away:.2f}")
print(f"Away Goals - Cross-validation MAE: {-cv_scores_away.mean():.2f}")

# Save the models for future use
with open(os.path.join(base_dir, '../models/rf_home_model.pkl'), 'wb') as f:
    pickle.dump(rf_home, f)

with open(os.path.join(base_dir, '../models/rf_away_model.pkl'), 'wb') as f:
    pickle.dump(rf_away, f)
