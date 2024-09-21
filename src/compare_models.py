# Import necessary libraries
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Get file paths, so full path doesn't need to be explict 
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, '../data/football_data_cleaned.csv')

# Load the dataset (use the cleaned and feature-engineered data)
data = pd.read_csv(data_file)

# Prepare the features (X) and target variable (y)
X = data[['Goal_Diff', 'Home_Accuracy', 'Away_Accuracy', 'Total_Goals']]
y = data['Result']  # Target is the match result

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load the logistic regression model for comparison
with open(os.path.join(base_dir, '../models/logistic_regression_model.pkl'), 'rb') as f:
    logistic_model = pickle.load(f)

# Load the XGBoost model
with open(os.path.join(base_dir, '../models/xgboost_model.pkl'), 'rb') as f:
    xgb_model = pickle.load(f)

# Function to compare models
def compare_models(model1, model2, X_test_scaled, y_test):
    # Predict with both models
    y_pred_model1 = model1.predict(X_test_scaled)
    y_pred_model2 = model2.predict(X_test_scaled)
    
    # Calculate accuracy for both models
    accuracy_model1 = accuracy_score(y_test, y_pred_model1)
    accuracy_model2 = accuracy_score(y_test, y_pred_model2)
    
    # Print accuracy comparison
    print(f"Accuracy of Model 1 (Logistic Regression): {accuracy_model1:.2f}")
    print(f"Accuracy of Model 2 (XGBoost): {accuracy_model2:.2f}")
    
    # Print classification reports
    print("\nClassification Report for Model 1 (Logistic Regression):")
    print(classification_report(y_test, y_pred_model1))
    
    print("\nClassification Report for Model 2 (XGBoost):")
    print(classification_report(y_test, y_pred_model2))

# Run the comparison
compare_models(logistic_model, xgb_model, X_test_scaled, y_test)