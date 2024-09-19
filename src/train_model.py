# Import necessary libraries
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Get file paths, so full path doesn't need to be explict 
base_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(base_dir, '../data/football_data_cleaned.csv')

# Load the dataset (use the cleaned and feature-engineered data)
data = pd.read_csv(data_file)

# Select the features and target variable
X = data[['Goal_Diff', 'Home_Accuracy', 'Away_Accuracy', 'Total_Goals']]
y = data['Result']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the size of train and test sets
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Standardize the feature data (scale it)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Logistic Regression model with L2 regularization (default)
# Reduce the value of 'C' to increase regularization
model = LogisticRegression(random_state=42, C=0.01)  # Smaller C means stronger regularization

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Save the model using pickle
# Snapshot of the model to be reused without the need to rerun
with open(os.path.join(base_dir, '../models/logistic_regression_model.pkl'), 'wb') as f:
    pickle.dump(model, f)

# Save the scaler for future use
with open(os.path.join(base_dir, '../models/scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
