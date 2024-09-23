# Import necessary libraries
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# Scale the features (best practice for tree-based methods)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize XGBoost Classifier with regularization to prevent overfitting
xgb_model = xgb.XGBClassifier(
    objective='multi:softprob',  # Multi-class classification
    num_class=3,                 # 3 possible outcomes: Home Win, Draw, Away Win
    random_state=42,             # Reproducibility
    max_depth=4,                 # Limit the depth of the trees (regularization)
    n_estimators=100,            # Reduce the number of trees
    learning_rate=0.1,           # Learning rate (smaller value for slower training)
    reg_alpha=0.1,               # L1 regularization term on weights (encourages sparsity)
    reg_lambda=1,                # L2 regularization term on weights (prevents overfitting)
    subsample=0.8,               # Subsample ratio (randomly samples 80% of data for each tree)
    colsample_bytree=0.8,        # Subsample ratio of columns for each tree
    early_stopping_rounds=10     # Stop training if there's no improvement after 10 rounds
)

# Use early stopping during training to prevent overfitting
evals = [(X_test_scaled, y_test)]  # Evaluation set
evals_result = {}  # To store the evaluation results

# Train the model with early stopping
xgb_model.fit(
    X_train_scaled, 
    y_train, 
    eval_set=evals,              # Pass the evaluation set
    early_stopping_rounds=10,    # Stop training if no improvement after 10 rounds
    verbose=True
)

# Save the evaluation results (optional, for analysis)
print("Evaluation history:", xgb_model.evals_result())

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(X_test_scaled)

# Evaluate model performance
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
class_report_xgb = classification_report(y_test, y_pred_xgb)

# Print the results
print(f"XGBoost Model Accuracy: {accuracy_xgb:.2f}")
print("XGBoost Confusion Matrix:\n", conf_matrix_xgb)
print("XGBoost Classification Report:\n", class_report_xgb)

# Save the XGBoost model and scaler for future use
with open(os.path.join(base_dir,'../models/xgboost_model.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)

with open(os.path.join(base_dir, '../models/scaler_xgboost.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
