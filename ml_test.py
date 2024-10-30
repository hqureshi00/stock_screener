import numpy as np
from sklearn.datasets import load_iris
from xgboost import XGBClassifier

# Load some data
data = load_iris()
X, y = data.data, data.target

# Split data into train and validation sets
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = XGBClassifier(early_stopping_rounds = 10)

# Fit model with early stopping
model.fit(
    X_train, y_train, 
    eval_set=[(X_valid, y_valid)],
    verbose=True
)