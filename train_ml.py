import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import joblib

# Load dataset
data = pd.read_csv("dataset/soil_csv/soil_data.csv")

print("Dataset Loaded Successfully!\n")

# Features
X = data.drop("Output", axis=1)

# Target
y = data["Output"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", len(X_train))
print("Testing Data:", len(X_test))

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train model
print("\nTraining Random Forest Model...\n")

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy * 100)

# Save model
joblib.dump(model, "models/random_forest_model.pkl")

print("\nModel Saved Successfully!")