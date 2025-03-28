import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load features and labels from CSV
X = pd.read_csv("dataset/tfidf_features.csv").values
y = pd.read_csv("dataset/labels.csv")["Category"].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
print("🔄 Training Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred_rf = rf_model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred_rf))

# Save model
with open("models/resume_classifier_rf.pkl", "wb") as f:
    pickle.dump(rf_model, f)

print("✅ RF Model saved in `models/resume_classifier_rf.pkl`")
