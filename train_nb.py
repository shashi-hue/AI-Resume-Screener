import numpy as np
import pandas as pd
import pickle
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Load extracted TF-IDF features
X = pd.read_csv("dataset/tfidf_features.csv").values  # Load features
y = pd.read_csv("dataset/labels.csv")["Category"].values  # Load labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Naïve Bayes model
print("🔄 Training Naïve Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred_nb = model.predict(X_test)

# Classification Report
print("📊 Classification Report:")
print(classification_report(y_test, y_pred_nb))

# Save Model
os.makedirs("models", exist_ok=True)
with open("models/resume_classifier_nb.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Naïve Bayes Model saved in `models/resume_classifier_nb.pkl`")
