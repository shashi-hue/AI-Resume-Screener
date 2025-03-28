import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# Load features and labels
X = pd.read_csv("dataset/tfidf_features.csv").values
y = pd.read_csv("dataset/labels.csv")["Category"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize SVM model
model = SVC(kernel="linear")  # Linear kernel works well for text data

# Train the model
print("🔄 Training SVM model...")
model.fit(X_train, y_train)

# Predictions
y_pred_svm = model.predict(X_test)

# Evaluation
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_svm))
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

# Save the model
os.makedirs("models", exist_ok=True)
with open("models/resume_classifier_svm.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ SVM Model saved in `models/resume_classifier_svm.pkl`")
