import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load TF-IDF features & labels
X = pd.read_csv("dataset/tfidf_features.csv")
df = pd.read_csv("dataset/cleaned_resumes.csv")

# Extract labels
y = df['Category']

# Split data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a Logistic Regression Model
print("🔄 Training model...")
model = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial')
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Print Classification Report
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# Save Model
os.makedirs("models", exist_ok=True)
with open("models/resume_classifier_lr.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved in `models/resume_classifier_lr.pkl`")
