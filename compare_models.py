import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load test dataset
X_test = np.load("dataset/tfidf_features_test.npy", allow_pickle=True)
y_test = np.load("dataset/labels_test.npy", allow_pickle=True)

# Load label mapping (class names) if available
try:
    label_classes = np.load("models/label_classes.npy", allow_pickle=True)
except FileNotFoundError:
    print("Warning: label_classes.npy not found. Proceeding without label mapping.")
    label_classes = None

# Convert y_test to numeric if necessary
if y_test.dtype.kind in ['U', 'O']:
    if label_classes is not None:
        mapping = {str(label): i for i, label in enumerate(label_classes)}
        y_test = np.array([mapping[str(label)] for label in y_test])
    else:
        y_test = y_test.astype(int)
else:
    y_test = y_test.astype(int)

# Load trained models with correct filenames
models = {
    "Logistic Regression": joblib.load("models/resume_classifier_lr.pkl"),
    "Random Forest": joblib.load("models/resume_classifier_rf.pkl"),
    "SVM": joblib.load("models/resume_classifier_svm.pkl"),
    "Neural Network": load_model("models/resume_classifier_nn.keras")  # Keras model
}

results = {}

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    
    if model_name == "Neural Network":
        # For NN, get predicted class indices from probabilities
        y_pred = np.argmax(model.predict(X_test), axis=1)
    else:
        y_pred = model.predict(X_test)
    
    # Ensure predictions are 1-D
    y_pred = np.array(y_pred).reshape(-1)
    
    # If predictions are strings, convert them to numeric using label mapping
    if y_pred.dtype.kind in ['U', 'O']:
        if label_classes is not None:
            mapping = {str(label): i for i, label in enumerate(label_classes)}
            y_pred = np.array([mapping[str(pred)] for pred in y_pred])
        else:
            y_pred = y_pred.astype(int)
    else:
        y_pred = y_pred.astype(int)
    
    # Calculate accuracy and print classification report
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc
    
    print(f"✅ {model_name} Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Display overall model performance comparison
print("\n📊 Model Performance Comparison:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} with Accuracy {results[best_model]:.4f}")
