from flask import Flask, request, jsonify
import joblib
import numpy as np
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the trained SVM model
svm_model = joblib.load("models/resume_classifier_svm.pkl")

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load the label mapping (an array of class names)
label_classes = np.load("models/label_classes.npy", allow_pickle=True)

def preprocess_text(text):
    """
    Preprocess resume text by removing HTML tags, non-word characters,
    converting to lower case, and stripping extra spaces.
    """
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\W+', ' ', text)
    return text.lower().strip()

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json(force=True)
    resume_text = data.get("resume", "")
    if not resume_text:
        return jsonify({"error": "No resume text provided."}), 400

    # Preprocess and vectorize the input text
    preprocessed = preprocess_text(resume_text)
    features = tfidf_vectorizer.transform([preprocessed]).toarray()
    
    # Predict using the SVM model
    pred = svm_model.predict(features)
    
    # If prediction is numeric, convert it to the corresponding label
    if isinstance(pred[0], (int, np.integer)):
        predicted_label = label_classes[int(pred[0])]
    else:
        predicted_label = pred[0]
    
    return jsonify({"predicted_category": predicted_label})

if __name__ == '__main__':
    # Bind to 0.0.0.0 so the server is reachable externally if needed
    app.run(host="0.0.0.0", port=5000, debug=True)
