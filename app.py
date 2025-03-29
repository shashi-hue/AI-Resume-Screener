from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load models and other assets
svm_model = joblib.load("models/resume_classifier_svm.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_classes = np.load("models/label_classes.npy", allow_pickle=True)

def preprocess_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\W+', ' ', text)
    return text.lower().strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Check if the form is submitted via HTML form or JSON
    if request.is_json:
        data = request.get_json(force=True)
        resume_text = data.get("resume", "")
    else:
        resume_text = request.form.get("resume", "")

    if not resume_text:
        return jsonify({"error": "No resume text provided."}), 400

    preprocessed = preprocess_text(resume_text)
    features = tfidf_vectorizer.transform([preprocessed]).toarray()
    pred = svm_model.predict(features)
    
    # If prediction is numeric, convert to label; otherwise assume it's a string.
    if isinstance(pred[0], (int, np.integer)):
        predicted_label = label_classes[int(pred[0])]
    else:
        predicted_label = pred[0]

    # If the request came via a form, render a new template with the result
    if not request.is_json:
        return render_template("result.html", predicted_category=predicted_label, resume=resume_text)
    
    return jsonify({"predicted_category": predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
