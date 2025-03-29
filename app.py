from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the trained SVM model, TF-IDF vectorizer, and label mapping
svm_model = joblib.load("models/resume_classifier_svm.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
label_classes = np.load("models/label_classes.npy", allow_pickle=True)

def preprocess_text(text):
    """
    Preprocess the resume text by removing HTML tags, non-word characters,
    converting to lower-case, and stripping extra spaces.
    """
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\W+', ' ', text)
    return text.lower().strip()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the resume text from form data (for HTML form submissions)
    resume_text = request.form.get("resume", "")
    if not resume_text:
        return render_template("index.html", error="Please enter resume text.")

    preprocessed = preprocess_text(resume_text)
    features = tfidf_vectorizer.transform([preprocessed]).toarray()
    pred = svm_model.predict(features)
    
    if isinstance(pred[0], (int, np.integer)):
        predicted_label = label_classes[int(pred[0])]
    else:
        predicted_label = pred[0]
    
    return render_template("result.html", predicted_category=predicted_label, resume=resume_text)

if __name__ == '__main__':
    app.run()  # Production: remove debug=True
