import joblib
import numpy as np
import re
from bs4 import BeautifulSoup
from tensorflow.keras.models import load_model

# Load the trained SVM model
svm_model = joblib.load("models/resume_classifier_svm.pkl")

# Load the TF-IDF vectorizer
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load the label mapping (saved as an array of class names)
label_classes = np.load("models/label_classes.npy", allow_pickle=True)

def preprocess_text(text):
    """
    Preprocess resume text by removing HTML tags, non-word characters,
    converting to lower case, and stripping extra spaces.
    """
    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    # Remove non-word characters
    text = re.sub(r'\W+', ' ', text)
    # Lower-case and strip extra whitespace
    return text.lower().strip()

def predict_category(resume_text):
    """
    Preprocess the resume, convert it to TF-IDF features, and predict the category using the SVM model.
    If the prediction is numeric, it will be converted to the corresponding label using the label mapping.
    Otherwise, the prediction is assumed to be a string and is returned directly.
    """
    preprocessed = preprocess_text(resume_text)
    features = tfidf_vectorizer.transform([preprocessed])
    features_dense = features.toarray()  # Convert sparse to dense
    pred = svm_model.predict(features_dense)
    
    # Check if prediction is numeric; if not, assume it's already the label string.
    if isinstance(pred[0], (int, np.integer)):
        predicted_label = label_classes[int(pred[0])]
    else:
        predicted_label = pred[0]
        
    return predicted_label

if __name__ == "__main__":
    # Example resume text
    sample_resume = (
        "Experienced software developer with expertise in Python, machine learning, "
        "and data analysis. Proven track record of developing scalable solutions."
    )
    category = predict_category(sample_resume)
    print("Predicted Category:", category)
