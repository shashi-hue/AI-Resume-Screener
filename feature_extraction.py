import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

# Load cleaned resumes
df = pd.read_csv("dataset/cleaned_resumes.csv")

# 🔍 Handle NaN Values
df['Cleaned_Resume'] = df['Cleaned_Resume'].fillna(" ")

# Apply TF-IDF Vectorization
print("🔄 Running TF-IDF feature extraction...")
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['Cleaned_Resume']).toarray()

# Ensure dataset folder exists
os.makedirs("dataset", exist_ok=True)

# ✅ Save extracted features
pd.DataFrame(X).to_csv("dataset/tfidf_features.csv", index=False)

# ✅ Save labels separately
df[['Category']].to_csv("dataset/labels.csv", index=False)  # 🔥 FIXED

# ✅ Save TF-IDF model for future reuse
os.makedirs("models", exist_ok=True)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ TF-IDF extraction complete! Features saved in `dataset/tfidf_features.csv` and `dataset/labels.csv`")
