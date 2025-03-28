import pandas as pd

# Load the TF-IDF feature set
X = pd.read_csv("dataset/tfidf_features.csv")

# Load original cleaned resumes to get labels
df = pd.read_csv("dataset/cleaned_resumes.csv")

# 🔍 Verify Data
print(f"✅ TF-IDF Shape: {X.shape}")
print(f"✅ Labels Available: {'Category' in df.columns}")
print(f"✅ Sample Labels: {df['Category'].unique() if 'Category' in df.columns else 'No Labels Found'}")
