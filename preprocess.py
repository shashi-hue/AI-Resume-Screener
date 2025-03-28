import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("dataset/Resume.csv")


#df = df.head(500)  # Process only 500 resumes for now


# Function to clean resume text
STOPWORDS = set(stopwords.words('english'))  # Convert to a set for faster lookup

def clean_resume(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join(word for word in text.split() if word.lower() not in STOPWORDS)  # Faster stopword removal
    return text

# Apply cleaning function
df['Cleaned_Resume'] = df['Resume_html'].apply(clean_resume)

# Split dataset for training
X_train, X_test, y_train, y_test = train_test_split(df['Cleaned_Resume'], df['Category'], test_size=0.2, random_state=42)


#Progress
for i, text in enumerate(df['Resume_html']):
    df.at[i, 'Cleaned_Resume'] = clean_resume(text)
    if i % 100 == 0:
        print(f"Processed {i}/{len(df)} resumes...")



print("✅ Preprocessing Done. Data is ready for modeling!")

df.to_csv("dataset/cleaned_resumes.csv", index=False)

print("✅ Cleaned resumes saved to data/cleaned_resumes.csv")