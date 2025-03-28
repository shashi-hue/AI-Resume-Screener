import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load TF-IDF features (assumes you saved them as a .npy file)
X = np.load("dataset/tfidf_features.npy", allow_pickle=True)

# Load labels from CSV using pandas (do not use np.load for CSV)
labels_df = pd.read_csv("dataset/labels.csv")  # Assumes the file has a header row, e.g., "Category"
y = labels_df["Category"].values  # Extract the labels column

# Encode string labels into numeric values
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Save the encoded labels and the label mapping for later use
np.save("dataset/labels.npy", y_encoded)
np.save("dataset/label_classes.npy", encoder.classes_)  # This saves the mapping of numbers to original labels

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Save the train-test splits
np.save("dataset/tfidf_features_train.npy", X_train)
np.save("dataset/tfidf_features_test.npy", X_test)
np.save("dataset/labels_train.npy", y_train)
np.save("dataset/labels_test.npy", y_test)

print("✅ Dataset successfully split and saved!")
