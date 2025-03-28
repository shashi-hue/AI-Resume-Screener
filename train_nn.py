import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load data
X = np.loadtxt("dataset/tfidf_features.csv", delimiter=",")
y = pd.read_csv("dataset/labels.csv")

# Trim to the same length
min_samples = min(X.shape[0], y.shape[0])
X = X[:min_samples]
y = y.iloc[:min_samples, 0]

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Print label distribution
unique, counts = np.unique(y, return_counts=True)
print("Label distribution:", dict(zip(unique, counts)))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define an improved model
model = keras.Sequential([
    keras.layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),  # Increased neurons, reduced regularization
    keras.layers.Dropout(0.2),  # Reduced dropout
    keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(len(np.unique(y)), activation="softmax")  # Output layer
])

# Compile model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train model
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate model
accuracy = model.evaluate(X_test, y_test)[1]
print(f"✅ Improved Neural Network Model Accuracy: {accuracy:.4f}")

# Save model
model.save("models/resume_classifier_nn.keras")
np.save("models/label_classes.npy", label_encoder.classes_)
