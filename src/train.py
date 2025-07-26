# src/train.py

import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# Paths to preprocessed data and model output
DATA_PATH = "data/processed/mini_data.pkl"
MODEL_PATH = "models/svm_eye_model.pkl"

def load_data():
    print("[INFO] Loading data...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} does not exist. Run preprocess.py first.")
    
    with open(DATA_PATH, 'rb') as f:
        X, y = pickle.load(f)

    print(f"[INFO] Loaded {len(X)} samples.")
    X = np.array([img.flatten() for img in X]) 
    y = np.array(y)
    return X, y

def train_model():
    print("[INFO] Starting model training...")
    X, y = load_data()

    # Splitting data
    print("[INFO] Splitting dataset into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creating a pipeline with scaler and linear SVM
    print("[INFO] Initializing LinearSVC model...")
    model = make_pipeline(StandardScaler(), LinearSVC(max_iter=10000))

    # Training model
    print("[INFO] Training model...")
    model.fit(X_train, y_train)

    # Evaluating model
    accuracy = model.score(X_val, y_val)
    print(f"[INFO] Validation Accuracy: {accuracy:.4f}")

    # Saving model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
