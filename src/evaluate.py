import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from train import load_data, MODEL_PATH

def evaluate():
    X, y = load_data()
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("Classification Report:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    evaluate()
