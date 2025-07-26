# src/inference_live.py
import cv2
import joblib
import numpy as np
from src.utils import detect_face_and_eyes

MODEL_PATH = "models/svm_eye_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_eye(eye_img):
    eye_resized = cv2.resize(eye_img, (64, 64)).flatten().reshape(1, -1)
    return model.predict(eye_resized)[0]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_face_and_eyes(frame)
    for (face_coords, eyes) in detections:
        for (ex, ey, ew, eh) in eyes:
            eye = frame[ey:ey+eh, ex:ex+ew]
            gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            label = predict_eye(gray_eye)
            text = "Focused" if label == 1 else "Distracted"
            cv2.putText(frame, text, (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow('Distraction Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
