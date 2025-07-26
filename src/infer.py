import cv2
import joblib
import numpy as np


model = joblib.load(r"models\svm_eye_model.pkl")
IMG_SIZE = (24, 24) 
FONT = cv2.FONT_HERSHEY_SIMPLEX


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


cap = cv2.VideoCapture(0)

print("[INFO] Starting webcam... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in eyes:
        eye = gray[y:y+h, x:x+w]

        
        eye_resized = cv2.resize(eye, IMG_SIZE).flatten().reshape(1, -1)

        # Predict
        pred = model.predict(eye_resized)[0]
        label = "Open" if pred == 1 else "Closed"

        
        color = (0, 255, 0) if pred == 1 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), FONT, 0.6, color, 2)

    
    cv2.imshow("Attention Detection", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
