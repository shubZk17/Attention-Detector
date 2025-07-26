# app.py
from flask import Flask, render_template, Response
import cv2
import joblib
import numpy as np
from src.utils import detect_face_and_eyes

app = Flask(__name__)
model = joblib.load("models/svm_eye_model.pkl")
camera = cv2.VideoCapture(0)

def predict_eye(eye_img):
    eye_resized = cv2.resize(eye_img, (64, 64)).flatten().reshape(1, -1)
    return model.predict(eye_resized)[0]

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
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

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')  # you'll create this

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
