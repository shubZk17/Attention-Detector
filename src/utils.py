import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_face_and_eyes(frame, grayscale=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if grayscale else frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_data = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_roi)
        face_data.append(((x, y, w, h), eyes))
    
    return face_data
