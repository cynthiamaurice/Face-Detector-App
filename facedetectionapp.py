import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Load face detection model
face_cascade = cv2.CascadeClassifier("C:\\Users\\personal\\Downloads\\haarcascade_frontalface_default  (1).xml")

def detect_faces_opencv(image):
    # Convert the frames to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image

def app():
    st.title("Face Detection App")
    st.write("Click the button below to start your webcam and detect faces.")

    # Toggle to start/stop webcam
    run = st.checkbox('Start Webcam')



    FRAME_WINDOW = st.image([])

    cap = None

    # Initialize the webcam
    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to access webcam.")
                break

                # Resize frame for display
            frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)

            frame = detect_faces_opencv(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame)

        cap.release()
    else:
        st.write("Click the button to start the webcam.")

if __name__ == "__main__":
    app()
