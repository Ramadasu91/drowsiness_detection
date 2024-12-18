import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
import os
from datetime import datetime

# Function to compute distance between points
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to determine if eyes are blinking
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0

# Streamlit UI
st.title("Drowsiness Detection from Video")
st.write("Upload a video to detect drowsiness and capture snapshots when detected.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save the uploaded file
    video_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Video uploaded successfully: {uploaded_file.name}")

    # Initialize dlib components
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)

    sleep = 0
    drowsy = 0
    active = 0
    status = ""
    color = (0, 0, 0)

    snapshot_dir = "snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            left_blink = blinked(landmarks[36], landmarks[37],
                                 landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43],
                                  landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = os.path.join(snapshot_dir, f"snapshot_{timestamp}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    st.write(f"Snapshot saved at {snapshot_path}")

            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 0, 255)

            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

    st.write("Processing completed.")
    st.write(f"Snapshots saved in the `{snapshot_dir}` directory.")
