import streamlit as st
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import imutils
import time
import dlib
import cv2
import playsound
import tempfile

# Function to sound the alarm
def sound_alarm(path):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        playsound.playsound(path)
    if alarm_status2:
        saying = True
        playsound.playsound(path)
        saying = False

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate final eye aspect ratio
def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

# Function to calculate lip distance
def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

# Streamlit UI setup
st.title("Drowsiness and Yawning Detection")
st.write("Upload a video to analyze for drowsiness or yawning.")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
alarm_path = st.text_input("Alarm Sound Path", "alert.wav")

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    # Load Haar Cascade and dlib predictor
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default(1).xml")
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in rects:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            ear, leftEye, rightEye = final_ear(shape)
            distance = lip_distance(shape)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            lip = shape[48:60]
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not alarm_status:
                        alarm_status = True
                        if alarm_path:
                            t = Thread(target=sound_alarm, args=(alarm_path,))
                            t.daemon = True
                            t.start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            else:
                COUNTER = 0
                alarm_status = False

            if distance > YAWN_THRESH:
                cv2.putText(frame, "Yawn Alert", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not alarm_status2 and not saying:
                    alarm_status2 = True
                    if alarm_path:
                        t = Thread(target=sound_alarm, args=(alarm_path,))
                        t.daemon = True
                        t.start()
            else:
                alarm_status2 = False

            cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        if st.button("Capture Snapshot"):
            snapshot_path = "snapshot.jpg"
            cv2.imwrite(snapshot_path, frame)
            st.success(f"Snapshot saved as {snapshot_path}")

    cap.release()
