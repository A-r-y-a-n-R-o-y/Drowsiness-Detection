import cv2
import dlib
import numpy as np
import pygame
from scipy.spatial import distance as dist
from imutils import face_utils
from twilio.rest import Client
import time
import os

# Initialize pygame for sound alerts
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(r"C:\IVP Project 2\siren-alert-96052.mp3")  # Update path if needed

def play_sound_alert():
    alert_sound.play()
    time.sleep(2)  # Pause to allow the sound to finish

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
model_path = r"models/shape_predictor_68_face_landmarks.dat"  # Update path if needed
if not os.path.exists(model_path):
    raise RuntimeError(f"Unable to find shape_predictor_68_face_landmarks.dat at {model_path}")
predictor = dlib.shape_predictor(model_path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks):
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype="double")

    size = (640, 480)
    focal_length = size[0]
    center = (size[0] / 2, size[1] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    image_points = np.array([
        (landmarks[30][0], landmarks[30][1]),  # Nose tip
        (landmarks[8][0], landmarks[8][1]),      # Chin
        (landmarks[36][0], landmarks[36][1]),    # Left eye corner
        (landmarks[45][0], landmarks[45][1]),    # Right eye corner
        (landmarks[48][0], landmarks[48][1]),    # Left mouth corner
        (landmarks[54][0], landmarks[54][1])     # Right mouth corner
    ], dtype="double")

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # Squeeze the rotation vector to get a 1D array
    rotation_vector = np.squeeze(rotation_vector)
    head_tilt = abs(rotation_vector[0])  # Tilt in the X-axis (left-right)
    head_nod  = abs(rotation_vector[1])  # Nod in the Y-axis (up-down)

    return rotation_vector, translation_vector, head_tilt, head_nod

def send_sms_alert():
    account_sid = ''
    auth_token = ''
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body="Drowsiness detected! Please stay alert.",
        from_='',  # Your Twilio phone number
        to=''      # Recipient's phone number
    )

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

eye_close_counter = 0
yawn_counter = 0
last_alert_time = 0
alert_cooldown = 10  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

        mouth = landmarks[48:60]
        mar = mouth_aspect_ratio(mouth)

        rotation_vector, translation_vector, head_tilt, head_nod = get_head_pose(landmarks)

        if ear < 0.25:
            eye_close_counter += 1
        else:
            eye_close_counter = 0

        if mar > 0.6:
            yawn_counter += 1
        else:
            yawn_counter = 0

        current_time = time.time()
        if (eye_close_counter > 20 or yawn_counter > 10 or head_tilt > 15 or head_nod > 15) and (current_time - last_alert_time > alert_cooldown):
            print(f"ALERT: Eye counter: {eye_close_counter}, Yawn counter: {yawn_counter}, Head Tilt: {head_tilt}, Head Nod: {head_nod}")
            play_sound_alert()
            send_sms_alert()
            last_alert_time = current_time
            cv2.putText(frame, "Drowsiness Detected!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
