from scipy.spatial import distance as dist
import numpy as np
import cv2

def calculate_ear(eye):
    # Compute Eye Aspect Ratio
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def sound_alert():
    print("DROWSINESS DETECTED!")  # You can use winsound.Beep on Windows
