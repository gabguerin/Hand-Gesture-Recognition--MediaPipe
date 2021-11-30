import cv2
import mediapipe as mp
import numpy as np


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, landmarks):
    if not landmarks:
        return

    mp_hands = mp.solutions.hands  # Holistic model
    mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

    # Draw hands
    mp_drawing.draw_landmarks(
        image,
        landmarks[0],
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(235, 52, 86), thickness=1, circle_radius=4),
        mp_drawing.DrawingSpec(color=(52, 55, 235), thickness=2, circle_radius=2),
    )


def mp_landmarks_to_np_array(mp_landmarks):
    landmarks = []
    for landmark in mp_landmarks.landmark:
        landmarks.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(landmarks)
