import cv2
import mediapipe
import pandas as pd
from utils import *

if __name__ == "__main__":

    sample_list = []
    num_samples = 1000
    recording = False

    blue_color = (245, 25, 16)
    red_color = (24, 44, 245)
    color = blue_color

    cap = cv2.VideoCapture(0)

    with mediapipe.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections
            frame, results = mediapipe_detection(frame, hands)

            landmarks = results.multi_hand_landmarks

            # Draw landmarks
            draw_landmarks(frame, landmarks)

            if recording and len(sample_list) < num_samples:
                sample_name = f"sample_{len(sample_list):05d}"
                if landmarks:
                    sample = mp_landmarks_to_np_array(landmarks[0]).reshape(63).tolist()
                    sample_list.append([sample_name] + sample)

                color = red_color
                print(f"Recording: {len(sample_list)}/{num_samples}", end="\r")

            elif recording and len(sample_list) == num_samples:
                pd.DataFrame(sample_list).to_csv("hand_gesture.csv", sep=",")

                recording = False
                sample_list = []
                color = blue_color

            # REC circle
            cv2.circle(frame, (30, 30), 20, color, -1)

            # Show to screen
            frame = cv2.flip(frame, 1)
            cv2.imshow("OpenCV Feed", frame)

            # Break pressing q
            if cv2.waitKey(5) == ord("q"):
                break

            # Record pressing s
            if cv2.waitKey(5) == ord("s"):
                recording = True

        cap.release()
        cv2.destroyAllWindows()
