import cv2
import mediapipe
import pandas as pd
from utils import *

if __name__ == '__main__':

    sample_list = []
    num_samples = 1000
    count = 0
    recording = False

    blue_color = (245, 25, 16)
    red_color = (24, 44, 245)
    color = blue_color

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mediapipe.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections
            frame, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_landmarks(frame, results)

            if recording and count < num_samples:
                sample_name = "sample_" + ("000" + str(count))[:-4]
                sample = mp_landmarks_to_np_array(results.right_hand).reshape(63).tolist()
                sample_list.append([sample_name] + sample)

                count += 1
                color = red_color

            elif recording and count == num_samples:
                pd.DataFrame(sample_list).to_csv("hand_gesture.csv", sep=',')
                count = 0
                recording = False
                sample_list = []
                color = blue_color

            # REC circle
            cv2.circle(frame, (30, 30), 20, color, -1)

            # Show to screen
            cv2.imshow('OpenCV Feed', frame)

            # Break pressing q
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

            # Record pressing s
            if cv2.waitKey(5) & 0xFF == ord('s'):
                recording = True

        cap.release()
        cv2.destroyAllWindows()
