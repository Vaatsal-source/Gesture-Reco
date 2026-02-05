import cv2
import mediapipe as mp
import numpy as np
import os
import time

GESTURES = {'fist': 0, 'open_palm': 1, 'thumb_up': 2, 'thumb_left': 3, 'thumb_right': 4}
SAVE_DIR = "dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

current_gesture = input(f"Enter gesture name {list(GESTURES.keys())}: ")
print("Get ready... recording starts in 3 seconds.")
time.sleep(3)

data = []
labels = []

while len(data) < 500: 
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        
        wrist = hand.landmark[0]
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        data.append(landmarks)
        labels.append(GESTURES[current_gesture])
        cv2.putText(frame, f"Samples: {len(data)}/500", (10, 30), 1, 2, (0,255,0), 2)

    cv2.imshow("Collecting - Press 'q' to stop early", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

np.save(f"{current_gesture}_X.npy", np.array(data))
np.save(f"{current_gesture}_y.npy", np.array(labels))
print(f"Saved {len(data)} samples for {current_gesture}")