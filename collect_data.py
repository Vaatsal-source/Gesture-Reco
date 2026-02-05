import cv2
import mediapipe as mp
import numpy as np
import os

GESTURES = {
    'fist': 0,
    'open_palm': 1,
    'thumb_up': 2,
    'thumb_left': 3,
    'thumb_right': 4
}

SAVE_DIR = "dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

cap = cv2.VideoCapture(0)

current_gesture = input("Enter gesture name: ")

data = []
labels = []

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        data.append(landmarks)
        labels.append(GESTURES[current_gesture])

        cv2.putText(frame, f"Samples: {len(data)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Collecting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

np.save(f"{current_gesture}_X.npy", np.array(data))
np.save(f"{current_gesture}_y.npy", np.array(labels))
