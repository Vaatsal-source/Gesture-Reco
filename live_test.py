import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("gesture_model.h5")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

GESTURE_NAMES = [
    "FIST", "OPEN", "UP", "LEFT", "RIGHT"
]

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

        X = np.array(landmarks).reshape(1, -1)
        pred = model.predict(X)
        gesture = GESTURE_NAMES[np.argmax(pred)]

        cv2.putText(frame, gesture, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
