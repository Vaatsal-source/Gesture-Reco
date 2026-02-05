import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import pickle

model = tf.keras.models.load_model("gesture_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

GESTURE_NAMES = ["FIST", "OPEN", "UP", "LEFT", "RIGHT"]

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        wrist = hand.landmark[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        X = np.array(landmarks).reshape(1, -1)
        X_scaled = scaler.transform(X)
        
        pred = model.predict(X_scaled, verbose=0)
        confidence = np.max(pred)
        
        if confidence > 0.8: 
            gesture = GESTURE_NAMES[np.argmax(pred)]
            cv2.putText(frame, f"{gesture} ({confidence:.2f})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()