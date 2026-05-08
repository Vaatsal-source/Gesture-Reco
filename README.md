This repository contains a high-performance, real-time hand gesture recognition system powered by Computer Vision and Deep Learning. Using MediaPipe for robust skeletal tracking and a custom TensorFlow neural network, this module translates physical hand movements into digital commands with high precision.
________________________________________
🛠️ Key Technical Components
1. MediaPipe Hand Tracking
The system leverages the MediaPipe Hand Landmarker to identify 21 distinct 3D hand joints. By utilizing these landmarks rather than raw pixel data, the system remains resilient to changes in lighting, background clutter, and skin tone.
2. Intelligent Feature Engineering
To ensure the model works regardless of the hand's position on the screen, the script performs Wrist-Relative Normalization:
•	Every landmark $(x, y, z)$ is recalculated as an offset from the wrist coordinate.
•	This creates Translation Invariance, meaning a "FIST" is recognized whether it is in the top-left or bottom-right of the camera frame.
3. Neural Network Classifier
•	Input Layer: 63 features ($21 \text{ landmarks} \times 3 \text{ dimensions}$).
•	Preprocessing: Includes a StandardScaler (scaler.pkl) to normalize input distributions for stable inference.
•	Inference Logic: Implements a strict 80% confidence threshold to eliminate "jitter" and false positives during rapid transitions.
________________________________________
🚀 Supported Gestures
The model is currently trained to classify the following five states:
•	FIST
•	OPEN
•	UP
•	LEFT
•	RIGHT
________________________________________
📂 Project Structure
Plaintext
├── gesture_recognition.py   # Main inference & CV2 loop
├── gesture_model.h5         # Pre-trained Keras Neural Network
├── scaler.pkl               # Pickled scikit-learn scaler for normalization
└── requirements.txt         # Project dependencies
________________________________________
⚙️ Installation & Usage
1. Clone & Install Dependencies
Ensure you have Python 3.8+ installed, then run:
Bash
pip install opencv-python mediapipe tensorflow numpy scikit-learn
2. Running the System
Simply execute the main script to launch the webcam interface:
Bash
python gesture_recognition.py
3. Controls
•	Webcam Feed: The script automatically mirrors the feed for intuitive interaction.
•	HUD: Real-time gesture labels and confidence scores are overlaid on the top-left corner.
•	Exit: Press 'q' on your keyboard to release the camera and close the window.
________________________________________
🧪 Architecture Overview
1.	Capture: OpenCV grabs frames from the hardware camera.
2.	Process: MediaPipe extracts the skeletal "mesh" of the hand.
3.	Normalize: Coordinates are centered around the wrist and scaled.
4.	Predict: TensorFlow/Keras model outputs a probability distribution.
5.	Act: If the top probability exceeds 0.8, the gesture is identified.
________________________________________
📝 Future Roadmap
•	Dashboard Integration: Porting predicted gestures via WebSockets to the MyoFlex Bionic Dashboard.
•	Dynamic Gestures: Implementing LSTM (Long Short-Term Memory) layers to recognize temporal gestures like "swiping" or "waving."
•	Low-Power Optimization: Converting the .h5 model to TensorFlow Lite for edge device deployment.
________________________________________
⚖️ License
This project is licensed under the MIT License. Developed for research in Human-Computer Interaction (HCI).
