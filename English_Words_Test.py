import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load the model and label classes
saved_data = joblib.load(r"C:\Users\moham\Downloads\English_Words\svm_model_with_labels.pkl")
svm_model = saved_data['model']
label_classes = saved_data['label_classes']

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture
cap = cv2.VideoCapture(0)

# Get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define codec and create VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
out = cv2.VideoWriter("output.mp4", fourcc, 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirrored effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract (x, y, z) coordinates (using z now)
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

            # Preprocessing: Center on wrist and normalize by middle finger tip
            wrist_x, wrist_y, wrist_z = landmarks[0]
            landmarks[:, 0] -= wrist_x
            landmarks[:, 1] -= wrist_y
            landmarks[:, 2] -= wrist_z

            # Use middle finger tip (index 12) for normalization
            mid_tip_x, mid_tip_y, mid_tip_z = landmarks[12]
            scale = np.sqrt(mid_tip_x**2 + mid_tip_y**2 + mid_tip_z**2)
            if scale == 0:  # Avoid divide-by-zero
                continue
            landmarks[:, 0] /= scale
            landmarks[:, 1] /= scale
            landmarks[:, 2] /= scale

            # Flatten to 1D feature vector (now includes z coordinates)
            features = landmarks.flatten().reshape(1, -1)

            # Predict using the loaded model
            prediction = svm_model.predict(features)[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display prediction
            cv2.putText(frame, f'Prediction: {prediction}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Write and show the frame
    out.write(frame)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
