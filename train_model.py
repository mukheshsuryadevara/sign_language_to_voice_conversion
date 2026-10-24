import cv2
import numpy as np
import pyttsx3
import joblib

# Load the trained model and label map
model = joblib.load("gesture_model.pkl")
label_map = joblib.load("label_map.pkl")
reverse_label_map = {idx: label for label, idx in label_map.items()}

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Set speech rate (optional)
engine.setProperty('volume', 1)  # Set volume level (optional)

def preprocess_frame(frame):
    """Preprocess the frame for gesture recognition."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))  # Resize to match model input
    return resized.flatten()

# Attempt to open webcam with DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow instead of MSMF

if not cap.isOpened():
    print("Error: Could not open webcam. Please check your webcam connection.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image. Please check your webcam.")
        break

    # Flip the frame for selfie view
    frame = cv2.flip(frame, 1)

    try:
        # Preprocess frame and classify
        input_data = preprocess_frame(frame)
        prediction = model.predict([input_data])[0]  # Predict gesture label
        gesture = reverse_label_map.get(prediction, "Unknown")  # Retrieve gesture label

        # Display the result on the frame
        cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Speak the result
        engine.say(gesture)
        engine.runAndWait()

    except Exception as e:
        print(f"Error during processing: {e}")

    # Show the frame with gesture text
    cv2.imshow("Gesture Recognition", frame)

    # Quit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
