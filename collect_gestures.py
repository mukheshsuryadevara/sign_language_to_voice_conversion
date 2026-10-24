import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def capture_gesture(output_dir, gesture_name):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    print(f"Capturing gesture: {gesture_name}. Press 'q' to quit.")

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Show frame using Matplotlib instead of cv2.imshow()
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis('off')  # Hide axis
        plt.show()

        # Save frames when spacebar is pressed
        if cv2.waitKey(1) & 0xFF == ord(' '):
            img_path = os.path.join(output_dir, f"{gesture_name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"Saved: {img_path}")
            count += 1

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_name = input("Enter gesture name: ").strip()
    output_dir = "dataset/" + gesture_name
    capture_gesture(output_dir, gesture_name)
