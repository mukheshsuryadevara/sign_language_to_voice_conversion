import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Function to add salt and pepper noise to images
def salt_and_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = image.copy()
    total_pixels = image.size
    salt = int(total_pixels * salt_prob)
    pepper = int(total_pixels * pepper_prob)

    # Add salt noise
    salt_coords = [np.random.randint(0, i-1, salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255

    # Add pepper noise
    pepper_coords = [np.random.randint(0, i-1, pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# Function to preprocess images: resize, add noise, and flatten
def preprocess_images(input_dir):
    data = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(os.listdir(input_dir))}
    
    for label, idx in label_map.items():
        gesture_dir = os.path.join(input_dir, label)
        if os.path.isdir(gesture_dir):
            for img_name in os.listdir(gesture_dir):
                img_path = os.path.join(gesture_dir, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        raise ValueError(f"Could not read image: {img_path}")
                    
                    # Resize image
                    img = cv2.resize(img, (64, 64))
                    
                    # Apply salt-and-pepper noise
                    noisy_img = salt_and_pepper_noise(img, salt_prob=0.02, pepper_prob=0.02)
                    
                    # Add the noisy image to the data
                    data.append(noisy_img.flatten())
                    labels.append(idx)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    
    return np.array(data), np.array(labels), label_map


if __name__ == "__main__":
    # Load preprocessed data
    X, y, label_map = preprocess_images("dataset\American")

    # Check if data is loaded properly
    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")

    # If dataset is empty, raise an error
    if X.size == 0:
        print("Error: The dataset is empty. Check the images in your dataset directory.")
    else:
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a KNN classifier
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

        # Save the model and label map
        joblib.dump(model, "gesture_model.pkl")
        joblib.dump(label_map, "label_map.pkl")
