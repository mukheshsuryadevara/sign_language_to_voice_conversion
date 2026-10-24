# Sign Language to Voice Conversion

## About the Project

Sign Language to Voice Conversion is a Python-based machine learning project that recognizes hand gestures through a webcam and converts the predicted gestures into spoken output. It aims to support communication between people who use sign language and people who may not understand it.

## Features

- **Gesture Data Collection:** Capture hand gesture images using a webcam.
- **Image Preprocessing:** Convert images to grayscale, resize them to 64 × 64 pixels, and apply salt-and-pepper noise.
- **Machine Learning:** Train a K-Nearest Neighbors (KNN) classifier to recognize gestures.
- **Real-Time Recognition:** Predict gestures using a live webcam feed.
- **Text-to-Speech Conversion:** Convert recognized gesture labels into speech.
- **Model Saving:** Save the trained model and gesture label mapping using Joblib.

## Technologies Used

- Python
- OpenCV
- NumPy
- Scikit-learn
- Joblib
- pyttsx3

## Installation

1. Clone this repository:

```bash
git clone https://github.com/mukheshsuryadevara/sign_language_to_voice_conversion.git
```

2. Navigate to the project directory:

```bash
cd sign_language_to_voice_conversion
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### Step 1: Collect Gesture Images

Run the data collection script:

```bash
python collect_gestures.py
```

Enter the gesture name when prompted. Press **Space** to save an image and **Q** to quit. Repeat for each gesture class.

### Step 2: Train the Model

Ensure your gesture images are organized into class-specific folders under `dataset/American/`, then run:

```bash
python train_model.py
```

The script trains the KNN classifier, evaluates its accuracy, and saves the following files:

- `gesture_model.pkl`
- `label_map.pkl`

### Step 3: Recognize Gestures and Generate Speech

After training, run:

```bash
python gesture_recognition.py
```

The webcam displays the predicted gesture, and the text-to-speech engine speaks the gesture label. Press **Q** to exit.

## How It Works

1. Capture images of different hand gestures.
2. Preprocess the images into grayscale 64 × 64 pixel arrays.
3. Train a KNN classifier using the collected gesture dataset.
4. Capture live video frames from the webcam.
5. Predict the gesture using the trained model.
6. Convert the predicted gesture label into speech using `pyttsx3`.

## Requirements

- Python 3.9 or later recommended
- A working webcam
- A microphone is not required
- Speakers or headphones for audio output
- A labeled dataset containing gesture images

## Limitations

- Recognition accuracy depends on the quality and variety of training images.
- Changes in lighting, background, hand position, and camera angle may affect predictions.
- The current implementation recognizes trained static gestures and does not interpret complete sign-language sentences or continuous motion.

## Future Enhancements

- Use CNN-based deep learning models for improved recognition.
- Add more gesture classes and vocabulary.
- Improve recognition under different lighting conditions.
- Add confidence thresholds and prediction smoothing.
- Develop a graphical user interface for easier interaction.

## License

No license has been specified for this project yet.
