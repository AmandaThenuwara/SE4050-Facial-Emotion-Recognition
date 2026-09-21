import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

from preprocessing import preprocess_image


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "efficientnetb0"
    / "IT23193772_EfficientNetB0_FER2013.keras"
)


# --------------------------------------------------
# Emotion classes
# Must match the training dataset class order
# --------------------------------------------------

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

print("Loading EfficientNetB0 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# --------------------------------------------------
# Prediction function
# --------------------------------------------------

def predict_emotion(image):
    """
    Predict facial emotion from a PIL image.
    """

    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image, verbose=0)

    # Get probabilities for first image
    probabilities = predictions[0]

    # Find class with highest probability
    predicted_index = np.argmax(probabilities)

    predicted_emotion = CLASS_NAMES[predicted_index]

    confidence = probabilities[predicted_index] * 100

    return predicted_emotion, confidence, probabilities


# --------------------------------------------------
# Test using an image path
# --------------------------------------------------

if __name__ == "__main__":

    image_path = input("Enter image path: ").strip().strip('"')

    image = Image.open(image_path)

    emotion, confidence, probabilities = predict_emotion(image)

    print("\n------------------------------")
    print("Prediction Result")
    print("------------------------------")

    print(f"Predicted Emotion: {emotion.upper()}")
    print(f"Confidence: {confidence:.2f}%")

    print("\nAll Emotion Probabilities:")

    for class_name, probability in zip(CLASS_NAMES, probabilities):
        print(f"{class_name.capitalize():10s}: {probability * 100:.2f}%")