from pathlib import Path

import numpy as np
import tensorflow as tf

from preprocessing import preprocess_image


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "cnn"
    / "FER2013_Custom_CNN.keras"
)


def load_model():
    """Load the trained Custom CNN model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading Custom CNN model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    print("Model loaded successfully.")

    return model


def predict_emotion(model, image_path):
    """Predict emotion from a single image."""

    image_array = preprocess_image(image_path)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(predictions))
    predicted_emotion = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": probabilities,
    }