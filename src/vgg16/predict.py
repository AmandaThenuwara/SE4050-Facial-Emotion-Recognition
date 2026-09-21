"""
Single-image emotion prediction using the trained VGG16 FER-2013 model.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vgg16"
    / "IT23220560_VGG16_FER2013.keras"
)


# ============================================================
# Load Model
# ============================================================

def load_model():
    """Load the trained VGG16 FER-2013 model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("Loading VGG16 model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    return model


# ============================================================
# Preprocess Single Image
# ============================================================

def preprocess_image(image_path):
    """
    Load an image and prepare it for prediction.

    The saved model already contains VGG16 preprocess_input,
    so this function does NOT normalize or preprocess twice.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # Open image and force RGB format.
    image = Image.open(image_path).convert("RGB")

    # Resize to VGG16 input size.
    image = image.resize(IMG_SIZE)

    # Convert to NumPy array.
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# Predict Emotion
# ============================================================

def predict_emotion(model, image_path):
    """Predict the emotion of one facial image."""

    image_array = preprocess_image(
        image_path
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "emotion": predicted_emotion,
        "confidence": confidence,
        "probabilities": probabilities
    }


# ============================================================
# Display Prediction
# ============================================================

def print_prediction(result):

    print("\n" + "=" * 60)
    print("FACIAL EMOTION PREDICTION")
    print("=" * 60)

    print(
        f"\nPredicted Emotion : "
        f"{result['emotion'].upper()}"
    )

    print(
        f"Confidence        : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print("\nClass Probabilities")
    print("-" * 35)

    sorted_probabilities = sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True
    )

    for emotion, probability in sorted_probabilities:

        print(
            f"{emotion:<10} : "
            f"{probability * 100:6.2f}%"
        )

    print("=" * 60)