import numpy as np

from src.preprocess import load_and_prepare_image


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]


def predict_emotion(model, image_path):

    image = load_and_prepare_image(image_path)

    predictions = model.predict(
        image,
        verbose=0
    )

    probabilities = predictions[0]

    predicted_index = np.argmax(probabilities)

    predicted_emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ]

    return predicted_emotion, confidence, probabilities