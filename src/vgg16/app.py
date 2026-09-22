"""
Streamlit prototype for VGG16 Facial Emotion Recognition.

Features:
- Single image prediction
- Multiple image prediction
- Prediction confidence
- Probability distribution
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vgg16"
    / "IT23220560_VGG16_FER2013.keras"
)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🙂",
    layout="wide"
)


# ============================================================
# Load Model
# ============================================================

@st.cache_resource
def load_vgg16_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH
    )


# ============================================================
# Image Preprocessing
# ============================================================

def preprocess_pil_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    return image_array


# ============================================================
# Single Prediction
# ============================================================

def predict_single(model, image):

    image_array = preprocess_pil_image(
        image
    )

    batch = np.expand_dims(
        image_array,
        axis=0
    )

    predictions = model.predict(
        batch,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    return {
        "emotion": CLASS_NAMES[predicted_index],
        "confidence": float(
            predictions[predicted_index]
        ),
        "probabilities": predictions
    }


# ============================================================
# Batch Prediction
# ============================================================

def predict_multiple(model, images):

    processed_images = [
        preprocess_pil_image(image)
        for image in images
    ]

    batch = np.stack(
        processed_images,
        axis=0
    )

    predictions = model.predict(
        batch,
        verbose=0
    )

    results = []

    for prediction in predictions:

        predicted_index = int(
            np.argmax(prediction)
        )

        results.append(
            {
                "emotion":
                    CLASS_NAMES[predicted_index],

                "confidence":
                    float(
                        prediction[predicted_index]
                    ),

                "probabilities":
                    prediction
            }
        )

    return results


# ============================================================
# Header
# ============================================================

st.title("Facial Emotion Recognition")

st.write(
    "VGG16-based facial emotion classification "
    "using the FER-2013 dataset."
)

st.caption(
    "Supported emotions: Angry, Disgust, Fear, Happy, "
    "Neutral, Sad and Surprise."
)


# ============================================================
# Load Model
# ============================================================

try:

    model = load_vgg16_model()

except Exception as error:

    st.error(
        f"Unable to load VGG16 model: {error}"
    )

    st.stop()


# ============================================================
# Prediction Mode
# ============================================================

st.sidebar.header("Prediction Settings")

mode = st.sidebar.radio(
    "Select prediction mode",
    [
        "Single Image",
        "Multiple Images"
    ]
)


# ============================================================
# SINGLE IMAGE MODE
# ============================================================

if mode == "Single Image":

    st.header("Single Image Prediction")

    uploaded_file = st.file_uploader(
        "Upload a facial image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=False
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        )

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                width=350
            )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        with col2:

            st.subheader("Prediction")

            with st.spinner(
                "Analysing image..."
            ):

                result = predict_single(
                    model,
                    image
                )

            st.metric(
                "Predicted Emotion",
                result["emotion"].upper()
            )

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        st.subheader(
            "Emotion Probabilities"
        )

        probability_df = pd.DataFrame(
            {
                "Emotion": CLASS_NAMES,
                "Probability": (
                    result["probabilities"] * 100
                )
            }
        )

        probability_df = (
            probability_df
            .sort_values(
                "Probability",
                ascending=False
            )
            .reset_index(drop=True)
        )

        st.bar_chart(
            probability_df.set_index(
                "Emotion"
            )
        )

        st.dataframe(
            probability_df.style.format(
                {
                    "Probability": "{:.2f}%"
                }
            ),
            use_container_width=True
        )


# ============================================================
# MULTIPLE IMAGE MODE
# ============================================================

else:

    st.header(
        "Multiple Image Prediction"
    )

    uploaded_files = st.file_uploader(
        "Upload multiple facial images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        images = [
            Image.open(file).convert("RGB")
            for file in uploaded_files
        ]

        with st.spinner(
            f"Analysing {len(images)} images..."
        ):

            results = predict_multiple(
                model,
                images
            )

        st.success(
            f"{len(images)} images analysed successfully."
        )

        # ----------------------------------------------------
        # Individual results
        # ----------------------------------------------------

        st.subheader(
            "Prediction Results"
        )

        table_results = []

        for uploaded_file, result in zip(
            uploaded_files,
            results
        ):

            table_results.append(
                {
                    "Image":
                        uploaded_file.name,

                    "Predicted Emotion":
                        result["emotion"].capitalize(),

                    "Confidence":
                        result["confidence"] * 100
                }
            )

        results_df = pd.DataFrame(
            table_results
        )

        st.dataframe(
            results_df.style.format(
                {
                    "Confidence": "{:.2f}%"
                }
            ),
            use_container_width=True
        )

        # ----------------------------------------------------
        # Image cards
        # ----------------------------------------------------

        st.subheader(
            "Individual Predictions"
        )

        for i, (
            uploaded_file,
            image,
            result
        ) in enumerate(
            zip(
                uploaded_files,
                images,
                results
            )
        ):

            with st.expander(
                f"{uploaded_file.name} — "
                f"{result['emotion'].upper()} "
                f"({result['confidence'] * 100:.2f}%)"
            ):

                col1, col2 = st.columns(
                    [1, 2]
                )

                with col1:

                    st.image(
                        image,
                        width=250
                    )

                with col2:

                    probabilities_df = pd.DataFrame(
                        {
                            "Emotion":
                                CLASS_NAMES,

                            "Probability":
                                result[
                                    "probabilities"
                                ] * 100
                        }
                    )

                    probabilities_df = (
                        probabilities_df
                        .sort_values(
                            "Probability",
                            ascending=False
                        )
                    )

                    st.bar_chart(
                        probabilities_df.set_index(
                            "Emotion"
                        )
                    )

        # ----------------------------------------------------
        # Batch summary
        # ----------------------------------------------------

        st.subheader(
            "Batch Summary"
        )

        emotion_counts = (
            results_df[
                "Predicted Emotion"
            ]
            .value_counts()
            .rename_axis("Emotion")
            .reset_index(name="Count")
        )

        st.bar_chart(
            emotion_counts.set_index(
                "Emotion"
            )
        )

        st.dataframe(
            emotion_counts,
            use_container_width=True
        )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "VGG16 transfer-learning model trained on FER-2013. "
    "Predictions represent model classifications and may "
    "not always reflect a person's actual emotional state."
)