import sys
from pathlib import Path

import streamlit as st
from PIL import Image


# ==================================================
# Project Paths
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT_DIR / "src" / "efficientnetb0"

# Allow Streamlit app to import EfficientNetB0 modules
sys.path.append(str(SRC_DIR))

from predict import predict_emotion, CLASS_NAMES


# ==================================================
# Page Configuration
# ==================================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="centered"
)


# ==================================================
# Application Header
# ==================================================

st.title("Facial Emotion Recognition")

st.write(
    "Upload a facial image and the trained EfficientNetB0 "
    "model will predict the facial emotion."
)

st.info(
    "Supported emotions: Angry, Disgust, Fear, Happy, "
    "Neutral, Sad and Surprise."
)


# ==================================================
# Image Upload
# ==================================================

uploaded_file = st.file_uploader(
    "Upload a facial image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# Prediction
# ==================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Uploaded facial image",
        width=300
    )

    if st.button("Predict Emotion"):

        with st.spinner("Analyzing facial expression..."):

            emotion, confidence, probabilities = predict_emotion(image)

        # Main prediction
        st.success(
            f"Predicted Emotion: {emotion.upper()}"
        )

        # Confidence
        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        # All probabilities
        st.subheader("Emotion Probabilities")

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            percentage = float(probability * 100)

            st.write(
                f"**{class_name.capitalize()}** — "
                f"{percentage:.2f}%"
            )

            st.progress(
                min(max(percentage / 100, 0.0), 1.0)
            )