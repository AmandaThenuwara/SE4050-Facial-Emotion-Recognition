import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# ==================================================
# Project paths
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src" / "efficientnetb0"

sys.path.append(str(SRC_DIR))

from predict import predict_emotion, CLASS_NAMES


# ==================================================
# Page configuration
# ==================================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊",
    layout="wide"
)


# ==================================================
# Header
# ==================================================

st.title("Facial Emotion Recognition")

st.write(
    "Upload a facial image to classify its expression using "
    "the trained EfficientNetB0 deep learning model."
)

st.info(
    "Supported emotions: Angry, Disgust, Fear, Happy, "
    "Neutral, Sad and Surprise."
)

st.divider()


# ==================================================
# Image upload
# ==================================================

uploaded_file = st.file_uploader(
    "Upload a facial image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)

        image_column, result_column = st.columns([1, 1])

        # ==========================================
        # Uploaded image
        # ==========================================

        with image_column:

            st.subheader("Uploaded Image")

            st.image(
                image,
                caption="Input facial image",
                width=350
            )

        # ==========================================
        # Prediction
        # ==========================================

        with result_column:

            st.subheader("Prediction")

            if st.button(
                "Predict Emotion",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("Analyzing facial expression..."):

                    emotion, confidence, probabilities = predict_emotion(
                        image
                    )

                st.success(
                    f"Predicted Emotion: {emotion.upper()}"
                )

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                # Confidence warning
                if confidence < 50:

                    st.warning(
                        "Low-confidence prediction. "
                        "The model may be uncertain between "
                        "multiple facial expressions."
                    )

                elif confidence < 70:

                    st.info(
                        "Moderate-confidence prediction. "
                        "Interpret the result with some caution."
                    )

                else:

                    st.success(
                        "High-confidence model prediction."
                    )

                # ==================================
                # Probability results
                # ==================================

                st.subheader("Emotion Probabilities")

                probability_data = pd.DataFrame({
                    "Emotion": [
                        name.capitalize()
                        for name in CLASS_NAMES
                    ],
                    "Probability (%)": [
                        float(value * 100)
                        for value in probabilities
                    ]
                })

                probability_data = probability_data.sort_values(
                    "Probability (%)",
                    ascending=False
                )

                st.bar_chart(
                    probability_data,
                    x="Emotion",
                    y="Probability (%)"
                )

                # ==================================
                # Detailed probability table
                # ==================================

                st.subheader("Detailed Results")

                st.dataframe(
                    probability_data,
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as error:

        st.error(
            f"Unable to process the uploaded image: {error}"
        )


# ==================================================
# Model information
# ==================================================

st.divider()

with st.expander("Model Information"):

    st.write("**Architecture:** EfficientNetB0")
    st.write("**Task:** Facial Emotion Recognition")
    st.write("**Dataset:** FER-2013")
    st.write("**Input Size:** 224 × 224 × 3")
    st.write("**Number of Classes:** 7")
    st.write("**Final Test Accuracy:** 58.15%")

    st.caption(
        "Predictions are model estimates and may be incorrect, "
        "especially for visually similar or ambiguous expressions."
    )