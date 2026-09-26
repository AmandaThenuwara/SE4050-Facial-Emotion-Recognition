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

# Allow Streamlit to import the EfficientNetB0 modules
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import predict_emotion, CLASS_NAMES
from realtime import show_live_camera


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
    "Predict facial emotions using the trained EfficientNetB0 "
    "deep learning model. You can upload a facial image or use "
    "the live webcam for real-time emotion recognition."
)

st.info(
    "Supported emotions: Angry, Disgust, Fear, Happy, "
    "Neutral, Sad and Surprise."
)

st.divider()


# ==================================================
# Input method selection
# ==================================================

st.subheader("Select Input Method")

input_method = st.radio(
    "Choose how you want to provide the facial input:",
    ["Upload Image", "Live Webcam"],
    horizontal=True
)


# ==================================================
# Upload Image Mode
# ==================================================

if input_method == "Upload Image":

    st.subheader("Upload Facial Image")

    uploaded_file = st.file_uploader(
        "Upload a facial image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:

            image = Image.open(uploaded_file)

            st.divider()

            image_column, result_column = st.columns([1, 1])

            # ==========================================
            # Uploaded image
            # ==========================================

            with image_column:

                st.subheader("Uploaded Image")

                st.image(
                    image,
                    caption="Facial image used for prediction",
                    width=350
                )

            # ==========================================
            # Prediction
            # ==========================================

            with result_column:

                st.subheader("Prediction")

                st.write(
                    "Click the button below to analyze "
                    "the facial expression."
                )

                predict_button = st.button(
                    "Predict Emotion",
                    type="primary",
                    use_container_width=True
                )

                if predict_button:

                    with st.spinner(
                        "Analyzing facial expression..."
                    ):

                        emotion, confidence, probabilities = (
                            predict_emotion(image)
                        )

                    # ==================================
                    # Main prediction
                    # ==================================

                    st.success(
                        f"Predicted Emotion: "
                        f"{emotion.upper()}"
                    )

                    st.metric(
                        label="Prediction Confidence",
                        value=f"{confidence:.2f}%"
                    )

                    # ==================================
                    # Confidence interpretation
                    # ==================================

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

                    probability_data = (
                        probability_data.sort_values(
                            "Probability (%)",
                            ascending=False
                        )
                    )

                    st.bar_chart(
                        probability_data,
                        x="Emotion",
                        y="Probability (%)"
                    )

                    # ==================================
                    # Detailed results
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
# Live Webcam Mode
# ==================================================

elif input_method == "Live Webcam":

    st.subheader("Live Emotion Recognition")

    st.write(
        "Start the webcam and position your face clearly in "
        "front of the camera. The system will automatically "
        "detect your face and continuously predict your emotion."
    )

    st.info(
        "For better predictions, keep your face clearly visible, "
        "look toward the camera, and use adequate lighting."
    )

    # ==============================================
    # Live camera
    # ==============================================

    try:

        show_live_camera()

    except Exception as error:

        st.error(
            f"Unable to start live emotion recognition: {error}"
        )


# ==================================================
# Instructions
# ==================================================

st.divider()

with st.expander("How to Use the Application"):

    st.markdown(
        """
        ### Upload Image Mode

        1. Select **Upload Image**.
        2. Upload a JPG, JPEG or PNG facial image.
        3. Click **Predict Emotion**.
        4. View the predicted emotion.
        5. View the prediction confidence and probability chart.

        ### Live Webcam Mode

        1. Select **Live Webcam**.
        2. Click **START** to start the webcam.
        3. Allow camera access when requested by your browser.
        4. Position your face clearly in front of the camera.
        5. The system automatically detects your face.
        6. The detected face is sent to the EfficientNetB0 model.
        7. The predicted emotion and confidence are displayed
           directly on the live video.
        8. Click **STOP** when you want to stop the webcam.
        """
    )


# ==================================================
# Model information
# ==================================================

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


# ==================================================
# Application pipeline information
# ==================================================

with st.expander("Live Prediction Pipeline"):

    st.markdown(
        """
        **Live Webcam**

        ↓

        **Video Frame**

        ↓

        **OpenCV Face Detection**

        ↓

        **Face Cropping**

        ↓

        **Image Preprocessing (224 × 224 × 3)**

        ↓

        **EfficientNetB0**

        ↓

        **Softmax Classification**

        ↓

        **Emotion + Confidence**
        """
    )


# ==================================================
# Footer
# ==================================================

st.divider()

st.caption(
    "FER-2013 Facial Emotion Recognition using EfficientNetB0"
)