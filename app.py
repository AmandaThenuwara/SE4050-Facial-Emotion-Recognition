import os
import tempfile

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.face_detector import detect_and_crop_face
from src.predict import predict_emotion, CLASS_NAMES


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_PATH = "models/resnet50/resnet50_fer2013_final.keras"

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="📷",
    layout="wide"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

@st.cache_resource
def load_fer_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_fer_model()

except Exception as error:
    st.error(f"Unable to load ResNet50 model: {error}")
    st.stop()


# ============================================================
# 3. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This application demonstrates facial emotion recognition "
        "using a ResNet50 deep-learning model trained on FER-2013."
    )

    st.markdown("### Emotion Classes")

    for emotion in CLASS_NAMES:
        st.write(f"• {emotion.capitalize()}")

    st.divider()

    st.write("**Model:** ResNet50")
    st.write("**Dataset:** FER-2013")
    st.write("**Input Size:** 224 × 224 RGB")
    st.write("**Classes:** 7")


# ============================================================
# 4. HEADER
# ============================================================

st.title("Facial Emotion Recognition")

st.subheader(
    "FER-2013 Emotion Classification using ResNet50"
)

st.write(
    "Capture a facial image using your camera and let the "
    "trained ResNet50 model analyze the facial expression."
)


# ============================================================
# 5. CAMERA
# ============================================================

st.markdown("## Camera Capture")

st.info(
    "Position your face clearly in front of the camera "
    "and capture an image."
)

camera_image = st.camera_input(
    "Take a picture"
)


# ============================================================
# 6. PROCESS CAMERA IMAGE
# ============================================================

if camera_image is not None:

    image = Image.open(camera_image).convert("RGB")

    st.success("Image captured successfully.")

    left_column, right_column = st.columns(2)

    # --------------------------------------------------------
    # LEFT SIDE - ORIGINAL IMAGE
    # --------------------------------------------------------

    with left_column:

        st.markdown("### Captured Image")

        st.image(
            image,
            caption="Camera Capture",
            use_container_width=True
        )

    # --------------------------------------------------------
    # RIGHT SIDE - ANALYSIS
    # --------------------------------------------------------

    with right_column:

        st.markdown("### Emotion Prediction")

        analyze_button = st.button(
            "Analyze Emotion",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            input_path = None
            cropped_path = None

            try:

                # ============================================
                # SAVE CAMERA IMAGE TEMPORARILY
                # ============================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as temp_file:

                    image.save(
                        temp_file.name,
                        format="JPEG"
                    )

                    input_path = temp_file.name


                # ============================================
                # CREATE TEMPORARY CROPPED FACE FILE
                # ============================================

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".jpg"
                ) as cropped_file:

                    cropped_path = cropped_file.name


                # ============================================
                # FACE DETECTION
                # ============================================

                with st.spinner("Detecting face..."):

                    cropped_face_path, face_box = (
                        detect_and_crop_face(
                            input_path,
                            cropped_path
                        )
                    )


                # ============================================
                # EMOTION PREDICTION
                # ============================================

                with st.spinner("Analyzing emotion..."):

                    emotion, confidence, probabilities = (
                        predict_emotion(
                            model,
                            cropped_face_path
                        )
                    )


                # ============================================
                # DISPLAY RESULT
                # ============================================

                st.success("Analysis completed.")

                st.markdown("#### Detected Emotion")

                st.markdown(
                    f"# {emotion.upper()}"
                )

                st.metric(
                    label="Confidence",
                    value=f"{confidence * 100:.2f}%"
                )


                # ============================================
                # SHOW CROPPED FACE
                # ============================================

                st.markdown("#### Detected Face")

                cropped_image = Image.open(
                    cropped_face_path
                ).convert("RGB")

                st.image(
                    cropped_image,
                    caption="Face used for prediction",
                    width=250
                )


                # ============================================
                # PROBABILITY DATA
                # ============================================

                probability_data = pd.DataFrame(
                    {
                        "Emotion": [
                            name.capitalize()
                            for name in CLASS_NAMES
                        ],
                        "Probability": [
                            float(probability) * 100
                            for probability in probabilities
                        ]
                    }
                )

                probability_data = (
                    probability_data
                    .sort_values(
                        "Probability",
                        ascending=False
                    )
                    .reset_index(drop=True)
                )


                # ============================================
                # PROBABILITY CHART
                # ============================================

                st.markdown(
                    "#### Emotion Probabilities"
                )

                chart_data = (
                    probability_data
                    .set_index("Emotion")
                )

                st.bar_chart(
                    chart_data
                )


                # ============================================
                # PROBABILITY TABLE
                # ============================================

                display_table = (
                    probability_data.copy()
                )

                display_table["Probability"] = (
                    display_table["Probability"]
                    .map(lambda value: f"{value:.2f}%")
                )

                st.dataframe(
                    display_table,
                    use_container_width=True,
                    hide_index=True
                )


            # ================================================
            # NO FACE DETECTED
            # ================================================

            except ValueError as error:

                st.error(
                    f"Face detection failed: {error}"
                )

                st.info(
                    "Please capture another image with your "
                    "face clearly visible and facing the camera."
                )


            # ================================================
            # OTHER ERRORS
            # ================================================

            except Exception as error:

                st.error(
                    f"An error occurred during prediction: {error}"
                )


            # ================================================
            # DELETE TEMPORARY FILES
            # ================================================

            finally:

                if (
                    input_path is not None
                    and os.path.exists(input_path)
                ):
                    try:
                        os.remove(input_path)
                    except OSError:
                        pass

                if (
                    cropped_path is not None
                    and os.path.exists(cropped_path)
                ):
                    try:
                        os.remove(cropped_path)
                    except OSError:
                        pass


# ============================================================
# 7. WAITING STATE
# ============================================================

else:

    st.info(
        "Waiting for camera capture..."
    )


# ============================================================
# 8. HOW IT WORKS
# ============================================================

st.divider()

st.markdown("### How It Works")

st.write(
    """
    1. Capture an image using the camera.
    2. OpenCV detects the largest visible face.
    3. The detected face is cropped from the image.
    4. The face is resized to 224 × 224 RGB.
    5. The trained ResNet50 model performs emotion classification.
    6. The predicted emotion, confidence and class probabilities are displayed.
    """
)


# ============================================================
# 9. DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "This application was developed for educational and "
    "experimental purposes. Facial emotion predictions should "
    "not be interpreted as psychological or medical assessments."
)