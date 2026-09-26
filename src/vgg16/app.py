"""
Streamlit UI - VGG16 Facial Emotion Recognition

Features:
1. Single image emotion prediction
2. Multiple image emotion prediction
3. Live webcam emotion recognition
4. Automatic OpenCV face detection
5. Automatic face cropping with 15% padding
6. Multiple-face support
7. VGG16 emotion classification
8. Emotion probability visualization
9. Frame skipping for more efficient live webcam inference

Model:
VGG16 transfer-learning model trained on FER-2013.

Important:
The saved VGG16 model already contains VGG16 preprocess_input.
Therefore, preprocess_input must NOT be applied again here.
"""

from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image, ImageDraw

from streamlit_webrtc import (
    VideoProcessorBase,
    webrtc_streamer,
)

from face_detection import crop_faces


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
    "surprise",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vgg16"
    / "IT23220560_VGG16_FER2013.keras"
)

# Run VGG16 once every N webcam frames.
# Higher value = faster video but slower prediction updates.
WEBCAM_PREDICTION_INTERVAL = 3


# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🙂",
    layout="wide",
)


# ============================================================
# Header
# ============================================================

st.title("Facial Emotion Recognition")

st.write(
    "VGG16-based facial emotion classification using the "
    "FER-2013 dataset with automatic face detection and cropping."
)

st.caption(
    "Supported emotion classes: Angry, Disgust, Fear, Happy, "
    "Neutral, Sad and Surprise."
)


# ============================================================
# Load VGG16 Model
# ============================================================

@st.cache_resource
def load_vgg16_model():
    """
    Load and cache the trained VGG16 model.
    """

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


# ============================================================
# Prepare Face for VGG16
# ============================================================

def preprocess_face(face_image):
    """
    Prepare a PIL face image for VGG16.

    The saved model already contains VGG16 preprocess_input.
    Therefore this function only:
        - converts to RGB
        - resizes to 224 x 224
        - converts to float32
    """

    face_image = face_image.convert(
        "RGB"
    )

    face_image = face_image.resize(
        IMG_SIZE
    )

    face_array = np.asarray(
        face_image,
        dtype=np.float32,
    )

    return face_array


# ============================================================
# Predict Cropped PIL Faces
# ============================================================

def predict_faces(model, face_images):
    """
    Predict emotion for one or more cropped PIL faces.

    All faces are processed as a single batch.
    """

    if not face_images:
        return []

    processed_faces = [
        preprocess_face(face)
        for face in face_images
    ]

    batch = np.stack(
        processed_faces,
        axis=0,
    )

    predictions = model.predict(
        batch,
        verbose=0,
    )

    results = []

    for probabilities in predictions:

        predicted_index = int(
            np.argmax(probabilities)
        )

        emotion = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            probabilities[
                predicted_index
            ]
        )

        results.append(
            {
                "emotion": emotion,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )

    return results


# ============================================================
# Predict OpenCV Webcam Face
# ============================================================

def predict_webcam_face(
    model,
    face_bgr,
):
    """
    Predict emotion from an OpenCV BGR face crop.
    """

    # OpenCV BGR -> RGB
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB,
    )

    # Resize
    face_rgb = cv2.resize(
        face_rgb,
        IMG_SIZE,
    )

    # Convert to float32
    face_array = np.asarray(
        face_rgb,
        dtype=np.float32,
    )

    # Add batch dimension
    face_array = np.expand_dims(
        face_array,
        axis=0,
    )

    # Do NOT apply preprocess_input here.
    # It is already inside the saved model.

    probabilities = model.predict(
        face_array,
        verbose=0,
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    return (
        emotion,
        confidence,
        probabilities,
    )


# ============================================================
# Create Detection Preview
# ============================================================

def create_detection_preview(
    original_image,
    detected_faces,
):
    """
    Draw bounding boxes around detected faces.
    """

    preview = (
        original_image
        .convert("RGB")
        .copy()
    )

    drawer = ImageDraw.Draw(
        preview
    )

    for index, face_data in enumerate(
        detected_faces,
        start=1,
    ):

        x1, y1, x2, y2 = (
            face_data["box"]
        )

        drawer.rectangle(
            [x1, y1, x2, y2],
            outline="lime",
            width=4,
        )

        drawer.text(
            (
                x1 + 5,
                max(0, y1 - 20),
            ),
            f"Face {index}",
            fill="lime",
        )

    return preview


# ============================================================
# Probability DataFrame
# ============================================================

def create_probability_dataframe(
    result,
):
    """
    Convert seven-class probabilities into a DataFrame.
    """

    probability_df = pd.DataFrame(
        {
            "Emotion": [
                emotion.capitalize()
                for emotion
                in CLASS_NAMES
            ],

            "Probability": (
                result[
                    "probabilities"
                ] * 100
            ),
        }
    )

    probability_df = (
        probability_df
        .sort_values(
            "Probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    return probability_df


# ============================================================
# Display One Face Result
# ============================================================

def display_face_result(
    face_number,
    face_image,
    result,
):
    """
    Display cropped face and prediction information.
    """

    st.markdown(
        f"### Face {face_number}"
    )

    image_column, result_column = (
        st.columns(
            [1, 2]
        )
    )

    # --------------------------------------------------------
    # Cropped Face
    # --------------------------------------------------------

    with image_column:

        st.image(
            face_image,
            caption=(
                f"Detected Face "
                f"{face_number}"
            ),
            width=250,
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with result_column:

        metric_col1, metric_col2 = (
            st.columns(2)
        )

        with metric_col1:

            st.metric(
                "Predicted Emotion",
                result[
                    "emotion"
                ].upper(),
            )

        with metric_col2:

            st.metric(
                "Model Probability",
                (
                    f"{result['confidence'] * 100:.2f}%"
                ),
            )

        probability_df = (
            create_probability_dataframe(
                result
            )
        )

        st.markdown(
            "#### Emotion Probabilities"
        )

        st.bar_chart(
            probability_df.set_index(
                "Emotion"
            )
        )

        st.dataframe(
            probability_df.style.format(
                {
                    "Probability":
                        "{:.2f}%"
                }
            ),
            use_container_width=True,
        )


# ============================================================
# Process Uploaded Image
# ============================================================

def process_uploaded_image(
    model,
    image,
    image_name,
):
    """
    Complete still-image processing pipeline.

    Image
        ->
    Face Detection
        ->
    Face Cropping
        ->
    VGG16
        ->
    Emotion Prediction
    """

    st.subheader(
        image_name
    )

    # --------------------------------------------------------
    # Detect + Crop Faces
    # --------------------------------------------------------

    detected_faces = crop_faces(
        image,
        padding=0.15,
    )

    # --------------------------------------------------------
    # No Face
    # --------------------------------------------------------

    if len(detected_faces) == 0:

        st.warning(
            "No face was detected in this image. "
            "Try an image with a clearer frontal face."
        )

        st.image(
            image,
            caption="Input Image",
            width=400,
        )

        return []

    # --------------------------------------------------------
    # Face Count
    # --------------------------------------------------------

    face_count = len(
        detected_faces
    )

    st.success(
        f"{face_count} face(s) detected."
    )

    # --------------------------------------------------------
    # Detection Preview
    # --------------------------------------------------------

    detection_preview = (
        create_detection_preview(
            image,
            detected_faces,
        )
    )

    original_col, detection_col = (
        st.columns(2)
    )

    with original_col:

        st.markdown(
            "#### Original Image"
        )

        st.image(
            image,
            use_container_width=True,
        )

    with detection_col:

        st.markdown(
            "#### Detected Face Region(s)"
        )

        st.image(
            detection_preview,
            use_container_width=True,
        )

    # --------------------------------------------------------
    # Extract Faces
    # --------------------------------------------------------

    face_images = [
        face_data["image"]
        for face_data
        in detected_faces
    ]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    results = predict_faces(
        model,
        face_images,
    )

    st.markdown(
        "## Emotion Prediction"
    )

    summary_rows = []

    for index, (
        face_image,
        result,
    ) in enumerate(
        zip(
            face_images,
            results,
        ),
        start=1,
    ):

        display_face_result(
            index,
            face_image,
            result,
        )

        summary_rows.append(
            {
                "Image":
                    image_name,

                "Face":
                    index,

                "Predicted Emotion":
                    result[
                        "emotion"
                    ].capitalize(),

                "Probability":
                    result[
                        "confidence"
                    ] * 100,
            }
        )

        st.divider()

    return summary_rows


# ============================================================
# Load Model
# ============================================================

try:

    with st.spinner(
        "Loading VGG16 model..."
    ):

        model = load_vgg16_model()

except Exception as error:

    st.error(
        "Unable to load the VGG16 model."
    )

    st.exception(
        error
    )

    st.stop()


# ============================================================
# Live Webcam Processor
# ============================================================

class EmotionVideoProcessor(
    VideoProcessorBase
):
    """
    Processes live webcam frames.

    Face detection occurs on every frame.

    VGG16 prediction occurs every
    WEBCAM_PREDICTION_INTERVAL frames to reduce CPU load.
    """

    def __init__(self):

        # Reuse cached model
        self.model = model

        # Haar Cascade
        self.face_cascade = (
            cv2.CascadeClassifier(
                cv2.data.haarcascades
                + "haarcascade_frontalface_default.xml"
            )
        )

        if self.face_cascade.empty():

            raise RuntimeError(
                "Unable to load Haar Cascade."
            )

        self.frame_count = 0

        # Last predictions are stored between frames.
        self.last_predictions = []


    def recv(
        self,
        frame,
    ):

        # ----------------------------------------------------
        # WebRTC Frame -> OpenCV
        # ----------------------------------------------------

        image = frame.to_ndarray(
            format="bgr24"
        )

        frame_height, frame_width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # Face Detection
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        faces = (
            self.face_cascade
            .detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
        )

        self.frame_count += 1

        # ----------------------------------------------------
        # Run VGG16 Every N Frames
        # ----------------------------------------------------

        should_predict = (
            self.frame_count
            % WEBCAM_PREDICTION_INTERVAL
            == 0
        )

        current_predictions = []

        # ----------------------------------------------------
        # Process Faces
        # ----------------------------------------------------

        for face_index, (
            x,
            y,
            w,
            h,
        ) in enumerate(faces):

            # 15% padding
            pad_x = int(
                w * 0.15
            )

            pad_y = int(
                h * 0.15
            )

            x1 = max(
                0,
                int(x - pad_x),
            )

            y1 = max(
                0,
                int(y - pad_y),
            )

            x2 = min(
                frame_width,
                int(x + w + pad_x),
            )

            y2 = min(
                frame_height,
                int(y + h + pad_y),
            )

            # ------------------------------------------------
            # Face Crop
            # ------------------------------------------------

            face_crop = image[
                y1:y2,
                x1:x2
            ]

            if face_crop.size == 0:
                continue

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            emotion = "Detecting"
            confidence = 0.0

            if should_predict:

                (
                    emotion,
                    confidence,
                    _,
                ) = predict_webcam_face(
                    self.model,
                    face_crop,
                )

                current_predictions.append(
                    {
                        "emotion":
                            emotion,

                        "confidence":
                            confidence,
                    }
                )

            elif (
                face_index
                < len(
                    self.last_predictions
                )
            ):

                previous = (
                    self.last_predictions[
                        face_index
                    ]
                )

                emotion = previous[
                    "emotion"
                ]

                confidence = previous[
                    "confidence"
                ]

            # ------------------------------------------------
            # Bounding Box
            # ------------------------------------------------

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            # ------------------------------------------------
            # Label
            # ------------------------------------------------

            if emotion == "Detecting":

                label = (
                    "DETECTING..."
                )

            else:

                label = (
                    f"{emotion.upper()} "
                    f"{confidence * 100:.1f}%"
                )

            # ------------------------------------------------
            # Label Background
            # ------------------------------------------------

            (
                text_width,
                text_height,
            ), baseline = (
                cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    2,
                )
            )

            label_top = max(
                0,
                y1
                - text_height
                - baseline
                - 10,
            )

            cv2.rectangle(
                image,
                (
                    x1,
                    label_top,
                ),
                (
                    min(
                        frame_width,
                        x1
                        + text_width
                        + 10,
                    ),
                    y1,
                ),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                image,
                label,
                (
                    x1 + 5,
                    max(
                        text_height,
                        y1 - 7,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        # ----------------------------------------------------
        # Save Predictions
        # ----------------------------------------------------

        if (
            should_predict
            and current_predictions
        ):

            self.last_predictions = (
                current_predictions
            )

        # Clear stale prediction if no face exists
        if len(faces) == 0:

            self.last_predictions = []

        # ----------------------------------------------------
        # Application Label
        # ----------------------------------------------------

        cv2.putText(
            image,
            "VGG16 Live Emotion Recognition",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----------------------------------------------------
        # Return Frame
        # ----------------------------------------------------

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24",
        )


# ============================================================
# Sidebar
# ============================================================

st.sidebar.header(
    "Prediction Settings"
)

mode = st.sidebar.radio(
    "Select prediction mode",
    [
        "Single Image",
        "Multiple Images",
        "Live Webcam",
    ],
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Processing Pipeline**

    1. Input image/video
    2. Detect face(s)
    3. Crop facial region
    4. Resize to 224 × 224
    5. VGG16 prediction
    6. Display emotion
    """
)


# ============================================================
# Model Information
# ============================================================

with st.sidebar.expander(
    "Model Information"
):

    st.write(
        "**Architecture:** VGG16"
    )

    st.write(
        "**Pretrained weights:** ImageNet"
    )

    st.write(
        "**Input:** 224 × 224 × 3"
    )

    st.write(
        "**Dataset:** FER-2013"
    )

    st.write(
        "**Classes:** 7"
    )

    st.write(
        "**Test accuracy:** 61.70%"
    )


# ============================================================
# MODE 1 - Single Image
# ============================================================

if mode == "Single Image":

    st.header(
        "Single Image Prediction"
    )

    st.write(
        "Upload an image containing "
        "one or more faces."
    )

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=False,
        key="single_image",
    )

    if uploaded_file is not None:

        try:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            with st.spinner(
                "Detecting faces and "
                "predicting emotions..."
            ):

                process_uploaded_image(
                    model,
                    image,
                    uploaded_file.name,
                )

        except Exception as error:

            st.error(
                "Unable to process image."
            )

            st.exception(
                error
            )


# ============================================================
# MODE 2 - Multiple Images
# ============================================================

elif mode == "Multiple Images":

    st.header(
        "Multiple Image Prediction"
    )

    st.write(
        "Upload multiple images. "
        "Each image may contain "
        "one or more faces."
    )

    uploaded_files = st.file_uploader(
        "Choose multiple images",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
        accept_multiple_files=True,
        key="multiple_images",
    )

    if uploaded_files:

        all_results = []

        total_images = len(
            uploaded_files
        )

        st.info(
            f"{total_images} image(s) uploaded."
        )

        # ----------------------------------------------------
        # Process Images
        # ----------------------------------------------------

        for image_number, uploaded_file in enumerate(
            uploaded_files,
            start=1,
        ):

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### Image "
                        f"{image_number} of "
                        f"{total_images}"
                    )

                    with st.spinner(
                        f"Processing "
                        f"{uploaded_file.name}..."
                    ):

                        image_results = (
                            process_uploaded_image(
                                model,
                                image,
                                uploaded_file.name,
                            )
                        )

                    all_results.extend(
                        image_results
                    )

            except Exception as error:

                st.error(
                    f"Unable to process "
                    f"{uploaded_file.name}."
                )

                st.exception(
                    error
                )

        # ----------------------------------------------------
        # Overall Summary
        # ----------------------------------------------------

        if all_results:

            st.header(
                "Overall Prediction Summary"
            )

            summary_df = pd.DataFrame(
                all_results
            )

            st.dataframe(
                summary_df.style.format(
                    {
                        "Probability":
                            "{:.2f}%"
                    }
                ),
                use_container_width=True,
            )

            emotion_counts = (
                summary_df[
                    "Predicted Emotion"
                ]
                .value_counts()
                .rename_axis(
                    "Emotion"
                )
                .reset_index(
                    name="Count"
                )
            )

            st.subheader(
                "Predicted Emotion Distribution"
            )

            st.bar_chart(
                emotion_counts.set_index(
                    "Emotion"
                )
            )


# ============================================================
# MODE 3 - LIVE WEBCAM
# ============================================================

elif mode == "Live Webcam":

    st.header(
        "Live Webcam Emotion Recognition"
    )

    st.write(
        "The webcam continuously detects faces "
        "and predicts facial-expression categories "
        "using the trained VGG16 model."
    )

    st.info(
        "For better results, keep your face clearly visible, "
        "look toward the camera and use sufficient lighting."
    )

    # --------------------------------------------------------
    # Architecture Information
    # --------------------------------------------------------

    st.markdown(
        """
        **Live processing flow**

        `Webcam → Face Detection → Face Crop → VGG16 → Emotion`
        """
    )

    # --------------------------------------------------------
    # WebRTC Live Stream
    # --------------------------------------------------------

    webrtc_streamer(
        key="vgg16-live-emotion",

        video_processor_factory=(
            EmotionVideoProcessor
        ),

        media_stream_constraints={
            "video": True,
            "audio": False,
        },

        async_processing=True,
    )

    st.caption(
        "VGG16 prediction is performed periodically rather "
        "than on every video frame to reduce CPU processing load."
    )


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "VGG16 transfer-learning prototype trained on FER-2013. "
    "Predictions represent facial-expression classifications "
    "produced by the model and should not be interpreted as "
    "measurements of a person's actual emotional state."
)