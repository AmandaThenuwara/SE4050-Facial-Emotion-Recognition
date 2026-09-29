from pathlib import Path
import sys
import tempfile

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src" / "cnn"))

from face_detection import detect_and_crop_face
from predict import load_model, predict_emotion


st.set_page_config(
    page_title="FER-2013 Custom CNN",
    page_icon="🙂",
    layout="centered"
)


CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


@st.cache_resource
def get_model():
    return load_model()


model = get_model()


st.title("Facial Emotion Recognition")
st.caption("Custom CNN trained on FER-2013")


mode = st.radio(
    "Choose Mode",
    ["Upload Image", "Live Emotions"],
    horizontal=True
)


if mode == "Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a face image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.subheader("Uploaded Image")
        st.image(image, width=350)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)

            input_path = temp_dir / "input_image.png"
            cropped_path = temp_dir / "cropped_face.jpg"

            image.convert("RGB").save(input_path)

            try:
                detected_face_path, face_box = detect_and_crop_face(
                    input_path,
                    cropped_path
                )

                st.subheader("Detected Face")
                st.image(
                    str(detected_face_path),
                    width=250
                )

                result = predict_emotion(
                    model,
                    detected_face_path
                )

                st.subheader("Prediction")

                st.success(
                    f"Predicted Emotion: "
                    f"{result['emotion'].upper()}"
                )

                st.write(
                    f"Confidence: "
                    f"{result['confidence'] * 100:.2f}%"
                )

                st.subheader("Class Probabilities")

                sorted_probabilities = sorted(
                    result["probabilities"].items(),
                    key=lambda item: item[1],
                    reverse=True
                )

                for emotion, probability in sorted_probabilities:
                    st.write(
                        f"{emotion.capitalize()}: "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(
                        min(
                            int(probability * 100),
                            100
                        )
                    )

            except Exception as error:
                st.error(str(error))


if mode == "Live Emotions":

    st.subheader("Live Webcam Emotion Detection")

    st.write(
        "Turn on the camera and keep your face clearly visible."
    )

    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )


    class EmotionVideoProcessor(VideoProcessorBase):

        def recv(self, frame):

            image = frame.to_ndarray(
                format="bgr24"
            )

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60)
            )

            for x, y, w, h in faces:

                face = gray[
                    y:y + h,
                    x:x + w
                ]

                face = cv2.resize(
                    face,
                    (48, 48)
                )

                face = face.astype(
                    np.float32
                )

                face = np.expand_dims(
                    face,
                    axis=-1
                )

                face = np.expand_dims(
                    face,
                    axis=0
                )

                predictions = model.predict(
                    face,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                emotion = CLASS_NAMES[
                    predicted_index
                ]

                confidence = float(
                    predictions[predicted_index]
                )

                label = (
                    f"{emotion.upper()} "
                    f"{confidence * 100:.1f}%"
                )

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    image,
                    label,
                    (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )


    webrtc_streamer(
        key="cnn-live-emotion",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )