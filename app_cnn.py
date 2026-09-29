from pathlib import Path
import sys
import tempfile

import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src" / "cnn"))

from face_detection import detect_and_crop_face
from predict import load_model, predict_emotion


st.set_page_config(
    page_title="FER-2013 Custom CNN",
    page_icon="🙂",
    layout="centered"
)


@st.cache_resource
def get_model():
    return load_model()


st.title("Facial Emotion Recognition")
st.caption("Custom CNN trained on FER-2013")

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

            model = get_model()

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