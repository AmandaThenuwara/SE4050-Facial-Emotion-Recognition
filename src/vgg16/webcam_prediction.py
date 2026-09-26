"""
Real-time webcam facial emotion recognition using VGG16.

Pipeline:
Webcam -> Face Detection -> Face Crop -> VGG16 -> Emotion Prediction

Press Q to close the webcam.
"""

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path


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


# ============================================================
# Load VGG16 Model
# ============================================================

print("=" * 60)
print("VGG16 REAL-TIME EMOTION RECOGNITION")
print("=" * 60)

print("\nLoading VGG16 model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully.")


# ============================================================
# Load Face Detector
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError(
        "Unable to load Haar Cascade face detector."
    )

print("Face detector loaded successfully.")


# ============================================================
# Prediction Function
# ============================================================

def predict_emotion(face_bgr):
    """
    Predict emotion from a cropped OpenCV BGR face.
    """

    # OpenCV uses BGR, but training input uses RGB
    face_rgb = cv2.cvtColor(
        face_bgr,
        cv2.COLOR_BGR2RGB
    )

    face_rgb = cv2.resize(
        face_rgb,
        IMG_SIZE
    )

    face_array = np.asarray(
        face_rgb,
        dtype=np.float32
    )

    face_array = np.expand_dims(
        face_array,
        axis=0
    )

    # Do NOT call preprocess_input here.
    # It is already contained in the saved model.
    probabilities = model.predict(
        face_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    emotion = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return emotion, confidence


# ============================================================
# Open Webcam
# ============================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError(
        "Unable to open webcam."
    )

print("\nWebcam started.")
print("Press Q to quit.\n")


# ============================================================
# Webcam Loop
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read webcam frame.")
        break

    # --------------------------------------------------------
    # Convert to grayscale for face detection
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Detect Faces
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # --------------------------------------------------------
    # Process Each Face
    # --------------------------------------------------------

    for x, y, w, h in faces:

        # Add 15% padding around detected face
        pad_x = int(w * 0.15)
        pad_y = int(h * 0.15)

        frame_height, frame_width = (
            frame.shape[:2]
        )

        x1 = max(
            0,
            x - pad_x
        )

        y1 = max(
            0,
            y - pad_y
        )

        x2 = min(
            frame_width,
            x + w + pad_x
        )

        y2 = min(
            frame_height,
            y + h + pad_y
        )

        # Crop face
        face_crop = frame[
            y1:y2,
            x1:x2
        ]

        if face_crop.size == 0:
            continue

        # Predict emotion
        emotion, confidence = (
            predict_emotion(
                face_crop
            )
        )

        # ----------------------------------------------------
        # Bounding Box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # Prediction Label
        # ----------------------------------------------------

        label = (
            f"{emotion.upper()} "
            f"{confidence * 100:.1f}%"
        )

        label_y = max(
            y1 - 10,
            25
        )

        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "VGG16 Facial Emotion Recognition",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Display Webcam
    # --------------------------------------------------------

    cv2.imshow(
        "VGG16 Real-Time Emotion Recognition",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Cleanup
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\nWebcam closed.")