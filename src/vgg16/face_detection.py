"""
Face detection and cropping utilities for the
VGG16 Facial Emotion Recognition prototype.
"""

import cv2
import numpy as np
from PIL import Image


# ============================================================
# Haar Cascade Face Detector
# ============================================================

FACE_CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(
    FACE_CASCADE_PATH
)

if face_cascade.empty():
    raise RuntimeError(
        "Unable to load OpenCV Haar Cascade face detector."
    )


# ============================================================
# Detect Faces
# ============================================================

def detect_faces(image):

    rgb_image = np.array(
        image.convert("RGB")
    )

    gray_image = cv2.cvtColor(
        rgb_image,
        cv2.COLOR_RGB2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    return rgb_image, faces


# ============================================================
# Crop Faces
# ============================================================

def crop_faces(
    image,
    padding=0.15,
):

    rgb_image, faces = (
        detect_faces(image)
    )

    image_height, image_width = (
        rgb_image.shape[:2]
    )

    cropped_faces = []

    for x, y, w, h in faces:

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        pad_x = int(
            w * padding
        )

        pad_y = int(
            h * padding
        )

        # ----------------------------------------------------
        # Safe Bounding Box
        # ----------------------------------------------------

        x1 = max(
            0,
            int(x - pad_x),
        )

        y1 = max(
            0,
            int(y - pad_y),
        )

        x2 = min(
            image_width,
            int(x + w + pad_x),
        )

        y2 = min(
            image_height,
            int(y + h + pad_y),
        )

        # ----------------------------------------------------
        # Crop Face
        # ----------------------------------------------------

        face_crop = rgb_image[
            y1:y2,
            x1:x2
        ]

        if face_crop.size == 0:
            continue

        face_pil = Image.fromarray(
            face_crop
        )

        cropped_faces.append(
            {
                "image": face_pil,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2,
                ),
            }
        )

    return cropped_faces


# ============================================================
# Draw Face Bounding Boxes
# ============================================================

def draw_face_boxes(image):

    rgb_image, faces = (
        detect_faces(image)
    )

    output_image = (
        rgb_image.copy()
    )

    for x, y, w, h in faces:

        cv2.rectangle(
            output_image,
            (
                int(x),
                int(y),
            ),
            (
                int(x + w),
                int(y + h),
            ),
            (0, 255, 0),
            2,
        )

    return Image.fromarray(
        output_image
    )