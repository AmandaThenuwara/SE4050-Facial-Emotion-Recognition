from pathlib import Path

import cv2


def detect_and_crop_face(
    image_path,
    output_path=None,
    padding_ratio=0.15
):
    """
    Detect the largest face in an image and crop it.

    Returns:
        cropped_path, face_box
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(
        cascade_path
    )

    faces = face_detector.detectMultiScale(
        grayscale,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        raise ValueError(
            "No face was detected in the image."
        )

    # Use the largest detected face.
    x, y, width, height = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    padding_x = int(width * padding_ratio)
    padding_y = int(height * padding_ratio)

    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)

    x2 = min(
        image.shape[1],
        x + width + padding_x
    )

    y2 = min(
        image.shape[0],
        y + height + padding_y
    )

    cropped_face = image[
        y1:y2,
        x1:x2
    ]

    if output_path is None:
        output_path = (
            image_path.parent
            / f"{image_path.stem}_cropped.jpg"
        )

    output_path = Path(output_path)

    cv2.imwrite(
        str(output_path),
        cropped_face
    )

    face_box = (
        int(x1),
        int(y1),
        int(x2 - x1),
        int(y2 - y1)
    )

    return output_path, face_box