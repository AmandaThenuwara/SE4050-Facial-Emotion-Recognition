import cv2


def detect_and_crop_face(image_path, output_path="test_images/cropped_face.jpg"):

    # Load original image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        raise ValueError("No face detected in the image.")

    # Use the largest detected face
    x, y, w, h = max(
        faces,
        key=lambda face: face[2] * face[3]
    )

    # Crop face
    cropped_face = image[
        y:y + h,
        x:x + w
    ]

    # Save cropped face
    cv2.imwrite(
        output_path,
        cropped_face
    )

    return output_path, (x, y, w, h)