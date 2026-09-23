import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/resnet50/resnet50_fer2013_final.keras"

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

IMG_SIZE = (224, 224)


# ============================================================
# LOAD RESNET50 MODEL
# ============================================================

print("=" * 55)
print("FER-2013 ResNet50 - Live Emotion Recognition")
print("=" * 55)

print("\nLoading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load Haar Cascade face detector.")

print("Face detector loaded successfully!")


# ============================================================
# START WEBCAM
# ============================================================

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

print("\nWebcam started successfully!")
print("Press Q to quit.")


# ============================================================
# LIVE LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read webcam frame.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    # ========================================================
    # PROCESS EACH DETECTED FACE
    # ========================================================

    for (x, y, w, h) in faces:

        # Crop face
        face = frame[
            y:y + h,
            x:x + w
        ]

        # Convert BGR -> RGB
        face_rgb = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Resize to model input size
        face_rgb = cv2.resize(
            face_rgb,
            IMG_SIZE
        )

        # Convert to float array
        face_array = np.array(
            face_rgb,
            dtype=np.float32
        )

        # Add batch dimension
        face_array = np.expand_dims(
            face_array,
            axis=0
        )

        # IMPORTANT:
        # Do NOT apply preprocess_input here.
        # Your saved model already contains
        # ResNet50 preprocessing.

        # Predict
        predictions = model.predict(
            face_array,
            verbose=0
        )[0]

        predicted_index = np.argmax(
            predictions
        )

        emotion = CLASS_NAMES[
            predicted_index
        ]

        confidence = predictions[
            predicted_index
        ] * 100

        # ====================================================
        # DRAW RESULT
        # ====================================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        label = (
            f"{emotion.upper()} "
            f"{confidence:.1f}%"
        )

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


    # ========================================================
    # DISPLAY WEBCAM
    # ========================================================

    cv2.imshow(
        "ResNet50 Facial Emotion Recognition",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()
cv2.destroyAllWindows()

print("\nWebcam stopped.")