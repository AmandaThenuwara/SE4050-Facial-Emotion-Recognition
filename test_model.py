import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from src.predict import predict_emotion, CLASS_NAMES
from src.face_detector import detect_and_crop_face


# ============================================================
# 1. FILE PATHS
# ============================================================

MODEL_PATH = "models/resnet50/resnet50_fer2013_final.keras"
ORIGINAL_IMAGE_PATH = "test_images/test_face.jpg"


# ============================================================
# 2. LOAD TRAINED RESNET50 MODEL
# ============================================================

print("=" * 55)
print("FER-2013 ResNet50 Facial Emotion Recognition")
print("=" * 55)

print("\nLoading ResNet50 model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")


# ============================================================
# 3. DETECT AND CROP FACE
# ============================================================

print("\nDetecting face...")

try:
    cropped_path, face_box = detect_and_crop_face(
        ORIGINAL_IMAGE_PATH
    )

    print("Face detected successfully!")
    print(f"Face location: {face_box}")

except Exception as error:
    print("\nFace detection failed.")
    print(f"Error: {error}")
    exit()


# ============================================================
# 4. PREDICT EMOTION
# ============================================================

print("\nAnalyzing facial emotion...")

try:
    emotion, confidence, probabilities = predict_emotion(
        model,
        cropped_path
    )

except Exception as error:
    print("\nEmotion prediction failed.")
    print(f"Error: {error}")
    exit()


# ============================================================
# 5. DISPLAY RESULT IN TERMINAL
# ============================================================

print("\n" + "=" * 55)
print("PREDICTION RESULT")
print("=" * 55)

print(f"\nPredicted Emotion : {emotion.upper()}")
print(f"Confidence        : {confidence * 100:.2f}%")

print("\nClass Probabilities")
print("-" * 40)

for class_name, probability in zip(
    CLASS_NAMES,
    probabilities
):
    print(
        f"{class_name.capitalize():10s} : "
        f"{probability * 100:.2f}%"
    )

print("=" * 55)


# ============================================================
# 6. DISPLAY CROPPED FACE
# ============================================================

cropped_image = Image.open(cropped_path)

plt.figure(figsize=(6, 6))

plt.imshow(cropped_image)

plt.title(
    f"ResNet50 Prediction: {emotion.upper()}\n"
    f"Confidence: {confidence * 100:.2f}%"
)

plt.axis("off")
plt.tight_layout()

plt.show()


# ============================================================
# 7. DISPLAY EMOTION PROBABILITY GRAPH
# ============================================================

probability_percentages = [
    probability * 100
    for probability in probabilities
]

plt.figure(figsize=(9, 5))

plt.bar(
    CLASS_NAMES,
    probability_percentages
)

plt.title(
    "ResNet50 Facial Emotion Prediction Probabilities"
)

plt.xlabel("Emotion")
plt.ylabel("Probability (%)")

plt.xticks(rotation=45)

# Probability scale: 0% - 100%
plt.ylim(0, 100)

plt.tight_layout()

plt.show()


# ============================================================
# 8. FINISHED
# ============================================================

print("\nPrediction completed successfully!")