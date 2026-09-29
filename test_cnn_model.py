from pathlib import Path
import sys

# Allow imports from src/cnn
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src" / "cnn"))

from face_detection import detect_and_crop_face
from predict import load_model, predict_emotion


TEST_IMAGE = PROJECT_ROOT / "test_images" / "cnn" / "sample_face.png"

CROPPED_IMAGE = (
    PROJECT_ROOT
    / "test_images"
    / "cnn"
    / "sample_face_cropped.jpg"
)


def main():
    print("Detecting face...")

    cropped_path, face_box = detect_and_crop_face(
        TEST_IMAGE,
        CROPPED_IMAGE
    )

    print(f"Face detected: {face_box}")
    print(f"Cropped image saved to: {cropped_path}")

    model = load_model()

    result = predict_emotion(
        model,
        cropped_path
    )

    print("\n" + "=" * 50)
    print("CUSTOM CNN EMOTION PREDICTION")
    print("=" * 50)

    print(f"\nPredicted Emotion : {result['emotion'].upper()}")
    print(f"Confidence        : {result['confidence'] * 100:.2f}%")

    print("\nClass Probabilities")
    print("-" * 35)

    for emotion, probability in sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(
            f"{emotion.capitalize():10s} : "
            f"{probability * 100:6.2f}%"
        )

    print("=" * 50)


if __name__ == "__main__":
    main()