"""
Test single-image emotion prediction.
"""

from pathlib import Path
import sys

from predict import (
    load_model,
    predict_emotion,
    print_prediction
)


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("VGG16 SINGLE IMAGE PREDICTION TEST")
    print("=" * 60)

    # Require image path from command line.
    if len(sys.argv) < 2:

        print(
            "\nUsage:"
            "\npython src/vgg16/test_prediction.py "
            "\"path/to/image.jpg\""
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    print("\nImage:")
    print(image_path)

    # Load model.
    model = load_model()

    # Predict.
    result = predict_emotion(
        model,
        image_path
    )

    # Display result.
    print_prediction(
        result
    )


if __name__ == "__main__":
    main()