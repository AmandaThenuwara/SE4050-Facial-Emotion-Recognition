from pathlib import Path
from PIL import Image

from predict import predict_emotion


# Project root
ROOT_DIR = Path(__file__).resolve().parents[2]

# Folder containing test images
TEST_FOLDER = ROOT_DIR / "test_images"

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}


print("\n======================================")
print(" EfficientNetB0 Multiple Image Test")
print("======================================\n")


# Check if test folder exists
if not TEST_FOLDER.exists():
    print(f"Test folder not found: {TEST_FOLDER}")
    exit()


# Find all supported images
image_files = [
    file
    for file in TEST_FOLDER.iterdir()
    if file.suffix.lower() in SUPPORTED_FORMATS
]


if not image_files:
    print("No test images found.")

else:

    print(f"Found {len(image_files)} test image(s).\n")

    for image_path in image_files:

        try:
            image = Image.open(image_path)

            emotion, confidence, probabilities = predict_emotion(image)

            print(f"Image      : {image_path.name}")
            print(f"Prediction : {emotion.upper()}")
            print(f"Confidence : {confidence:.2f}%")
            print("--------------------------------------")

        except Exception as error:

            print(f"Could not process {image_path.name}")
            print(f"Error: {error}")
            print("--------------------------------------")