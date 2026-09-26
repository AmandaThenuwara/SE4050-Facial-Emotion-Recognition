"""
Test automatic face detection and cropping.

Usage:
python src/vgg16/test_face_detection.py "path/to/image.jpg"
"""

import sys
from pathlib import Path

from PIL import Image

from face_detection import (
    crop_faces,
    draw_face_boxes
)


def main():

    if len(sys.argv) < 2:

        print(
            "Usage: python "
            "src/vgg16/test_face_detection.py "
            "\"path/to/image.jpg\""
        )

        return

    image_path = Path(
        sys.argv[1]
    )

    if not image_path.exists():

        print(
            f"Image not found: {image_path}"
        )

        return

    image = Image.open(
        image_path
    ).convert("RGB")

    cropped_faces = crop_faces(
        image
    )

    print("\n" + "=" * 60)
    print("FACE DETECTION TEST")
    print("=" * 60)

    print(
        f"\nImage: {image_path.name}"
    )

    print(
        f"Detected faces: {len(cropped_faces)}"
    )

    if len(cropped_faces) == 0:

        print(
            "\nNo face was detected."
        )

        return

    output_dir = Path(
        "outputs/face_detection_test"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save image with bounding boxes
    boxed_image = draw_face_boxes(
        image
    )

    boxed_path = (
        output_dir
        / f"{image_path.stem}_detected.jpg"
    )

    boxed_image.save(
        boxed_path
    )

    print(
        f"\nBounding-box image saved:\n{boxed_path}"
    )

    # Save individual face crops
    for index, face_data in enumerate(
        cropped_faces,
        start=1
    ):

        face = face_data["image"]

        face_path = (
            output_dir
            / f"{image_path.stem}_face_{index}.jpg"
        )

        face.save(
            face_path
        )

        print(
            f"\nFace {index}"
        )

        print(
            f"Bounding box: "
            f"{face_data['box']}"
        )

        print(
            f"Saved: {face_path}"
        )

    print(
        "\nFace detection test completed."
    )


if __name__ == "__main__":
    main()