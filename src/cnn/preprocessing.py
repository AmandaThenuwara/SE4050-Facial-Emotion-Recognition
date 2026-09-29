from PIL import Image
import numpy as np

IMAGE_SIZE = (48, 48)


def preprocess_image(image_path):
    """
    Load an image and prepare it for the Custom CNN.

    Important:
    The saved model already contains a Rescaling(1/255) layer,
    so this function does NOT divide pixel values by 255.
    """

    image = Image.open(image_path).convert("L")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    # Add channel dimension:
    # (48, 48) -> (48, 48, 1)
    image_array = np.expand_dims(image_array, axis=-1)

    # Add batch dimension:
    # (48, 48, 1) -> (1, 48, 48, 1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array