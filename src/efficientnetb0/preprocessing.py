import numpy as np
from PIL import Image


# Input size expected by EfficientNetB0
IMG_SIZE = (224, 224)


def preprocess_image(image):
    """
    Preprocess an image before sending it to the trained
    EfficientNetB0 facial emotion recognition model.

    Parameters:
        image (PIL.Image.Image): Input facial image.

    Returns:
        numpy.ndarray: Image array with shape (1, 224, 224, 3).
    """

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image to match model input size
    image = image.resize(IMG_SIZE)

    # Convert image to NumPy array
    image_array = np.array(image)

    # Add batch dimension
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array