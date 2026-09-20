import numpy as np
from PIL import Image

IMG_SIZE = (224, 224)

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array