import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


def load_and_prepare_image(image_path):
    """
    Load an image for the trained ResNet50 FER-2013 model.

    ResNet50 preprocess_input is already included inside
    the saved model, so it is NOT applied here again.
    """

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE,
        color_mode="rgb"
    )

    image_array = tf.keras.utils.img_to_array(image)

    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array