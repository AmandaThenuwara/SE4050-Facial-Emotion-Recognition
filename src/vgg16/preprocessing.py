import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def load_test_dataset(test_dir):
    """
    Load FER-2013 test dataset for VGG16 evaluation.
    """

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        color_mode="rgb"
    )

    return test_ds