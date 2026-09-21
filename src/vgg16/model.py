"""
VGG16 model architecture for FER-2013 emotion classification.

The model uses ImageNet-pretrained VGG16 as the feature extractor
and a custom classification head for seven facial emotion classes.
"""

import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

from preprocessing import create_data_augmentation


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = (224, 224)
NUM_CLASSES = 7


# ============================================================
# Build VGG16 Model
# ============================================================

def build_vgg16_model():
    """
    Build the VGG16 transfer-learning model.

    Returns
    -------
    model:
        Complete FER-2013 classification model.

    base_model:
        ImageNet-pretrained VGG16 feature extractor.
    """

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    data_augmentation = create_data_augmentation()

    # --------------------------------------------------------
    # Pretrained VGG16
    # --------------------------------------------------------

    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Stage 1: freeze pretrained VGG16
    base_model.trainable = False

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(224, 224, 3),
        name="input_image"
    )

    # Training-only augmentation
    x = data_augmentation(inputs)

    # VGG16-specific preprocessing
    x = preprocess_input(x)

    # Feature extraction
    x = base_model(
        x,
        training=False
    )

    # --------------------------------------------------------
    # Classification Head
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling"
    )(x)

    x = layers.Dense(
        256,
        activation="relu",
        name="dense_256"
    )(x)

    x = layers.Dropout(
        0.5,
        name="dropout"
    )(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="emotion_predictions"
    )(x)

    # --------------------------------------------------------
    # Complete model
    # --------------------------------------------------------

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="FER2013_VGG16"
    )

    return model, base_model