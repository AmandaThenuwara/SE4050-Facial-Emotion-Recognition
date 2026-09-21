"""
FER-2013 preprocessing utilities for the VGG16 model.

This module prepares the training, validation and test datasets
using the same configuration as the VGG16 experiment.
"""

import tensorflow as tf
from tensorflow.keras import layers


# ============================================================
# Configuration
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.20


# ============================================================
# Data Augmentation
# ============================================================

def create_data_augmentation():
    """
    Create the data augmentation pipeline used during training.

    Augmentation is applied only to training images.
    """

    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.05),
            layers.RandomZoom(0.10),
            layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05
            ),
        ],
        name="data_augmentation"
    )


# ============================================================
# Training Dataset
# ============================================================

def load_train_dataset(train_dir):
    """Load 80% of the FER-2013 training directory."""

    return tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        color_mode="rgb",
        shuffle=True
    )


# ============================================================
# Validation Dataset
# ============================================================

def load_validation_dataset(train_dir):
    """Load 20% of the FER-2013 training directory."""

    return tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        color_mode="rgb",
        shuffle=True
    )


# ============================================================
# Test Dataset
# ============================================================

def load_test_dataset(test_dir):
    """
    Load the original FER-2013 test set.

    Shuffle must remain False so predictions stay aligned
    with their true labels.
    """

    return tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )


# ============================================================
# Complete Dataset Loader
# ============================================================

def load_datasets(train_dir, test_dir):
    """Load and optimize train, validation and test datasets."""

    train_ds = load_train_dataset(train_dir)
    val_ds = load_validation_dataset(train_dir)
    test_ds = load_test_dataset(test_dir)

    # Save class names before applying prefetch.
    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds, class_names