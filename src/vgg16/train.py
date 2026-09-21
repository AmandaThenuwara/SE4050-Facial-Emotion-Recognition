"""
Training pipeline for VGG16 on FER-2013.

Stage 1:
    Train the custom classification head while the
    ImageNet-pretrained VGG16 base remains frozen.

Stage 2:
    Fine-tune the final four layers of VGG16 using
    a smaller learning rate.
"""

from pathlib import Path

import tensorflow as tf

from preprocessing import load_datasets
from model import build_vgg16_model


# ============================================================
# Configuration
# ============================================================

SEED = 42

INITIAL_EPOCHS = 20
FINE_TUNE_EPOCHS = 10

INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "data" / "fer2013" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "fer2013" / "test"

MODEL_DIR = PROJECT_ROOT / "models" / "vgg16"

MODEL_PATH = (
    MODEL_DIR /
    "IT23220560_VGG16_FER2013.keras"
)


# ============================================================
# Reproducibility
# ============================================================

tf.random.set_seed(SEED)


# ============================================================
# Callbacks
# ============================================================

def create_callbacks():
    """Create callbacks used during training."""

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7
        )
    ]


# ============================================================
# Stage 1
# ============================================================

def train_classifier_head(
    model,
    train_ds,
    val_ds
):
    """
    Stage 1:
    Train the classification head while VGG16 is frozen.
    """

    print("\n" + "=" * 60)
    print("STAGE 1 - TRANSFER LEARNING")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=INITIAL_EPOCHS,
        callbacks=create_callbacks()
    )

    return history


# ============================================================
# Stage 2
# ============================================================

def fine_tune_model(
    model,
    base_model,
    train_ds,
    val_ds
):
    """
    Stage 2:
    Unfreeze VGG16 and train only its final four layers.
    """

    print("\n" + "=" * 60)
    print("STAGE 2 - FINE-TUNING")
    print("=" * 60)

    # Allow VGG16 layers to become trainable.
    base_model.trainable = True

    # Freeze every layer except the final four.
    for layer in base_model.layers[:-4]:
        layer.trainable = False

    # Recompile after changing trainable layers.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=FINE_TUNE_LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=create_callbacks()
    )

    return history


# ============================================================
# Main Training Pipeline
# ============================================================

def main():

    print("=" * 60)
    print("FER-2013 VGG16 TRAINING PIPELINE")
    print("=" * 60)

    # Check dataset.
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    # Create model directory if necessary.
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load datasets.
    train_ds, val_ds, _, class_names = load_datasets(
        TRAIN_DIR,
        TEST_DIR
    )

    print("\nClasses:")
    print(class_names)

    # Build VGG16.
    model, base_model = build_vgg16_model()

    print("\nInitial trainable parameters:")

    initial_trainable = sum(
        tf.keras.backend.count_params(weight)
        for weight in model.trainable_weights
    )

    print(f"{initial_trainable:,}")

    # --------------------------------------------------------
    # Stage 1
    # --------------------------------------------------------

    train_classifier_head(
        model,
        train_ds,
        val_ds
    )

    # --------------------------------------------------------
    # Stage 2
    # --------------------------------------------------------

    fine_tune_model(
        model,
        base_model,
        train_ds,
        val_ds
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    model.save(MODEL_PATH)

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()