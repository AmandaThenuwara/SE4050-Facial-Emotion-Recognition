import tensorflow as tf
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parents[2]

# Trained EfficientNetB0 model
MODEL_PATH = (
    ROOT_DIR
    / "models"
    / "efficientnetb0"
    / "IT23193772_EfficientNetB0_FER2013.keras"
)

print("Model path:")
print(MODEL_PATH)

print("\nLoading EfficientNetB0 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("\nModel loaded successfully!")

print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)