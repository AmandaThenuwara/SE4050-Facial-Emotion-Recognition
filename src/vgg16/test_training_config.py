"""
Verify the VGG16 transfer-learning and fine-tuning configuration
without performing model training.
"""

import tensorflow as tf

from model import build_vgg16_model
from train import (
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE
)


print("=" * 60)
print("VGG16 TRAINING CONFIGURATION TEST")
print("=" * 60)


# ============================================================
# Build model
# ============================================================

model, base_model = build_vgg16_model()


# ============================================================
# Stage 1
# ============================================================

print("\nSTAGE 1 - TRANSFER LEARNING")

print("Initial epochs:", INITIAL_EPOCHS)
print("Learning rate:", INITIAL_LEARNING_RATE)
print("VGG16 frozen:", not base_model.trainable)


stage1_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

print("Trainable parameters:")
print(f"{stage1_params:,}")


assert base_model.trainable is False
assert stage1_params == 133127


# ============================================================
# Simulate Stage 2 configuration
# ============================================================

base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False


print("\nSTAGE 2 - FINE-TUNING")

print("Fine-tuning epochs:", FINE_TUNE_EPOCHS)
print("Learning rate:", FINE_TUNE_LEARNING_RATE)


trainable_vgg_layers = [
    layer.name
    for layer in base_model.layers
    if layer.trainable
]


print("\nTrainable VGG16 layers:")

for layer_name in trainable_vgg_layers:
    print(" -", layer_name)


stage2_params = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)


print("\nTrainable parameters after unfreezing:")
print(f"{stage2_params:,}")


print("\n" + "=" * 60)
print("TRAINING CONFIGURATION TEST COMPLETED SUCCESSFULLY")
print("=" * 60)