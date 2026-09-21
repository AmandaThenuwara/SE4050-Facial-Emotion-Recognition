"""
Test the VGG16 architecture without training the model.
"""

from model import build_vgg16_model


print("=" * 60)
print("VGG16 ARCHITECTURE TEST")
print("=" * 60)


# ------------------------------------------------------------
# Build model
# ------------------------------------------------------------

model, base_model = build_vgg16_model()


# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

print("\nModel name:")
print(model.name)

print("\nInput shape:")
print(model.input_shape)

print("\nOutput shape:")
print(model.output_shape)

print("\nNumber of output classes:")
print(model.output_shape[-1])

print("\nVGG16 base model trainable:")
print(base_model.trainable)


# ------------------------------------------------------------
# Parameter counts
# ------------------------------------------------------------

total_params = model.count_params()

trainable_params = sum(
    int(tf_var.shape.num_elements())
    for tf_var in model.trainable_weights
)

non_trainable_params = total_params - trainable_params


print("\nTotal parameters:")
print(f"{total_params:,}")

print("\nTrainable parameters:")
print(f"{trainable_params:,}")

print("\nNon-trainable parameters:")
print(f"{non_trainable_params:,}")


# ------------------------------------------------------------
# Architecture
# ------------------------------------------------------------

print("\nModel Summary:\n")

model.summary()


# ------------------------------------------------------------
# Validation checks
# ------------------------------------------------------------

assert model.input_shape == (None, 224, 224, 3)
assert model.output_shape == (None, 7)
assert base_model.trainable is False


print("\n" + "=" * 60)
print("VGG16 ARCHITECTURE TEST COMPLETED SUCCESSFULLY")
print("=" * 60)