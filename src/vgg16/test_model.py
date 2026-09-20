import tensorflow as tf
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from preprocessing import load_test_dataset


MODEL_PATH = "models/vgg16/IT23220560_VGG16_FER2013.keras"
TEST_DIR = "data/fer2013/test"


# Load dataset
test_ds = load_test_dataset(TEST_DIR)

class_names = test_ds.class_names


# Load trained VGG16 model
model = tf.keras.models.load_model(MODEL_PATH)

print("VGG16 model loaded successfully.")


# Evaluate
test_loss, test_accuracy = model.evaluate(
    test_ds,
    verbose=1
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# Predictions
predictions = model.predict(test_ds)

y_pred = np.argmax(predictions, axis=1)

y_true = np.concatenate([
    np.argmax(labels.numpy(), axis=1)
    for images, labels in test_ds
])


# Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# Confusion matrix
print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        y_true,
        y_pred
    )
)