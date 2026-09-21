"""
Evaluation utilities for the trained VGG16 FER-2013 model.

Calculates:
- Test loss
- Test accuracy
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from preprocessing import load_test_dataset


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_ROOT / "data" / "fer2013" / "test"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vgg16"
    / "IT23220560_VGG16_FER2013.keras"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "vgg16"

RESULTS_PATH = (
    RESULTS_DIR
    / "VGG16_Local_Evaluation_Results.csv"
)


# ============================================================
# Load Model
# ============================================================

def load_trained_model():
    """Load the previously trained VGG16 model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("\nLoading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    return model


# ============================================================
# Get Predictions
# ============================================================

def get_predictions(model, test_ds):
    """Generate true and predicted class labels."""

    print("\nGenerating predictions...")

    probabilities = model.predict(
        test_ds,
        verbose=1
    )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    y_true = np.concatenate([
        np.argmax(labels.numpy(), axis=1)
        for _, labels in test_ds
    ])

    return y_true, y_pred, probabilities


# ============================================================
# Calculate Metrics
# ============================================================

def calculate_metrics(y_true, y_pred):

    return {
        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Macro Precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Macro Recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Macro F1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        ),

        "Weighted Precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Weighted Recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        ),

        "Weighted F1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )
    }


# ============================================================
# Save Results
# ============================================================

def save_metrics(metrics, test_loss):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    row = {
        "Model": "VGG16",
        "Test Loss": test_loss,
        **metrics
    }

    dataframe = pd.DataFrame([row])

    dataframe.to_csv(
        RESULTS_PATH,
        index=False
    )

    print("\nEvaluation results saved to:")
    print(RESULTS_PATH)


# ============================================================
# Main Evaluation
# ============================================================

def main():

    print("=" * 60)
    print("VGG16 FER-2013 FINAL EVALUATION")
    print("=" * 60)

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DIR}"
        )

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_ds = load_test_dataset(
        TEST_DIR
    )

    class_names = test_ds.class_names

    print("\nEmotion classes:")
    print(class_names)

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = load_trained_model()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print("\nEvaluating model...")

    test_loss, test_accuracy = model.evaluate(
        test_ds,
        verbose=1
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_true, y_pred, _ = get_predictions(
        model,
        test_ds
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    print("\nFINAL METRICS")
    print("-" * 40)

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nCLASSIFICATION REPORT")
    print("-" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\nCONFUSION MATRIX")
    print("-" * 60)
    print(cm)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics,
        test_loss
    )

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()