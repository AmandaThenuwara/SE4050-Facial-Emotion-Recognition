from pathlib import Path

from preprocessing import load_datasets


# Project root:
# SE4050-Facial-Emotion-Recognition/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = PROJECT_ROOT / "data" / "fer2013" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "fer2013" / "test"


print("=" * 60)
print("FER-2013 PREPROCESSING TEST")
print("=" * 60)

print("\nTraining directory:")
print(TRAIN_DIR)

print("\nTest directory:")
print(TEST_DIR)


# ------------------------------------------------------------
# Check paths
# ------------------------------------------------------------

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training directory not found: {TRAIN_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

train_ds, val_ds, test_ds, class_names = load_datasets(
    TRAIN_DIR,
    TEST_DIR
)


# ------------------------------------------------------------
# Display information
# ------------------------------------------------------------

print("\nClasses:")
print(class_names)

print("\nNumber of classes:")
print(len(class_names))


# ------------------------------------------------------------
# Inspect one training batch
# ------------------------------------------------------------

for images, labels in train_ds.take(1):

    print("\nImage batch shape:")
    print(images.shape)

    print("\nLabel batch shape:")
    print(labels.shape)

    print("\nImage data type:")
    print(images.dtype)

    print("\nLabel data type:")
    print(labels.dtype)


print("\n" + "=" * 60)
print("PREPROCESSING TEST COMPLETED SUCCESSFULLY")
print("=" * 60)