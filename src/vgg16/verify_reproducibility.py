"""
Verify reproducibility of the final VGG16 FER-2013 evaluation.

The script compares the metrics recorded during the original
experiment with metrics reproduced locally from the saved model.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "results" / "vgg16"

ORIGINAL_RESULTS = (
    RESULTS_DIR /
    "VGG16_Final_Test_Results.csv"
)

LOCAL_RESULTS = (
    RESULTS_DIR /
    "VGG16_Local_Evaluation_Results.csv"
)


# ============================================================
# Configuration
# ============================================================

TOLERANCE = 1e-4


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 65)
    print("VGG16 REPRODUCIBILITY VERIFICATION")
    print("=" * 65)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not ORIGINAL_RESULTS.exists():
        raise FileNotFoundError(
            f"Original results not found: {ORIGINAL_RESULTS}"
        )

    if not LOCAL_RESULTS.exists():
        raise FileNotFoundError(
            f"Local results not found: {LOCAL_RESULTS}"
        )

    # --------------------------------------------------------
    # Load results
    # --------------------------------------------------------

    original = pd.read_csv(
        ORIGINAL_RESULTS
    )

    local = pd.read_csv(
        LOCAL_RESULTS
    )

    print("\nOriginal results:")
    print(original)

    print("\nLocally reproduced results:")
    print(local)

    # --------------------------------------------------------
    # Metrics to compare
    # --------------------------------------------------------

    candidate_metrics = [
        "Test Accuracy",
        "Test Loss",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1"
    ]

    # Only compare metrics that exist in BOTH CSV files.
    metrics = [
        metric
        for metric in candidate_metrics
        if metric in original.columns
        and metric in local.columns
    ]

    if not metrics:
        raise ValueError(
            "No matching evaluation metrics were found "
            "between the two CSV files."
        )

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print("\nMetric Comparison")
    print("-" * 65)

    all_passed = True

    for metric in metrics:

        original_value = float(
            original.loc[0, metric]
        )

        local_value = float(
            local.loc[0, metric]
        )

        difference = abs(
            original_value - local_value
        )

        passed = difference <= TOLERANCE

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"

        print(
            f"{metric:<20} "
            f"Original={original_value:.6f}  "
            f"Local={local_value:.6f}  "
            f"Diff={difference:.8f}  "
            f"[{status}]"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    if all_passed:

        print(
            "REPRODUCIBILITY CHECK: PASSED"
        )

        print(
            "The locally evaluated saved model reproduces "
            "the original recorded results within tolerance."
        )

    else:

        print(
            "REPRODUCIBILITY CHECK: FAILED"
        )

        print(
            "One or more metrics differ beyond the "
            "configured tolerance."
        )

        raise SystemExit(1)

    print("=" * 65)


if __name__ == "__main__":
    main()