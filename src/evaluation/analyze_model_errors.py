
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    ROOT / "outputs" / "reports"
    / "mlp_classification_report.csv"
)

CONFUSION_PATH = (
    ROOT / "outputs" / "reports"
    / "mlp_confusion_matrix.csv"
)


def main():

    if not REPORT_PATH.exists():
        raise FileNotFoundError(REPORT_PATH)

    if not CONFUSION_PATH.exists():
        raise FileNotFoundError(CONFUSION_PATH)

    report = pd.read_csv(REPORT_PATH)
    confusion = pd.read_csv(
        CONFUSION_PATH,
        index_col=0
    )

    print("\n" + "=" * 65)
    print("SIGNSPEAK AI - MODEL ERROR ANALYSIS")
    print("=" * 65)

    # Lowest-performing classes.
    print("\n1. TEN MOST DIFFICULT CLASSES (BY F1-SCORE)")
    print("-" * 65)

    difficult = report.sort_values(
        "f1_score",
        ascending=True
    ).head(10)

    print(
        difficult[
            ["label", "precision", "recall", "f1_score", "support"]
        ].to_string(
            index=False,
            formatters={
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "f1_score": "{:.3f}".format
            }
        )
    )

    # Highest-performing classes.
    print("\n2. TEN BEST CLASSES (BY F1-SCORE)")
    print("-" * 65)

    best = report.sort_values(
        "f1_score",
        ascending=False
    ).head(10)

    print(
        best[
            ["label", "precision", "recall", "f1_score", "support"]
        ].to_string(
            index=False,
            formatters={
                "precision": "{:.3f}".format,
                "recall": "{:.3f}".format,
                "f1_score": "{:.3f}".format
            }
        )
    )

    # Most common classification errors.
    print("\n3. TOP 20 CONFUSION PAIRS")
    print("-" * 65)

    errors = []

    for actual in confusion.index:

        for predicted in confusion.columns:

            if actual == predicted:
                continue

            count = int(confusion.loc[actual, predicted])

            if count > 0:
                errors.append({
                    "actual": actual,
                    "predicted": predicted,
                    "errors": count
                })

    errors_df = pd.DataFrame(errors)

    errors_df = errors_df.sort_values(
        "errors",
        ascending=False
    ).head(20)

    print(errors_df.to_string(index=False))

    # Overall class support.
    print("\n4. CLASS SUPPORT SUMMARY")
    print("-" * 65)

    print(f"Total classes: {len(report)}")
    print(f"Total test samples: {report['support'].sum()}")
    print(f"Minimum class support: {report['support'].min()}")
    print(f"Maximum class support: {report['support'].max()}")

    print("\nAnalysis completed.")


if __name__ == "__main__":
    main()
