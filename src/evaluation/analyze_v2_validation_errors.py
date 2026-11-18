
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT / "outputs" / "reports"

REPORT_PATH = (
    REPORT_DIR / "v2_validation_report.csv"
)

CONFUSION_PATH = (
    REPORT_DIR / "v2_validation_confusion.csv"
)

report = pd.read_csv(REPORT_PATH, index_col=0)
matrix = pd.read_csv(CONFUSION_PATH, index_col=0)

print("\nV2 VALIDATION - WORST PERFORMING CLASSES")
print("=" * 65)

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

for col in ["precision", "recall", "f1-score", "support"]:
    report[col] = pd.to_numeric(
        report[col], errors="coerce"
    )

class_report = report.loc[classes].copy()

print(
    class_report.sort_values("f1-score")[
        ["precision", "recall", "f1-score", "support"]
    ].head(10).to_string()
)

print("\nTOP 15 CONFUSION PAIRS")
print("=" * 65)

pairs = []

for actual in classes:
    for predicted in classes:
        if actual != predicted:
            count = int(matrix.loc[actual, predicted])

            if count > 0:
                pairs.append(
                    (actual, predicted, count)
                )

pairs.sort(key=lambda x: x[2], reverse=True)

for actual, predicted, count in pairs[:15]:
    print(
        f"Actual {actual} predicted {predicted}: {count}"
    )

print("\nAnalysis completed.")
