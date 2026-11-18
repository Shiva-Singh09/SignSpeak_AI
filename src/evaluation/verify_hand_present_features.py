
from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "realsign_deduplicated"
    / "features"
)

FILTERED_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "realsign_deduplicated"
    / "features_hand_present"
)

EXPECTED = {
    "training": {
        "original": 18183,
        "retained": 16209,
        "removed": 1974,
    },
    "validation": {
        "original": 2597,
        "retained": 2284,
        "removed": 313,
    },
}

EXPECTED_CLASSES = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def verify_split(split):
    print(f"\nVERIFYING {split.upper()}")
    print("-" * 55)

    original_path = SOURCE_DIR / f"{split}_features.csv"
    filtered_path = FILTERED_DIR / f"{split}_features.csv"

    if not original_path.exists() or not filtered_path.exists():
        raise FileNotFoundError(f"Missing feature file for {split}")

    original = pd.read_csv(original_path)
    filtered = pd.read_csv(filtered_path)

    # 1. Verify row counts
    assert len(original) == EXPECTED[split]["original"]
    assert len(filtered) == EXPECTED[split]["retained"]

    removed = len(original) - len(filtered)
    assert removed == EXPECTED[split]["removed"]

    # 2. Verify column consistency
    assert list(original.columns) == list(filtered.columns)

    # 3. Verify all classes remain
    classes = set(filtered["label"].astype(str))
    assert classes == EXPECTED_CLASSES, (
        f"Missing classes: {EXPECTED_CLASSES - classes}"
    )

    # 4. Verify no-hand rows are absent
    hand_0 = pd.to_numeric(filtered["hand_0_present"])
    hand_1 = pd.to_numeric(filtered["hand_1_present"])

    no_hand_count = ((hand_0 == 0) & (hand_1 == 0)).sum()
    assert no_hand_count == 0

    # 5. Verify binary presence flags
    assert set(hand_0.unique()).issubset({0, 1})
    assert set(hand_1.unique()).issubset({0, 1})

    # 6. Verify every class has retained samples
    class_counts = filtered["label"].value_counts()

    print(f"Original rows:       {len(original)}")
    print(f"Filtered rows:       {len(filtered)}")
    print(f"Removed rows:        {removed}")
    print(f"Remaining classes:   {len(classes)}")
    print(f"No-hand rows:        {no_hand_count}")
    print(f"Column consistency:  PASSED")
    print(f"Class consistency:   PASSED")

    print("\nRESULT: PASSED")

    return True


def main():
    print("=" * 65)
    print("SIGNSPEAK - HAND-PRESENT FEATURE VERIFICATION")
    print("=" * 65)

    try:
        for split in ("training", "validation"):
            verify_split(split)

        print("\n" + "=" * 65)
        print("ALL VERIFICATION CHECKS PASSED")
        print("=" * 65)

    except Exception as error:
        print(f"\nVERIFICATION FAILED: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
