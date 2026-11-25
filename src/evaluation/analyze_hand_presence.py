
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "dataset" / "processed"
    / "realsign_features" / "training_features.csv"
)

CLASSES = ["Q", "N", "S", "W", "M", "V", "F", "Z", "B"]


def main():
    data = pd.read_csv(DATA_PATH)

    print("\nHAND PRESENCE ANALYSIS")
    print("=" * 75)

    for label in CLASSES:
        subset = data[data["label"].astype(str) == label]

        hand_0 = subset.iloc[:, 128]
        hand_1 = subset.iloc[:, 129]

        both = ((hand_0 == 1) & (hand_1 == 1)).sum()
        only_0 = ((hand_0 == 1) & (hand_1 == 0)).sum()
        only_1 = ((hand_0 == 0) & (hand_1 == 1)).sum()

        print(f"\nClass: {label} | Samples: {len(subset)}")
        print(f"Both hands detected : {both:4d} ({both / len(subset) * 100:5.1f}%)")
        print(f"Only hand 0         : {only_0:4d} ({only_0 / len(subset) * 100:5.1f}%)")
        print(f"Only hand 1         : {only_1:4d} ({only_1 / len(subset) * 100:5.1f}%)")


if __name__ == "__main__":
    main()
