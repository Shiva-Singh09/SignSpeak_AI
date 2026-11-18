import csv
from pathlib import Path
from collections import defaultdict

root = Path("dataset/processed/realsign_clean")
splits = ["Training", "Validation", "Testing"]
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

print(f"{'LETTER':<9}{'TRAIN':>8}{'VAL':>8}{'TEST':>8}"
      f"{'1-HAND TRAIN':>15}{'2-HAND TRAIN':>15}")

for letter in letters:
    values = {}

    for split in splits:
        path = root / f"{split.lower()}_landmarks.csv"
        counts = {"total": 0, "one": 0, "two": 0}

        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["label"] == letter:
                    counts["total"] += 1
                    counts["one" if row["detected_hands"] == "1"
                           else "two"] += 1

        values[split] = counts

    print(
        f"{letter:<9}"
        f"{values['Training']['total']:>8}"
        f"{values['Validation']['total']:>8}"
        f"{values['Testing']['total']:>8}"
        f"{values['Training']['one']:>15}"
        f"{values['Training']['two']:>15}"
    )

print("\nHandedness category distribution in Training:")

handedness = defaultdict(int)

with (root / "training_landmarks.csv").open(
    "r", encoding="utf-8", newline=""
) as f:
    for row in csv.DictReader(f):
        for hand in ("hand_0", "hand_1"):
            category = row[f"{hand}_category"]
            if category:
                handedness[category] += 1

for category, count in sorted(handedness.items()):
    print(f"{category}: {count}")
