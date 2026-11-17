import csv
from collections import Counter
from pathlib import Path

path = Path("dataset/processed/realsign_clean/training_landmarks.csv")

categories = Counter()
hand_counts = Counter()
missing = Counter()

with path.open("r", encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        hand_counts[row["detected_hands"]] += 1

        for hand in ("hand_0", "hand_1"):
            category = row[f"{hand}_category"].strip()

            if category:
                categories[(hand, category)] += 1
            else:
                missing[hand] += 1

print("DETECTED HAND COUNTS")
for key, count in sorted(hand_counts.items()):
    print(f"{key} hand(s): {count}")

print("\nHANDEDNESS CATEGORIES")
for (hand, category), count in sorted(categories.items()):
    print(f"{hand} - {category}: {count}")

print("\nMISSING HANDEDNESS")
for hand, count in sorted(missing.items()):
    print(f"{hand}: {count}")

print("\nInspection completed. No files were changed.")
