import csv
from collections import Counter
from pathlib import Path

path = Path("dataset/processed/realsign_clean/training_landmarks.csv")

issues = Counter()
categories = Counter()

with path.open("r", encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        detected = int(row["detected_hands"])

        for index in range(2):
            hand = f"hand_{index}"
            category = row[f"{hand}_category"].strip()

            if category:
                categories[category] += 1

            if detected > index and category not in ("Left", "Right"):
                issues[f"{hand}: detected but invalid handedness"] += 1

            if detected <= index and category:
                issues[f"{hand}: absent but has handedness"] += 1

print("HANDEDNESS VALIDATION")
print("-" * 40)
print("Category counts:", dict(categories))
print("Issues:", dict(issues) if issues else "None")
print("Validation completed. No files were changed.")
