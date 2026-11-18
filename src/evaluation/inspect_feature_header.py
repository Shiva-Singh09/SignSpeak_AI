import csv
from pathlib import Path

path = Path("dataset/processed/realsign_features/training_features.csv")

with path.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.reader(file)
    header = next(reader)

print("Total columns:", len(header))
print("\nFirst 12 columns:")
for name in header[:12]:
    print(name)

print("\nColumns around the hand boundary:")
for name in header[58:72]:
    print(name)

print("\nLast 12 columns:")
for name in header[-12:]:
    print(name)

print("\nDuplicate column names:", len(header) - len(set(header)))
