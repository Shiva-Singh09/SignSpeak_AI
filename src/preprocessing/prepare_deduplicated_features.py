
import csv
from pathlib import Path

import numpy as np


# All paths remain inside the SignSpeak_AI project.
BASE = Path("dataset/processed/realsign_deduplicated")

SOURCE = BASE / "landmarks"
OUTPUT = BASE / "features"

SPLITS = ("training", "validation", "testing")

OUTPUT.mkdir(parents=True, exist_ok=True)


def make_feature_names():
    names = []

    for hand in range(2):
        prefix = f"hand_{hand}"

        names.extend([
            f"{prefix}_wrist_x",
            f"{prefix}_wrist_y",
            f"{prefix}_wrist_z",
        ])

        for point in range(1, 21):
            for axis in ("x", "y", "z"):
                names.append(
                    f"{prefix}_landmark_{point:02d}_rel_{axis}"
                )

    names.extend([
        "hand_0_present",
        "hand_1_present",
        "hand_0_is_left",
        "hand_0_is_right",
        "hand_1_is_left",
        "hand_1_is_right",
    ])

    return names


FEATURE_NAMES = make_feature_names()


def extract_hand_features(row, hand_index):
    prefix = f"hand_{hand_index}"

    coordinates = np.array([
        [
            float(row[f"{prefix}_{point:02d}_{axis}"])
            for axis in ("x", "y", "z")
        ]
        for point in range(21)
    ], dtype=np.float32)

    category = row[f"{prefix}_category"].strip()

    present = int(category in ("Left", "Right"))

    if not present:
        return np.zeros(63, dtype=np.float32), 0, 0, 0

    wrist = coordinates[0].copy()

    relative = coordinates[1:] - wrist

    # Scale relative coordinates using maximum 3D
    # distance from wrist to any other landmark.
    distances = np.linalg.norm(relative, axis=1)
    scale = float(np.max(distances))

    if scale > 1e-8:
        relative = relative / scale
    else:
        relative = np.zeros_like(relative)

    features = np.concatenate([
        wrist,
        relative.flatten(),
    ]).astype(np.float32)

    is_left = int(category == "Left")
    is_right = int(category == "Right")

    return features, present, is_left, is_right


def process_split(split):
    source_path = SOURCE / f"{split}_landmarks.csv"
    output_path = OUTPUT / f"{split}_features.csv"

    if not source_path.exists():
        raise FileNotFoundError(
            f"Landmark file not found: {source_path}"
        )

    count = 0
    class_counts = {}

    with source_path.open(
        "r", encoding="utf-8", newline=""
    ) as source_file:

        reader = csv.DictReader(source_file)

        with output_path.open(
            "w", encoding="utf-8", newline=""
        ) as output_file:

            writer = csv.writer(output_file)

            writer.writerow([
                "label",
                "image_name",
                *FEATURE_NAMES,
            ])

            for row in reader:
                hand_0, present_0, left_0, right_0 = (
                    extract_hand_features(row, 0)
                )

                hand_1, present_1, left_1, right_1 = (
                    extract_hand_features(row, 1)
                )

                metadata = [
                    row["label"],
                    row["image_name"],
                ]

                features = np.concatenate([
                    hand_0,
                    hand_1,
                    np.array([
                        present_0,
                        present_1,
                        left_0,
                        right_0,
                        left_1,
                        right_1,
                    ], dtype=np.float32),
                ])

                if features.size != 132:
                    raise ValueError(
                        f"Expected 132 features, got {features.size}"
                    )

                if not np.isfinite(features).all():
                    raise ValueError(
                        f"Invalid numeric feature in {row['image_name']}"
                    )

                writer.writerow([
                    *metadata,
                    *[f"{value:.7f}" for value in features],
                ])

                count += 1

                class_counts[row["label"]] = (
                    class_counts.get(row["label"], 0) + 1
                )

    print(f"\n{split.upper()}")
    print("-" * 40)
    print(f"Rows written: {count}")
    print(f"Classes: {len(class_counts)}")
    print(f"Features per image: {len(FEATURE_NAMES)}")
    print(f"Output: {output_path}")

    expected_classes = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    if set(class_counts) != expected_classes:
        raise ValueError(
            f"Unexpected class coverage in {split}: "
            f"{sorted(class_counts)}"
        )


def main():
    print("DEDUPLICATED REALSIGN FEATURE PREPROCESSING")
    print("=" * 55)

    print(f"Source: {SOURCE}")
    print(f"Output: {OUTPUT}")

    for split in SPLITS:
        process_split(split)

    print("\n" + "=" * 55)
    print("ALL SPLITS PROCESSED SUCCESSFULLY")
    print(f"Features per image: {len(FEATURE_NAMES)}")
    print("Original datasets and landmark files were not modified.")


if __name__ == "__main__":
    main()
