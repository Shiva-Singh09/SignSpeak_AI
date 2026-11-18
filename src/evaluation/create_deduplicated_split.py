
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "external"
    / "RealSign"
    / "extracted"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "processed"
    / "realsign_deduplicated"
)

IMAGE_OUTPUT = OUTPUT_ROOT / "images"
REPORT_OUTPUT = OUTPUT_ROOT / "reports"

SPLIT_RATIOS = {
    "Training": 0.70,
    "Validation": 0.10,
    "Testing": 0.20,
}

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def calculate_sha256(file_path):
    digest = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def is_readable_image(file_path):
    image = cv2.imread(str(file_path))
    return image is not None



def assign_groups_to_splits(groups):
    """
    Assign complete duplicate groups to splits while
    targeting the requested overall image proportions.
    """

    rng = random.Random(SEED)
    groups = list(groups)
    rng.shuffle(groups)

    # Process larger duplicate groups first.
    groups.sort(
        key=lambda group: len(group["images"]),
        reverse=True
    )

    total_images = sum(
        len(group["images"]) for group in groups
    )

    targets = {
        split: total_images * ratio
        for split, ratio in SPLIT_RATIOS.items()
    }

    assigned = {
        split: [] for split in SPLIT_RATIOS
    }

    counts = Counter()

    for group in groups:

        group_size = len(group["images"])

        # Choose the split that produces the smallest
        # increase (or greatest reduction) in total error.
        def assignment_cost(split):

            current_error = (
                counts[split] - targets[split]
            ) ** 2

            new_error = (
                counts[split] + group_size - targets[split]
            ) ** 2

            return new_error - current_error

        best_split = min(
            SPLIT_RATIOS,
            key=assignment_cost
        )

        assigned[best_split].append(group)
        counts[best_split] += group_size

    return assigned


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 65)
    print("CREATING EXACT-DUPLICATE-FREE REALSIGN SPLITS")
    print("=" * 65)

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {SOURCE_ROOT}"
        )

    # Prevent accidental mixing with an earlier run.
    if IMAGE_OUTPUT.exists() and any(IMAGE_OUTPUT.rglob("*")):
        raise RuntimeError(
            "Output images folder is not empty. "
            "Stop and inspect it before running again."
        )

    REPORT_OUTPUT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1. Discover images and calculate hashes
    # --------------------------------------------------

    print("\n[1/6] Scanning original images...")

    hash_groups = defaultdict(list)
    unreadable_records = []
    total_discovered = 0

    for original_split in SPLIT_RATIOS:

        split_root = SOURCE_ROOT / original_split

        if not split_root.exists():
            raise FileNotFoundError(
                f"Missing source split: {split_root}"
            )

        for class_dir in sorted(split_root.iterdir()):

            if not class_dir.is_dir():
                continue

            label = class_dir.name.upper()

            for image_path in sorted(class_dir.iterdir()):

                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                total_discovered += 1

                if not is_readable_image(image_path):
                    unreadable_records.append({
                        "source_split": original_split,
                        "label": label,
                        "source_path": str(image_path),
                    })
                    continue

                file_hash = calculate_sha256(image_path)

                hash_groups[file_hash].append({
                    "path": image_path,
                    "original_split": original_split,
                    "label": label,
                })

    print(f"Discovered images: {total_discovered}")
    print(f"Unreadable images: {len(unreadable_records)}")
    print(f"Readable images: {sum(map(len, hash_groups.values()))}")

    # --------------------------------------------------
    # 2. Check for cross-class exact duplicates
    # --------------------------------------------------

    print("\n[2/6] Checking duplicate group labels...")

    cross_class_groups = []

    for file_hash, images in hash_groups.items():

        labels = {image["label"] for image in images}

        if len(labels) > 1:
            cross_class_groups.append({
                "hash": file_hash,
                "labels": sorted(labels),
                "images": images,
            })

    if cross_class_groups:

        report_path = REPORT_OUTPUT / "cross_class_exact_duplicates.csv"

        with open(report_path, "w", newline="", encoding="utf-8") as file:

            writer = csv.writer(file)
            writer.writerow([
                "sha256",
                "labels",
                "source_split",
                "source_path",
            ])

            for group in cross_class_groups:
                for image in group["images"]:
                    writer.writerow([
                        group["hash"],
                        "|".join(group["labels"]),
                        image["original_split"],
                        str(image["path"]),
                    ])

        raise RuntimeError(
            f"Found {len(cross_class_groups)} readable "
            "cross-class duplicate groups. "
            f"Inspect: {report_path}"
        )

    print("No readable cross-class exact duplicate groups found.")

    # --------------------------------------------------
    # 3. Build groups separately for each alphabet class
    # --------------------------------------------------

    print("\n[3/6] Grouping exact duplicates...")

    class_groups = defaultdict(list)

    for file_hash, images in hash_groups.items():

        label = images[0]["label"]

        class_groups[label].append({
            "group_id": file_hash,
            "images": images,
        })

    expected_classes = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    actual_classes = set(class_groups)

    if actual_classes != expected_classes:
        raise RuntimeError(
            "Expected all 26 alphabet classes.\n"
            f"Missing: {sorted(expected_classes - actual_classes)}\n"
            f"Unexpected: {sorted(actual_classes - expected_classes)}"
        )

    # --------------------------------------------------
    # 4. Assign each complete group to one split
    # --------------------------------------------------

    print("\n[4/6] Assigning groups to Training/Validation/Testing...")

    final_assignments = {}

    for label in sorted(class_groups):

        groups = class_groups[label]

        if len(groups) < 3:
            raise RuntimeError(
                f"Class {label} has fewer than 3 unique groups."
            )

        assignments = assign_groups_to_splits(groups)

        for split, split_groups in assignments.items():
            for group in split_groups:
                final_assignments[(label, group["group_id"])] = split

    # --------------------------------------------------
    # 5. Copy images and create manifest
    # --------------------------------------------------

    print("\n[5/6] Copying images into new split folders...")

    manifest_path = REPORT_OUTPUT / "dataset_manifest.csv"

    split_image_counts = Counter()
    split_class_counts = defaultdict(Counter)
    copied_hash_splits = defaultdict(set)

    with open(
        manifest_path, "w", newline="", encoding="utf-8"
    ) as manifest_file:

        writer = csv.writer(manifest_file)

        writer.writerow([
            "label",
            "new_split",
            "original_split",
            "source_path",
            "destination_path",
            "sha256",
            "group_id",
        ])

        for (label, group_id), new_split in sorted(
            final_assignments.items()
        ):

            group = next(
                item for item in class_groups[label]
                if item["group_id"] == group_id
            )

            for image in group["images"]:

                source_path = image["path"]

                # Prefix source split to avoid filename collisions.
                destination_name = (
                    f'{image["original_split"]}__{source_path.name}'
                )

                destination_dir = IMAGE_OUTPUT / new_split / label
                destination_dir.mkdir(parents=True, exist_ok=True)

                destination_path = destination_dir / destination_name

                if destination_path.exists():
                    raise RuntimeError(
                        f"Destination collision: {destination_path}"
                    )

                shutil.copy2(source_path, destination_path)

                split_image_counts[new_split] += 1
                split_class_counts[new_split][label] += 1
                copied_hash_splits[group_id].add(new_split)

                writer.writerow([
                    label,
                    new_split,
                    image["original_split"],
                    str(source_path),
                    str(destination_path),
                    group_id,
                    group_id,
                ])

    # --------------------------------------------------
    # 6. Save reports and verify
    # --------------------------------------------------

    print("\n[6/6] Verifying the new dataset...")

    # Every exact duplicate group must belong to one split.
    group_split_check = defaultdict(set)

    with open(manifest_path, newline="", encoding="utf-8") as file:

        reader = csv.DictReader(file)

        for row in reader:
            group_split_check[row["group_id"]].add(row["new_split"])

    leaking_groups = {
        group_id: splits
        for group_id, splits in group_split_check.items()
        if len(splits) != 1
    }

    if leaking_groups:
        raise RuntimeError(
            f"Exact duplicate leakage detected: {len(leaking_groups)} groups"
        )

    # Verify all 26 classes exist in every split.
    for split in SPLIT_RATIOS:

        missing = expected_classes - set(split_class_counts[split])

        if missing:
            raise RuntimeError(
                f"{split} is missing classes: {sorted(missing)}"
            )

    # Save unreadable-image report.
    unreadable_path = REPORT_OUTPUT / "unreadable_images.csv"

    with open(
        unreadable_path, "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["source_split", "label", "source_path"]
        )

        writer.writeheader()
        writer.writerows(unreadable_records)

    # Save split summary.
    summary_path = REPORT_OUTPUT / "split_summary.txt"

    with open(summary_path, "w", encoding="utf-8") as file:

        file.write("REALSign DEDUPLICATED DATASET SUMMARY\n")
        file.write("=" * 55 + "\n\n")

        file.write(f"Discovered images: {total_discovered}\n")
        file.write(f"Unreadable images excluded: {len(unreadable_records)}\n")
        file.write(
            f"Readable images copied: {sum(split_image_counts.values())}\n"
        )
        file.write(f"Unique exact-hash groups: {len(hash_groups)}\n")
        file.write(
            f"Exact duplicate groups: "
            f"{sum(1 for group in hash_groups.values() if len(group) > 1)}\n\n"
        )

        for split in SPLIT_RATIOS:

            file.write(
                f"{split}: {split_image_counts[split]} images\n"
            )

            for label in sorted(expected_classes):
                file.write(
                    f"  {label}: {split_class_counts[split][label]}\n"
                )

            file.write("\n")

        file.write(
            f"Groups crossing splits: {len(leaking_groups)}\n"
        )

    print("\n" + "=" * 65)
    print("DATASET SPLIT CREATION COMPLETED")
    print("=" * 65)

    print(f"\nTotal readable images: {sum(split_image_counts.values())}")

    for split in SPLIT_RATIOS:
        print(
            f"{split}: {split_image_counts[split]} images "
            f"({split_image_counts[split] / sum(split_image_counts.values()) * 100:.2f}%)"
        )

    print(f"\nUnique groups: {len(hash_groups)}")
    print(f"Groups crossing splits: {len(leaking_groups)}")
    print("Classes in every split: 26")
    print(f"\nOutput: {OUTPUT_ROOT}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
