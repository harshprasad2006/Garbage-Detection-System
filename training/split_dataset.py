"""
split_dataset.py

Purpose:
    Re-split the Roboflow-exported garbage detection dataset into a proper
    70% / 20% / 10% train/valid/test split, WITHOUT data leakage.

Why this is needed:
    Roboflow's original export applied augmentation ("outputs per training
    example: 3") to the train split only, and used its own arbitrary
    train/valid/test ratio (92/4/4 in our case). Augmented copies of the same
    source photo share a filename prefix before ".rf.<hash>.jpg". If we
    naively re-split by individual file, siblings of the same source image
    could land in different splits -> data leakage -> inflated test metrics.

    This script groups files by their source-image prefix BEFORE splitting,
    so all augmented siblings of a source image always stay together in the
    same split.

Usage:
    python split_dataset.py

Requires:
    Run this from inside the `training/` folder, with the original Roboflow
    export sitting at ../dataset/{train,valid,test}/{images,labels}
"""

import re
import random
import shutil
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 42                      # fixed seed -> reproducible split
TRAIN_RATIO = 0.70
VALID_RATIO = 0.20
TEST_RATIO = 0.10

SOURCE_ROOT = Path("../dataset")            # original Roboflow export
OUTPUT_ROOT = Path("../dataset_split")       # new leakage-safe split

SOURCE_SPLITS = ["train", "valid", "test"]   # folders to pool together

# Matches everything before ".rf." as the group key
# e.g. "battery_7_jpg.rf.051948699a590b7bd254575643e7c008.jpg" -> "battery_7_jpg"
GROUP_PATTERN = re.compile(r"^(.*)\.rf\.[a-f0-9]+$")


def get_group_key(filename_stem: str) -> str:
    """Extract the source-image group key from a filename stem (no extension)."""
    match = GROUP_PATTERN.match(filename_stem)
    if match:
        return match.group(1)
    # Fallback: if a file doesn't match the pattern (shouldn't normally happen),
    # treat it as its own unique group so it's never silently dropped.
    return filename_stem


def collect_all_pairs():
    """
    Walk train/valid/test image folders, find matching label files,
    and group (image_path, label_path) tuples by source-image group key.
    """
    groups = defaultdict(list)
    missing_labels = []

    for split in SOURCE_SPLITS:
        images_dir = SOURCE_ROOT / split / "images"
        labels_dir = SOURCE_ROOT / split / "labels"

        if not images_dir.exists():
            print(f"WARNING: {images_dir} does not exist, skipping.")
            continue

        for image_path in images_dir.glob("*.jpg"):
            label_path = labels_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                missing_labels.append(image_path.name)
                continue

            group_key = get_group_key(image_path.stem)
            groups[group_key].append((image_path, label_path))

    if missing_labels:
        print(f"WARNING: {len(missing_labels)} images had no matching label file. "
              f"First few: {missing_labels[:5]}")

    return groups


def split_groups(groups: dict):
    """Shuffle group keys with a fixed seed, then split into train/valid/test."""
    group_keys = list(groups.keys())
    random.Random(SEED).shuffle(group_keys)

    n = len(group_keys)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    # remainder goes to test, so all groups are accounted for
    n_test = n - n_train - n_valid

    train_keys = group_keys[:n_train]
    valid_keys = group_keys[n_train:n_train + n_valid]
    test_keys = group_keys[n_train + n_valid:]

    return {
        "train": train_keys,
        "valid": valid_keys,
        "test": test_keys,
    }, (n_train, n_valid, n_test)


def copy_split(split_name: str, keys: list, groups: dict):
    """Copy all image/label pairs belonging to the given group keys into
    the new output structure for this split."""
    out_images = OUTPUT_ROOT / split_name / "images"
    out_labels = OUTPUT_ROOT / split_name / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    for key in keys:
        for image_path, label_path in groups[key]:
            shutil.copy2(image_path, out_images / image_path.name)
            shutil.copy2(label_path, out_labels / label_path.name)
            image_count += 1

    return image_count


def main():
    print("Scanning original dataset and grouping by source image...")
    groups = collect_all_pairs()

    total_images = sum(len(v) for v in groups.values())
    print(f"Total images found: {total_images}")
    print(f"Total unique source-image groups: {len(groups)}")

    print("\nSplitting groups (70/20/10) with fixed seed =", SEED)
    split_assignment, (n_train, n_valid, n_test) = split_groups(groups)
    print(f"Groups -> train: {n_train}, valid: {n_valid}, test: {n_test}")

    print("\nCopying files into new split-safe structure at", OUTPUT_ROOT)
    counts = {}
    for split_name, keys in split_assignment.items():
        counts[split_name] = copy_split(split_name, keys, groups)

    print("\n===== FINAL IMAGE COUNTS =====")
    total = sum(counts.values())
    for split_name, count in counts.items():
        pct = 100 * count / total
        print(f"{split_name:6s}: {count:5d} images ({pct:.1f}%)")
    print(f"{'total':6s}: {total:5d} images")

    if total != total_images:
        print(f"\nWARNING: total after split ({total}) != original total "
              f"({total_images}). Check for skipped/missing-label files above.")


if __name__ == "__main__":
    main()
