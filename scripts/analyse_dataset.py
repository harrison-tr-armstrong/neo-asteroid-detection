
from pathlib import Path
import math

# Resolve the project root and locate the NEOBench datasets.
project_root = Path(__file__).resolve().parent.parent

datasets_root = (
    project_root
    / "NEOBench"
    / "NEOBench_data"
    / "datasets"
)

# Supported file extensions.
image_extensions = {".jpg", ".jpeg", ".png"}
label_extensions = {".txt"}
splits = ("train", "validation", "test")


###############################################
# Validate that the dataset directory exists.

if not datasets_root.is_dir():
    raise FileNotFoundError(
        f"Dataset directory not found: {datasets_root}"
    )

print(f"Dataset root: {datasets_root}")


###############################################
# Analyse each dataset.

for child in sorted(datasets_root.iterdir()):
    if not child.is_dir():
        continue

    print("\n" + "=" * 60)
    print(f"Dataset: {child.name}")
    print("=" * 60)

    dataset_image_total = 0
    dataset_class_ids = set()

    # Process training, validation and test independently.
    for split in splits:
        images_dir = child / split / "images"
        labels_dir = child / split / "labels"

        print(f"\n--- {split.upper()} ---")

        if not images_dir.is_dir():
            print(f"Images directory not found: {images_dir}")
            continue

        # Collect all images, including nested directories.
        images = [
            file
            for file in images_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in image_extensions
        ]

        total_images = len(images)
        dataset_image_total += total_images

        print(f"Image directory: {images_dir}")
        print(f"Total images: {total_images}")

        # Collect all YOLO annotation files if the directory exists.
        if labels_dir.is_dir():
            labels = [
                file
                for file in labels_dir.rglob("*")
                if file.is_file()
                and file.suffix.lower() in label_extensions
            ]
        else:
            labels = []
            print("Labels directory not found.")

        # Map annotation filenames to their relative paths.
        # This supports matching files in nested folders.
        label_lookup = {
            file.relative_to(labels_dir)
            .with_suffix("")
            .as_posix(): file
            for file in labels
        } if labels_dir.is_dir() else {}

        # Counters for this split.
        positive_count = 0
        negative_count = 0
        missing_label_count = 0
        invalid_label_count = 0

        # Inspect each image and its corresponding label.
        for image_path in images:
            relative_image = image_path.relative_to(images_dir)

            label_key = (
                relative_image
                .with_suffix("")
                .as_posix()
            )

            label_path = label_lookup.get(label_key)

            # UAEMMN negative images are identified by their filenames.
            # They do not need annotation files to count as negatives.
            if (
                child.name == "UAEMMN_YOLO"
                and image_path.stem.startswith("uaemmn_neg_")
            ):
                negative_count += 1

                # Flag an unexpected annotation on a negative image.
                if label_path is not None:
                    with open(
                        label_path,
                        "r",
                        encoding="utf-8"
                    ) as label_file:
                        has_annotations = any(
                            line.strip()
                            for line in label_file
                        )

                    if has_annotations:
                        print(
                            "WARNING: UAEMMN negative image has "
                            f"non-empty labels: {image_path.name}"
                        )

                continue

            # Images without matching labels need investigation.
            if label_path is None:
                missing_label_count += 1
                continue

            # Read non-empty annotation lines.
            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as label_file:
                lines = [
                    line.strip()
                    for line in label_file
                    if line.strip()
                ]

            # An existing empty label file is a negative example.
            if not lines:
                negative_count += 1
                continue

            # Validate each YOLO bounding-box annotation.
            valid_label = True

            for line in lines:
                parts = line.split()

                # Standard YOLO detection format:
                # class_id x_center y_center width height
                if len(parts) != 5:
                    valid_label = False
                    break

                try:
                    class_id = int(parts[0])

                    x_center, y_center, width, height = map(
                        float,
                        parts[1:]
                    )

                except ValueError:
                    valid_label = False
                    break

                coordinates = (
                    x_center,
                    y_center,
                    width,
                    height
                )

                if (
                    class_id < 0
                    or not all(
                        math.isfinite(value)
                        for value in coordinates
                    )
                    or not all(
                        0 <= value <= 1
                        for value in coordinates
                    )
                    or width <= 0
                    or height <= 0
                ):
                    valid_label = False
                    break

            if valid_label:
                positive_count += 1
            else:
                invalid_label_count += 1

        # Count distinct class IDs across this split's labels.
        split_class_ids = set()

        for label_path in labels:
            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as label_file:
                for line in label_file:
                    values = line.split()

                    if not values:
                        continue

                    try:
                        class_id = int(values[0])
                    except ValueError:
                        continue

                    if class_id >= 0:
                        split_class_ids.add(class_id)

        dataset_class_ids.update(split_class_ids)

        # Display split results.
        accounted_images = (
            positive_count
            + negative_count
            + missing_label_count
            + invalid_label_count
        )

        print(f"Label files found: {len(labels)}")
        print(f"Distinct class IDs: {sorted(split_class_ids)}")
        print(f"Positive images: {positive_count}")
        print(f"Negative images: {negative_count}")
        print(f"Missing label files: {missing_label_count}")
        print(f"Invalid label files: {invalid_label_count}")
        print(
            f"Accounted for: "
            f"{accounted_images}/{total_images} images"
        )

        if accounted_images != total_images:
            print("WARNING: Image counts do not match!")

    # Display dataset-wide summary.
    print(f"\n--- {child.name} SUMMARY ---")
    print(
        "Total images across available splits: "
        f"{dataset_image_total}"
    )
    print(
        "Distinct class IDs across all splits: "
        f"{len(dataset_class_ids)}"
    )
    print(f"Class IDs: {sorted(dataset_class_ids)}")


print("\n" + "=" * 60)
print("DATASET ANALYSIS COMPLETE")
print("=" * 60)
