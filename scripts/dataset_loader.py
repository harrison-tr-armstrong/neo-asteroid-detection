
from pathlib import Path

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cv2


class NEODatasetLoader:
    """Load NEO images, validate YOLO annotations, and visualise bounding boxes."""

    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png"
    }

    CLASS_NAMES = {
        0: "streak"
    }

    def __init__(self, dataset_directory, split="train"):
        """Initialise the loader for a dataset split."""

        self.dataset_directory = Path(dataset_directory)
        self.split = split

        self.images_directory = self.dataset_directory / split / "images"
        self.labels_directory = self.dataset_directory / split / "labels"

        if not self.images_directory.is_dir():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_directory}"
            )

        if not self.labels_directory.is_dir():
            raise FileNotFoundError(
                f"Labels directory not found: {self.labels_directory}"
            )

        # Index image paths without loading every image into memory.
        self.images = sorted(
            path
            for path in self.images_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.IMAGE_EXTENSIONS
        )

    def __len__(self):
        """Return the number of indexed images."""
        return len(self.images)

    @staticmethod
    def validate_annotation(annotation_text):
        """Validate and parse one YOLO bounding-box annotation."""

        values = annotation_text.split()

        if len(values) != 5:
            return False, "Expected 5 values"

        try:
            class_id = int(values[0])
            x_center, y_center, width, height = map(float, values[1:])
        except ValueError:
            return False, "Annotation contains non-numeric values"

        if class_id not in NEODatasetLoader.CLASS_NAMES:
            return False, f"Unknown class ID: {class_id}"

        coordinates = [x_center, y_center, width, height]

        if not all(0 <= value <= 1 for value in coordinates):
            return False, "Coordinates must be between 0 and 1"

        if width <= 0 or height <= 0:
            return False, "Width and height must be positive"

        return True, {
            "class_id": class_id,
            "class_name": NEODatasetLoader.CLASS_NAMES[class_id],
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }

    def get_sample(self, index):
        """Load an image and its corresponding annotations."""

        image_path = self.images[index]
        label_path = self.labels_directory / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))

        sample = {
            "image_path": image_path,
            "label_path": label_path,
            "image": image,
            "label_exists": label_path.is_file(),
            "annotations": [],
            "errors": [],
        }

        if image is None:
            sample["errors"].append("Image could not be loaded")
            return sample

        if not label_path.is_file():
            sample["errors"].append("Label file is missing")
            return sample

        with open(label_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                valid, result = self.validate_annotation(line)

                if valid:
                    sample["annotations"].append(result)
                else:
                    sample["errors"].append(
                        f"Line {line_number}: {result}"
                    )

        return sample

    @staticmethod
    def show_sample(sample):
        """Display an image with its validated bounding boxes."""

        image = sample["image"]

        if image is None:
            print("Cannot display image: image failed to load")
            return

        # OpenCV uses BGR; Matplotlib expects RGB.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width = image.shape[:2]

        fig, ax = plt.subplots()
        ax.imshow(image_rgb)

        for annotation in sample["annotations"]:
            x_center = annotation["x_center"] * image_width
            y_center = annotation["y_center"] * image_height
            box_width = annotation["width"] * image_width
            box_height = annotation["height"] * image_height

            # YOLO uses centre coordinates; Rectangle uses the top-left corner.
            box_left = x_center - box_width / 2
            box_top = y_center - box_height / 2

            ax.add_patch(
                Rectangle(
                    (box_left, box_top),
                    box_width,
                    box_height,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
            )

        ax.set_title(sample["image_path"].name)
        ax.axis("off")
        plt.show()
