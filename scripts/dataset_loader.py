
from pathlib import Path
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import cv2




class NEODatasetLoader:
    """
    Reusable loader for NEO image datasets using YOLO annotations.

    Handles:
    - Finding image files in a dataset split
    - Matching images with their annotation files
    - Validating YOLO annotation values
    - Loading individual images and annotations
    - Displaying images with their bounding boxes
    """

    # File extensions that the loader recognises as images.
    # Using a set makes checking extensions straightforward.
    IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp"
    }

    # Maps each numerical YOLO class ID to its class name.
    # Our current dataset uses class 0 to represent a streak.
    CLASS_NAMES = {
        0: "streak"
    }

    def __init__(self, dataset_directory, split="train"):
        """
        Initialise the loader for a particular dataset and split.

        dataset_directory: Path to a dataset, e.g. HubbleAsteroidHunter_YOLO.
        split: Dataset split to load: train, validation, or test.
        """

        # Convert the supplied directory into a Path object.
        # Path objects make it easier to construct and inspect file paths.
        self.dataset_directory = Path(dataset_directory)

        # Store the selected split so the loader knows which images to use.
        self.split = split

        # Build paths to the images and labels folders.
        # Example: dataset/train/images and dataset/train/labels.
        self.images_directory = self.dataset_directory / split / "images"
        self.labels_directory = self.dataset_directory / split / "labels"

        # Stop early if the expected images directory does not exist.
        # This gives a clear error instead of failing later during loading.
        if not self.images_directory.is_dir():
            raise FileNotFoundError(
                f"Images directory not found: {self.images_directory}"
            )

        # Check that the matching labels directory also exists.
        if not self.labels_directory.is_dir():
            raise FileNotFoundError(
                f"Labels directory not found: {self.labels_directory}"
            )

        # Find every recognised image file in the images directory.
        # sorted() gives a predictable order, which is useful when testing.
        # This stores paths only; it does not load all images into memory.
        self.images = sorted(
            path
            for path in self.images_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.IMAGE_EXTENSIONS
        )

    def __len__(self):
        """
        Return the number of images indexed by the loader.

        This allows len(loader) to return the image count.
        """
        return len(self.images)

    @staticmethod
    def validate_annotation(annotation_text):
        """
        Validate one line of YOLO annotation text.

        Standard YOLO format:
        class_id x_center y_center width height

        Coordinates are normalised between 0 and 1.
        Returns (True, annotation_data) if valid,
        or (False, error_message) if invalid.
        """

        # Split the annotation string into individual values.
        # Example: "0 0.5 0.5 0.2 0.3" becomes five strings.
        values = annotation_text.split()

        # A standard YOLO bounding-box annotation has exactly five values.
        if len(values) != 5:
            return False, "Expected 5 values"

        # Convert the class ID and coordinates to numeric values.
        # Catch invalid text rather than allowing the function to crash.
        try:
            class_id = int(values[0])

            # The four remaining values are x-centre, y-centre, width, height.
            x_center, y_center, width, height = map(
                float, values[1:]
            )

        except ValueError:
            return False, "Annotation contains non-numeric values"

        # Check that the class ID exists in our current class mapping.
        if class_id not in NEODatasetLoader.CLASS_NAMES:
            return False, f"Unknown class ID: {class_id}"

        # Collect the four coordinates into one list for validation.
        coordinates = [x_center, y_center, width, height]

        # YOLO coordinates must be normalised between 0 and 1 inclusive.
        # all() returns True only if every coordinate passes the check.
        if not all(0 <= value <= 1 for value in coordinates):
            return False, "Coordinates must be between 0 and 1"

        # A bounding box must have a positive width and height.
        # Zero-sized boxes cannot cover an area.
        if width <= 0 or height <= 0:
            return False, "Width and height must be positive"

        # Return the validated information in a dictionary.
        # Named fields are easier to understand than remembering list indexes.
        return True, {
            "class_id": class_id,
            "class_name": NEODatasetLoader.CLASS_NAMES[class_id],
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        }

    def get_sample(self, index):
        """
        Load one image and its corresponding YOLO annotations.

        index: Position of the image in self.images.

        Returns a dictionary containing paths, the image,
        valid annotations, and any errors encountered.
        """

        # Select an image using its position in the indexed image list.
        image_path = self.images[index]

        # Match the label to the image using its filename without the extension.
        # For example, image_001.jpg matches image_001.txt.
        label_path = (
            self.labels_directory / f"{image_path.stem}.txt"
        )

        # OpenCV loads the image into a NumPy array.
        # OpenCV uses BGR colour order by default, not RGB.
        image = cv2.imread(str(image_path))

        # Store all information about this sample in one dictionary.
        # Starting with empty lists makes it easy to record multiple
        # annotations or errors for a single image.
        sample = {
            "image_path": image_path,
            "label_path": label_path,
            "image": image,
            "label_exists": label_path.is_file(),
            "annotations": [],
            "errors": [],
        }

        # If OpenCV cannot read the image, record the problem and stop.
        if image is None:
            sample["errors"].append("Image could not be loaded")
            return sample

        # Report a missing label file instead of trying to open it.
        if not label_path.is_file():
            sample["errors"].append("Label file is missing")
            return sample

        # Open the annotation file using UTF-8 text encoding.
        with open(label_path, "r", encoding="utf-8") as file:

            # Read each line separately.
            # enumerate(..., start=1) numbers lines from 1 for clearer errors.
            for line_number, line in enumerate(file, start=1):

                # Ignore blank lines instead of treating them as annotations.
                if not line.strip():
                    continue

                # Validate this annotation line using our validation method.
                valid, result = self.validate_annotation(line)

                if valid:
                    # Save the parsed annotation for later visualisation
                    # or use by other parts of the project.
                    sample["annotations"].append(result)

                else:
                    # Keep track of malformed annotations without
                    # stopping the entire dataset-loading process.
                    sample["errors"].append(
                        f"Line {line_number}: {result}"
                    )

        # Return the complete sample once processing is finished.
        return sample

    @staticmethod
    def show_sample(sample):
        """
        Display a loaded image with its validated bounding boxes.

        This method visualises annotations; it does not run AI inference.
        """

        # Retrieve the image array from the sample dictionary.
        image = sample["image"]

        # A failed image load means there is nothing to display.
        if image is None:
            print("Cannot display image: image failed to load")
            return

        # Convert OpenCV's BGR image into RGB for Matplotlib.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions so normalised coordinates can be
        # converted into pixel coordinates.
        image_height, image_width = image.shape[:2]

        # Create a figure and an axes object for displaying the image
        # and drawing rectangles on top of it.
        fig, ax = plt.subplots()

        # Display the image in RGB colour order.
        ax.imshow(image_rgb)

        # Draw one rectangle for every valid annotation in this image.
        for annotation in sample["annotations"]:

            # Convert normalised centre coordinates and dimensions to pixels.
            x_center = annotation["x_center"] * image_width
            y_center = annotation["y_center"] * image_height
            box_width = annotation["width"] * image_width
            box_height = annotation["height"] * image_height

            # YOLO stores the centre of the box, whereas Rectangle
            # expects the top-left corner.
            box_left = x_center - box_width / 2
            box_top = y_center - box_height / 2

            # Draw an outline around the labelled streak.
            # facecolor="none" keeps the image visible inside the box.
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

        # Display the filename as the plot title.
        ax.set_title(sample["image_path"].name)

        # Hide axis ticks so the image is easier to inspect.
        ax.axis("off")

        # Render the figure.
        plt.show()
