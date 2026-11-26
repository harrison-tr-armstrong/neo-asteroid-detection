from pathlib import Path
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import numpy as np
from ultralytics.engine.results import Results


class NEOInference:
   # Initialise the YOLO model and resolve the checkpoint path relative to the project root.
   def __init__(self, model_path="models/neo_yolo11s_best.pt"):
      project_root = Path(__file__).resolve().parent.parent
      self.model_path = project_root / model_path

      # Check that the trained model checkpoint exists before attempting to load it.
      if not self.model_path.exists():
         raise FileNotFoundError(
            f"Model checkpoint not found: {self.model_path}"
         )

      # Load the trained YOLO model so it can be reused for predictions.
      self.model = YOLO(str(self.model_path))

   # Run object detection on a single image and optionally save an annotated version.
   def predict(
         self,
         image_path: str | Path,
         conf: float = 0.10,
         iou: float = 0.40,
         imgsz: int = 640,
         save_annotated: bool = True,
         output_dir: str | Path = "runs/inference"
   ) -> dict:

      image_path = Path(image_path)

      # Define the image formats supported by the inference pipeline.
      valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

      # Check that the supplied image path points to an existing file.
      if not image_path.exists():
         raise FileNotFoundError(
            f"Image not found: {image_path}"
         )

      # Validate the file extension before attempting to process the image.
      if image_path.suffix.lower() not in valid_extensions:
         raise ValueError("Unsupported image format.")

      # Verify that the file contains valid image data rather than relying only on its extension.
      try:
         with Image.open(image_path) as image:
            image.verify()

      except (UnidentifiedImageError, OSError):
         raise ValueError("This file is not a valid image")

      # Validate the confidence and IoU inputs to prevent invalid prediction settings.
      if isinstance(conf, bool) or not isinstance(conf, (int, float)):
         raise TypeError("Confidence must be a number.")

      if isinstance(iou, bool) or not isinstance(iou, (int, float)):
         raise TypeError("IoU must be a number.")

      # Both confidence and IoU must be within the range of 0 to 1.
      if not 0 <= conf <= 1:
         raise ValueError("Confidence must be between 0 and 1.")

      if not 0 <= iou <= 1:
         raise ValueError("IoU must be between 0 and 1.")

      # Validate the image size parameter before passing it to YOLO.
      if isinstance(imgsz, bool) or not isinstance(imgsz, int):
         raise TypeError("Image size must be an integer.")

      if imgsz <= 0:
         raise ValueError("Image size must be greater than 0.")

      # Run YOLO inference using the supplied image and prediction settings.
      results = self.model.predict(
         source=str(image_path),
         conf=conf,
         iou=iou,
         imgsz=imgsz,
         verbose=False
      )

      detections = []

      # Extract the detected objects and their information from the YOLO results.
      for result in results:
         for box in result.boxes:

            # Retrieve bounding box coordinates in x1, y1, x2, y2 format.
            coordinates = box.xyxy[0].tolist()

            # Build a structured dictionary containing the class, confidence and bounding box.
            detection = {
               "class": result.names[int(box.cls[0])],
               "confidence": float(box.conf[0]),
               "bounding_box": {
                  "x1": coordinates[0],
                  "y1": coordinates[1],
                  "x2": coordinates[2],
                  "y2": coordinates[3]
               }
            }

            detections.append(detection)

      # Initialise the annotated image path as None in case saving is disabled.
      output_path = None

      # Generate and save an annotated image if requested.
      if save_annotated:
         annotated_image = self.annotate(results[0])
         output_path = self.save_prediction(annotated_image, image_path, output_dir)

      # Return the detection information and the annotated image path.
      return {
         "detections": detections,
         "annotated_image": str(output_path) if output_path else None
      }

   # Run inference on multiple images while handling errors for individual files.
   def predict_batch(
         self,
         image_paths: list[str | Path],
         conf: float = 0.10,
         iou: float = 0.40,
         imgsz: int = 640,
         save_annotated: bool = True,
         output_dir: str | Path = "runs/inference"
   ) -> list[dict]:

      # Ensure the input is a list containing at least one image path.
      if not isinstance(image_paths, list):
         raise TypeError("Image paths must be provided as a list.")

      if not image_paths:
         raise ValueError("At least one image path must be provided.")

      results = []

      # Process each image separately so one invalid image does not stop the entire batch.
      for image_path in image_paths:
         try:
            # Check that each individual path has a supported Python type.
            if not isinstance(image_path, (str, Path)):
               raise TypeError("Each image path must be a string or Path object.")

            # Reuse the single-image prediction method for consistent processing.
            result = self.predict(
               image_path,
               conf=conf,
               iou=iou,
               imgsz=imgsz,
               save_annotated=save_annotated,
               output_dir=output_dir
            )

            # Attach the source image path and mark the prediction as successful.
            result["image"] = str(image_path)
            result["status"] = "success"

            results.append(result)

         # Record supported validation and file errors without stopping the batch.
         except (FileNotFoundError, ValueError, TypeError) as error:
            results.append({
               "image": str(image_path),
               "status": "error",
               "error": str(error)
            })

      return results

   # Convert YOLO detection results into an annotated image containing bounding boxes and labels.
   def annotate(self, result: Results) -> np.ndarray:
      annotated_image = result.plot()

      return annotated_image

   # Save the annotated image to the specified output directory.
   def save_prediction(
         self,
         annotated_image: np.ndarray,
         image_path: Path,
         output_dir: str | Path
   ) -> Path:

      output_dir = Path(output_dir)

      # Create an output filename based on the original image name.
      output_path = output_dir / f"{image_path.stem}_prediction.jpg"

      # Create the output directory and any missing parent directories if necessary.
      output_path.parent.mkdir(parents=True, exist_ok=True)

      # Convert the annotated NumPy array from BGR to RGB before saving with Pillow.
      Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

      return output_path
