from pathlib import Path
from ultralytics import YOLO
from PIL import Image, UnidentifiedImageError
import numpy as np
from ultralytics.engine.results import Results

class NEOInference:
   def __init__(self, model_path="models/neo_yolo11s_best.pt"):
      project_root = Path(__file__).resolve().parent.parent
      self.model_path = project_root / model_path

      if not self.model_path.exists():
         raise FileNotFoundError(
            f"Model checkpoint not found: {self.model_path}"
         )

      self.model = YOLO(str(self.model_path))

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
      valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

      if not image_path.exists():
         raise FileNotFoundError(
            f"Image not found: {image_path}"
         )

      if image_path.suffix.lower() not in valid_extensions:
         raise ValueError("Unsupported image format.")

      try:
         with Image.open(image_path) as image:
            image.verify()

      except (UnidentifiedImageError, OSError):
         raise ValueError("This file is not a valid image")

      if isinstance(conf, bool) or not isinstance(conf, (int, float)):
         raise TypeError("Confidence must be a number.")

      if isinstance(iou, bool) or not isinstance(iou, (int, float)):
         raise TypeError("IoU must be a number.")

      if not 0 <= conf <= 1:
         raise ValueError("Confidence must be between 0 and 1.")

      if not 0 <= iou <= 1:
         raise ValueError("IoU must be between 0 and 1.")

      if isinstance(imgsz, bool) or not isinstance(imgsz, int):
         raise TypeError("Image size must be an integer.")

      if imgsz <= 0:
         raise ValueError("Image size must be greater than 0.")

      results = self.model.predict(
         source=str(image_path),
         conf=conf,
         iou=iou,
         imgsz=imgsz,
         verbose=False
      )

      detections = []

      for result in results:
         for box in result.boxes:
            coordinates = box.xyxy[0].tolist()

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

      output_path = None

      if save_annotated:
         annotated_image = self.annotate(results[0])
         output_path = self.save_prediction(annotated_image, image_path, output_dir)

      return {
         "detections": detections,
         "annotated_image": str(output_path) if output_path else None
      }

   def predict_batch(
         self,
         image_paths: list[str | Path],
         conf: float = 0.10,
         iou: float = 0.40,
         imgsz: int = 640,
         save_annotated: bool = True,
         output_dir: str | Path = "runs/inference"
   ) -> list[dict]:

      if not isinstance(image_paths, list):
         raise TypeError("Image paths must be provided as a list.")

      if not image_paths:
         raise ValueError("At least one image path must be provided.")

      results = []

      for image_path in image_paths:
         try:
            if not isinstance(image_path, (str, Path)):
               raise TypeError("Each image path must be a string or Path object.")

            result = self.predict(
               image_path,
               conf=conf,
               iou=iou,
               imgsz=imgsz,
               save_annotated=save_annotated,
               output_dir=output_dir
            )

            result["image"] = str(image_path)
            result["status"] = "success"

            results.append(result)

         except (FileNotFoundError, ValueError, TypeError) as error:
            results.append({
               "image": str(image_path),
               "status": "error",
               "error": str(error)
            })

      return results

   def annotate(self, result: Results) -> np.ndarray:
      annotated_image = result.plot()

      return annotated_image

   def save_prediction(
         self,
         annotated_image: np.ndarray,
         image_path: Path,
         output_dir: str | Path
   ) -> Path:

      output_dir = Path(output_dir)

      output_path = output_dir / f"{image_path.stem}_prediction.jpg"
      output_path.parent.mkdir(parents=True, exist_ok=True)

      Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

      return output_path
