from pathlib import Path
from ultralytics import YOLO
from PIL import Image

class NEOInference:
   def __init__(self, model_path="models/neo_yolo11s_best.pt"):
      project_root = Path(__file__).resolve().parent.parent
      self.model_path = project_root / model_path

      if not self.model_path.exists():
         raise FileNotFoundError(
            f"Model checkpoint not found: {self.model_path}"
         )

      self.model = YOLO(str(self.model_path))

   def predict(self, image_path, conf=0.10, iou=0.40):
      image_path = Path(image_path)

      if not image_path.exists():
         raise FileNotFoundError(
            f"Image not found: {image_path}"
         )

      results = self.model.predict(
         source=str(image_path),
         conf=conf,
         iou=iou,
         imgsz=640,
         verbose=False
      )

      annotated_image = results[0].plot()

      output_path = Path("runs/inference") / f"{image_path.stem}_prediction.jpg"
      output_path.parent.mkdir(parents=True, exist_ok=True)

      Image.fromarray(annotated_image[:, :, ::-1]).save(output_path)

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

      return detections
