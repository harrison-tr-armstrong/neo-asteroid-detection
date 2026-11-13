from pydantic import BaseModel, ConfigDict

class BoundingBox(BaseModel):
   x1: float
   y1: float
   x2: float
   y2: float

class DetectionResponse(BaseModel):
   class_name: str
   confidence: float
   bounding_box: BoundingBox

class PredictionResponse(BaseModel):
   prediction_id: int
   filename: str
   status: str
   detections: list[DetectionResponse]
