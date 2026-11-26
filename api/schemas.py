from pydantic import BaseModel


# Define the structure used to represent an object's bounding box coordinates.
class BoundingBox(BaseModel):
   x1: float
   y1: float
   x2: float
   y2: float


# Define the structure of an individual detection returned by the API.
class DetectionResponse(BaseModel):
   class_name: str
   confidence: float
   bounding_box: BoundingBox


# Define the overall response structure returned after a prediction is processed.
class PredictionResponse(BaseModel):
   prediction_id: int
   filename: str
   status: str
   detections: list[DetectionResponse]
