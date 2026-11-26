from sqlmodel import SQLModel, Field
from enum import Enum
from datetime import datetime, timezone


# Define the possible processing states for a prediction.
# Inheriting from str allows the enum values to be used as strings.
class PredictionStatus(str, Enum):
   PENDING = "pending"
   PROCESSING = "processing"
   COMPLETED = "completed"


# Define the database table used to store information about each image prediction.
class Prediction(SQLModel, table=True):

   # Automatically generated primary key identifying each prediction.
   id: int | None = Field(default=None, primary_key=True)

   # Store the original filename supplied by the user.
   filename: str

   # Track the current processing status of the prediction.
   # New predictions default to PENDING.
   status: PredictionStatus = Field(default=PredictionStatus.PENDING)

   # Record the date and time when the prediction record is created.
   # Uses UTC to provide a consistent timestamp regardless of local timezone.
   created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

   # Store the file path of the original uploaded image.
   original_image_path: str

   # Store the file path of the annotated image once it has been generated.
   # Optional because an annotated image may not yet be available.
   annotated_image_path: str | None = None


# Define the database table used to store individual object detections.
class Detection(SQLModel, table=True):

   # Automatically generated primary key identifying each detection.
   id: int | None = Field(default=None, primary_key=True)

   # Foreign key linking this detection to its parent prediction record.
   # Multiple detections can reference the same prediction ID.
   prediction_id: int = Field(foreign_key="prediction.id")

   # Store the detected object's class name and confidence score.
   class_name: str
   confidence: float

   # Store the bounding box coordinates of the detected object.
   # Coordinates represent the top-left (x1, y1) and bottom-right (x2, y2) corners.
   x1: float
   y1: float
   x2: float
   y2: float
