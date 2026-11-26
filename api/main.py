from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from api.schemas import PredictionResponse, BoundingBox, DetectionResponse
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
from contextlib import asynccontextmanager
from src.inference import NEOInference
from api.database import create_db_and_tables, get_session
from sqlmodel import Session
from api.models import PredictionStatus, Prediction, Detection
import uuid
import shutil


# Manage application startup and shutdown operations using FastAPI's lifespan.
@asynccontextmanager
async def lifespan(app: FastAPI):
   # Create the SQLite database tables before accepting requests.
   print("Initialising database...")
   create_db_and_tables()
   print("Database ready.")

   # Load the YOLO model once during startup and store it in the application state.
   # This allows API endpoints to reuse the same model instead of loading it for every request.
   print("Loading NEO model...")
   app.state.detector = NEOInference()
   print("NEO model loaded")

   yield


# Initialise the FastAPI application and register its lifespan handler.
app = FastAPI(lifespan=lifespan)


# Root endpoint used to check that the API is responding.
@app.get("/")
async def root():
   return {"message": "Hello World"}


# Health endpoint that returns the current API health status.
@app.get("/health")
async def health():
   return {"status": "healthy"}


# Example endpoint that returns mock prediction data using the defined response schema.
@app.get("/predictions/example", response_model=PredictionResponse)
async def prediction():
   return PredictionResponse(
      prediction_id=1,
      filename="asteroid.jpg",
      status="completed",
      detections=[
         DetectionResponse(
            class_name="streak",
            confidence=0.94,
            bounding_box=BoundingBox(
               x1=120.45,
               y1=80.72,
               x2=340.12,
               y2=160.88
            )
         )
      ]
   )


# Accept an uploaded image, run NEO detection and persist the prediction results.
@app.post("/predictions", response_model=PredictionResponse)
async def create_prediction(file: UploadFile = File(...), session: Session = Depends(get_session)):

   # Define the image formats accepted by the upload endpoint.
   allowed_extensions = [".jpg", ".jpeg", ".png"]

   # Extract and normalise the uploaded file extension.
   extension = os.path.splitext(file.filename)[1].lower()

   # Reject uploads that do not use a supported image format.
   if extension not in allowed_extensions:
      raise HTTPException(status_code=400, detail="Unsupported file type")

   # Read the uploaded file contents and enforce the maximum allowed file size.
   contents = await file.read()
   max_file_size = 10 * 1024 * 1024

   if len(contents) > max_file_size:
      raise HTTPException(status_code=413, detail="File too large")

   # Verify that the uploaded contents represent a valid image.
   # BytesIO allows Pillow to inspect the image directly from memory.
   try:
      with Image.open(BytesIO(contents)) as im:
         im.verify()

   except UnidentifiedImageError:
      raise HTTPException(status_code=400, detail="Invalid image file")

   # Generate a unique filename to prevent uploaded images from overwriting one another.
   unique_filename = f"{uuid.uuid4()}{extension}"
   original_image_path = os.path.join("storage", "originals", unique_filename)

   # Save the original uploaded image to persistent storage.
   with open(original_image_path, "wb") as image_file:
      image_file.write(contents)

   # Create a new prediction database record with an initial PENDING status.
   prediction = Prediction(
      filename=file.filename,
      status=PredictionStatus.PENDING,
      original_image_path=original_image_path
   )

   # Add the prediction to the database and commit it.
   # Refresh retrieves database-generated values, including the prediction ID.
   session.add(prediction)
   session.commit()
   session.refresh(prediction)

   # Update the prediction status to PROCESSING before starting model inference.
   prediction.status = PredictionStatus.PROCESSING
   session.commit()

   # Run the YOLO model against the saved original image.
   result = app.state.detector.predict(original_image_path)

   # Retrieve the generated annotated image path from the inference results.
   annotated_image_path = result["annotated_image"]

   # Generate a unique filename and destination path for the annotated image.
   annotated_filename = f"{uuid.uuid4()}_annotated.jpg"
   annotated_storage_path = os.path.join("storage", "annotated", annotated_filename)

   # Copy the annotated image into the dedicated storage directory.
   # The original inference output remains available in runs/inference/.
   shutil.copy2(annotated_image_path, annotated_storage_path)

   detection_responses = []

   # Process each detection returned by YOLO.
   for detection_data in result["detections"]:
      # Extract the class, confidence score and bounding box coordinates.
      class_name = detection_data["class"]
      confidence = detection_data["confidence"]
      bounding_box = detection_data["bounding_box"]

      # Create a database record for each detected object.
      # The prediction ID links each detection to its parent prediction.
      detection = Detection(
         prediction_id=prediction.id,
         class_name=class_name,
         confidence=confidence,
         x1=bounding_box["x1"],
         y1=bounding_box["y1"],
         x2=bounding_box["x2"],
         y2=bounding_box["y2"]
      )

      # Add the detection record to the current database session.
      session.add(detection)

      # Build the corresponding detection object for the API response.
      detection_response = DetectionResponse(
         class_name=class_name,
         confidence=confidence,
         bounding_box=bounding_box
      )

      detection_responses.append(detection_response)

   # Store the annotated image path and mark the prediction as completed.
   # The final commit saves the prediction updates and all added detection records.
   prediction.annotated_image_path = annotated_storage_path
   prediction.status = PredictionStatus.COMPLETED
   session.commit()

   # Print the detection results to the server console for debugging.
   print(detection_responses)

   # Construct the response returned to the client after successful processing.
   prediction_response = PredictionResponse(
      prediction_id = prediction.id,
      filename = file.filename,
      status = "completed",
      detections = detection_responses
   )

   return prediction_response
