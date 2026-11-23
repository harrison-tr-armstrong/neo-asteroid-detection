from fastapi import FastAPI, UploadFile, File, HTTPException
from api.schemas import PredictionResponse, BoundingBox, DetectionResponse
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os
import tempfile
from contextlib import asynccontextmanager
from src.inference import NEOInference

@asynccontextmanager
async def lifespan(app: FastAPI):
   print("Loading NEO model...")

   app.state.detector = NEOInference()

   print("NEO model loaded")

   yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
   return {"message": "Hello World"}


@app.get("/health")
async def health():
   return {"status": "healthy"}


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


@app.post("/predictions", response_model=PredictionResponse)
async def create_prediction(file: UploadFile = File(...)):
   allowed_extensions = [".jpg", ".jpeg", ".png"]

   extension = os.path.splitext(file.filename)[1].lower()

   if extension not in allowed_extensions:
      raise HTTPException(status_code=400, detail="Unsupported file type")

   contents = await file.read()
   max_file_size = 10 * 1024 * 1024

   if len(contents) > max_file_size:
      raise HTTPException(status_code=413, detail="File too large")

   try:
      with Image.open(BytesIO(contents)) as im:
         im.verify()

   except UnidentifiedImageError:
      raise HTTPException(status_code=400, detail="Invalid image file")

   tf = tempfile.NamedTemporaryFile(suffix=extension, delete=False)
   tf.write(contents)
   tf.close()
   temp_file = tf.name

   try:
      result = app.state.detector.predict(temp_file)

   finally:
      os.unlink(temp_file)

   detection_responses = []

   for detection_data in result["detections"]:
      class_name = detection_data["class"]
      confidence = detection_data["confidence"]
      bounding_box = detection_data["bounding_box"]

      detection_response = DetectionResponse(
         class_name=class_name,
         confidence=confidence,
         bounding_box=bounding_box
      )

      detection_responses.append(detection_response)

   print(detection_responses)

   prediction_response = PredictionResponse(
      prediction_id = 1,
      filename = file.filename,
      status = "completed",
      detections = detection_responses
   )

   return prediction_response


