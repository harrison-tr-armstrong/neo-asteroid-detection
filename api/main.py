from fastapi import FastAPI, UploadFile, File, HTTPException
from schemas import PredictionResponse, BoundingBox, DetectionResponse
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import os

app = FastAPI()

@app.get("/")
async def root():
   return {"message": "Hello World"}


@app.get("/health")
async def health():
   return {"status": "healthy"}


@app.get("/prediction/example", response_model=PredictionResponse)
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


@app.post("/predictions")
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

   return {
      "filename": file.filename,
      "file_size": len(contents),
      "content_type": file.content_type
   }
