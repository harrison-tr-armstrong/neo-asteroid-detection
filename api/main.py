from fastapi import FastAPI
from schemas import PredictionResponse, BoundingBox, DetectionResponse

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
