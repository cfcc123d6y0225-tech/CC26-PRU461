from fastapi import FastAPI
from pydantic import BaseModel

from predictor import predict_major

app = FastAPI()


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def root():
    return {"message": "AI Service Running"}

@app.get("/health")
def health():
    return {"status": "ok", "service": "fastapi"}


@app.post("/predict")
def predict(request: PredictionRequest):

    result = predict_major(request.features)

    return result
