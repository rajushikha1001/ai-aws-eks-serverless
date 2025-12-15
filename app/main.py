from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from app.predict import predict

app = FastAPI(title="AI Inference Service")

@app.post("/predict", response_model=PredictionResponse)
def run_prediction(request: PredictionRequest):
    result = predict(request.features)
    return PredictionResponse(prediction=result)
