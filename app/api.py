"""Infrastructure starter; implement Registry-backed prediction in project task 21."""
import os

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from app.command.PredictionCommand import PredictionCMD
app = FastAPI(title="DiaPredict API", version="0.1.0")
MODEL_URI = "models:/diabetes-prediction-model@winner"
model = mlflow.pyfunc.load_model(MODEL_URI)
##model path

@app.get("/")
def welcome():
    return {
        "Title": "welcome to DiaPredict ",
        "Description": "this is a classification model that predict your situation based on your diabetic Data"
    }
@app.get("/health")
def health():
    return {"status": "ok", "model_ready": True}


@app.get("/ready")
def ready():
    raise HTTPException(status_code=503, detail="Train and register a model, then implement model loading.")


@app.post("/predict")
def predict(data: PredictionCMD):
    prediction  =  model.predict(pd.DataFrame([data.model_dump()]))
    return prediction



