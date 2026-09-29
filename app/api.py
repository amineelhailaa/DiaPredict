"""Infrastructure starter; implement Registry-backed prediction in project task 21."""
import os

from fastapi import FastAPI, HTTPException

app = FastAPI(title="DiaPredict API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": False}


@app.get("/ready")
def ready():
    raise HTTPException(status_code=503, detail="Train and register a model, then implement model loading.")


@app.post("/predict")
def predict():
    raise HTTPException(
        status_code=503,
        detail=f"Prediction is not implemented yet. Planned model: {os.getenv('MLFLOW_MODEL_URI', 'unset')}",
    )
