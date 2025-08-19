# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import numpy as np
import joblib
import logging

# ----- Logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("iris-api")

# ----- Pydantic Schemas -----
class PredictionInput(BaseModel):
    sepal_length: float = Field(..., ge=0, description="Sepal length in cm")
    sepal_width:  float = Field(..., ge=0, description="Sepal width in cm")
    petal_length: float = Field(..., ge=0, description="Petal length in cm")
    petal_width:  float = Field(..., ge=0, description="Petal width in cm")

class PredictionOutput(BaseModel):
    prediction: str
    confidence: Optional[float] = None
    class_probabilities: Optional[Dict[str, float]] = None

class BatchPredictionInput(BaseModel):
    instances: List[PredictionInput]

class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]

# ----- Lifespan (load model on startup) -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Loading model.pkl ...")
        bundle = joblib.load("model.pkl")
        app.state.pipeline = bundle["pipeline"]
        app.state.feature_names = bundle["feature_names"]
        app.state.target_names = bundle["target_names"]
        app.state.metadata = {
            k: v for k, v in bundle.items() if k not in ["pipeline"]
        }
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model.pkl. Did you run train_model.py?")
        raise

    yield  # shutdown
    # nothing to clean up

app = FastAPI(
    title="Iris ML Model API",
    description="FastAPI service for Iris flower species prediction",
    version="1.0.0",
    lifespan=lifespan
)

# ----- Routes -----
@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Iris ML Model API is running"}

@app.get("/model-info")
def model_info():
    md = app.state.metadata
    return {
        "model_type": md.get("model_type", "unknown"),
        "problem_type": md.get("problem_type", "classification"),
        "trained_at": md.get("trained_at"),
        "features": app.state.feature_names,
        "target_names": app.state.target_names,
        "test_metrics": md.get("test_metrics", {})
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        features = np.array([[
            input_data.sepal_length,
            input_data.sepal_width,
            input_data.petal_length,
            input_data.petal_width
        ]], dtype=float)

        pipeline = app.state.pipeline
        target_names = app.state.target_names

        pred_idx = int(pipeline.predict(features)[0])
        proba = pipeline.predict_proba(features)[0]  # shape (3,)
        confidence = float(np.max(proba))
        probs = {target_names[i]: float(proba[i]) for i in range(len(target_names))}
        return PredictionOutput(
            prediction=target_names[pred_idx],
            confidence=confidence,
            class_probabilities=probs
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchPredictionInput):
    try:
        pipeline = app.state.pipeline
        target_names = app.state.target_names

        X = np.array([[i.sepal_length, i.sepal_width, i.petal_length, i.petal_width]
                      for i in batch.instances], dtype=float)
        pred_idxs = pipeline.predict(X)
        probas = pipeline.predict_proba(X)

        outputs: List[PredictionOutput] = []
        for idx, p in zip(pred_idxs, probas):
            idx = int(idx)
            confidence = float(np.max(p))
            probs = {target_names[i]: float(p[i]) for i in range(len(target_names))}
            outputs.append(PredictionOutput(
                prediction=target_names[idx],
                confidence=confidence,
                class_probabilities=probs
            ))
        return BatchPredictionOutput(predictions=outputs)
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=400, detail=str(e))
