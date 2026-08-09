from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import time

app = FastAPI()

class ModelRegistry:
    def __init__(self, models_dir="."):
        self.models_dir = models_dir
        self.current_model = None
        self.version = "v1"
        self.load_model()
        
    def load_model(self, version="v1"):
        path = os.path.join(self.models_dir, f"model_{version}.joblib")
        if os.path.exists(path):
            self.current_model = joblib.load(path)
            self.version = version
            return True
        return False
        
    def get_current(self):
        return self.current_model
        
    def list_versions(self):
        return [f.split('_')[1].split('.')[0] for f in os.listdir(self.models_dir) if f.startswith('model_') and f.endswith('.joblib')]

registry = ModelRegistry()
start_time = time.time()

class PredictRequest(BaseModel):
    features: list[float]

class BatchPredictRequest(BaseModel):
    features_list: list[list[float]]

@app.post("/predict")
def predict(req: PredictRequest):
    model = registry.get_current()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        pred = model.predict([req.features])[0]
        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    model = registry.get_current()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        preds = model.predict(req.features_list)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "version": registry.version, "uptime": time.time() - start_time}

@app.get("/model/info")
def model_info():
    return {"version": registry.version, "type": str(type(registry.get_current()))}
