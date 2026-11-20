import time
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent


app = FastAPI(
    title="Microservicio de Inferencia Iris",
    description="API REST contenerizada para predicción de especies de la flor Iris.",
    version="1.0.0",
)


try:
    artifact = joblib.load(BASE_DIR / "iris_model.pkl")
    model = artifact["model"]
    target_names = artifact["target_names"]
except Exception as e:
    raise RuntimeError(f"Error al cargar el archivo del modelo: {e}")


model.predict_proba(np.array([[5.1, 3.5, 1.4, 0.2]]))



class IrisRequest(BaseModel):
    sepal_length: float = Field(..., gt=0, le=15, examples=[5.1], description="Longitud del sépalo en cm")
    sepal_width: float = Field(..., gt=0, le=15, examples=[3.5], description="Ancho del sépalo en cm")
    petal_length: float = Field(..., gt=0, le=15, examples=[1.4], description="Longitud del pétalo en cm")
    petal_width: float = Field(..., gt=0, le=15, examples=[0.2], description="Ancho del pétalo en cm")



@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.3f}"
    return response



@app.get("/", include_in_schema=False)
def web_page():
    return FileResponse(BASE_DIR / "static" / "index.html")



@app.get("/health")
def read_root():
    return {
        "status": "online",
        "service": "Iris Classifier Microservice",
        "dataset_info": "150 instancias, 4 características, 3 clases",
    }



@app.post("/predict")
def predict_species(request: IrisRequest):
    try:
        start = time.perf_counter()

        
        features = np.array([[
            request.sepal_length,
            request.sepal_width,
            request.petal_length,
            request.petal_width,
        ]])

    
        probabilities = model.predict_proba(features)[0]
        prediction_id = int(np.argmax(probabilities))
        predicted_class = target_names[prediction_id]
        confidence = float(probabilities[prediction_id])

        return {
            "prediction": predicted_class,
            "class_id": prediction_id,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(prob), 4)
                for name, prob in zip(target_names, probabilities)
            },
            "inference_ms": round((time.perf_counter() - start) * 1000, 3),
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Error en la inferencia: {str(err)}")
