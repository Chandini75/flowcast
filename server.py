"""FlowCast API. Run with:  uvicorn server:app --reload --port 8000"""
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from forecaster import MAX_H, MODEL_PATH, Forecaster

app = FastAPI(title="FlowCast forecast API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_forecaster = None


def get_forecaster() -> Forecaster:
    global _forecaster
    if _forecaster is None:
        if not Path(MODEL_PATH).exists():
            raise HTTPException(503, "Model not trained yet. Run: python train.py --data data/uihc_ed_arrivals.csv")
        _forecaster = Forecaster()
    return _forecaster


class ForecastRequest(BaseModel):
    weekday: int = Field(..., ge=0, le=6)  # Monday = 0
    hour: int = Field(..., ge=0, le=23)  # last completed hour
    month: int = Field(10, ge=1, le=12)
    recent: List[float]  # hourly arrivals up to and including `hour`, oldest first
    horizons: int = Field(MAX_H, ge=1, le=MAX_H)


@app.get("/api/health")
def health():
    return {"ok": True, "model_trained": Path(MODEL_PATH).exists()}


@app.get("/api/metrics")
def metrics():
    return get_forecaster().metrics


@app.get("/api/profile")
def profile(weekday: int = 1):
    if not 0 <= weekday <= 6:
        raise HTTPException(422, "weekday must be 0 (Monday) to 6 (Sunday)")
    return {"weekday": weekday, "hourly": get_forecaster().profile(weekday)}


@app.post("/api/forecast")
def forecast(req: ForecastRequest):
    if not req.recent or len(req.recent) > 168:
        raise HTTPException(422, "recent must hold between 1 and 168 hourly counts")
    return get_forecaster().predict(req.weekday, req.hour, req.month, req.recent, req.horizons)


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
