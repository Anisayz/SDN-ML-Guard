"""
api.py — FastAPI entrypoint for the ML Engine
===============================================
Start the server:
    python src/api.py
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict          — classify a single flow
    POST /predict/batch    — classify a list of flows
    GET  /health           — liveness check
    GET  /info             — model metadata
    GET  /docs             — auto-generated Swagger UI (FastAPI built-in)


    
The engine loads all models on first request (lazy load) or eagerly on
startup via the lifespan event. Both RF and AE run on every flow.

Verdict → action mapping for the Mitigation Engine:
    ATTACK  → block             (RF says attack, conf >= 0.70)
    SUSPECT → alert only        (RF says attack, conf < 0.70, AE disagrees)
    ANOMALY → alert + rate limit (AE flags, RF says benign)
    BENIGN  → pass through      (both agree it's clean)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# predict.py is in the same src/ directory
import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import get_engine, Verdict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Mitigation Engine URL — alert.py reads this from .env too
MITIGATION_URL = os.getenv("MITIGATION_URL", "http://localhost:9000/alert")


# ---------------------------------------------------------------------------
# Lifespan — load models at startup, not on first request
# This avoids a slow first request when the server starts
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("ML Engine API starting up — loading models ...")
    t0 = time.time()
    get_engine()   # triggers MLEngine.load() — loads classifier + AE
    log.info(f"Models loaded in {time.time() - t0:.1f}s — ready to serve")
    yield
    log.info("ML Engine API shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ML Engine — Intrusion Detection API",
    description=(
        "Real-time network flow classification using Random Forest + "
        "Autoencoder anomaly detection. "
        "Trained on CIC-IDS2018 (15 classes)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class FlowRequest(BaseModel):
    """
    A single network flow represented as a flat dict of feature values.
    Keys must match feature_list.json. Missing keys default to 0.
    """
    features: dict[str, float] = Field(
        ...,
        description="Flow features keyed by CICFlowMeter feature name",
        example={
            "Dst Port": 443,
            "Protocol": 6,
            "Flow Duration": 1500000,
            "Tot Fwd Pkts": 5,
            "Tot Bwd Pkts": 4,
        }
    )
    flow_id: str | None = Field(
        default=None,
        description="Optional flow identifier for correlation (not used in inference)"
    )


class BatchFlowRequest(BaseModel):
    """A list of flows for batch inference."""
    flows: list[FlowRequest] = Field(..., min_length=1, max_length=1000)


class VerdictResponse(BaseModel):
    """Inference result for a single flow."""
    flow_id: str | None
    label: str
    confidence: float
    is_attack: bool
    anomaly_score: float | None
    anomaly_flagged: bool | None
    verdict: str        # ATTACK | SUSPECT | ANOMALY | BENIGN
    source: str         # RF | RF+AE | AE
    action: str         # block | alert | rate_limit+alert | pass
    class_probabilities: dict[str, float]
    inference_ms: float


class BatchVerdictResponse(BaseModel):
    results: list[VerdictResponse]
    total: int
    attack_count: int
    suspect_count: int
    anomaly_count: int
    benign_count: int
    total_inference_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    n_features: int
    n_classes: int


class InfoResponse(BaseModel):
    model_type: str
    anomaly_type: str
    n_features: int
    classes: list[str]
    ae_threshold: float
    rf_conf_high: float
    verdict_actions: dict[str, str]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
VERDICT_ACTIONS = {
    "ATTACK" : "block",
    "SUSPECT": "alert",
    "ANOMALY": "rate_limit+alert",
    "BENIGN" : "pass",
}


def _verdict_to_response(v: Verdict, flow_id: str | None, inference_ms: float) -> VerdictResponse:
    return VerdictResponse(
        flow_id          = flow_id,
        label            = v.label,
        confidence       = round(v.confidence, 4),
        is_attack        = v.is_attack,
        anomaly_score    = round(v.anomaly_score, 6) if v.anomaly_score is not None else None,
        anomaly_flagged  = v.anomaly_flagged,
        verdict          = v.verdict,
        source           = v.source,
        action           = VERDICT_ACTIONS.get(v.verdict, "pass"),
        class_probabilities = {k: round(val, 4) for k, val in v.class_probabilities.items()},
        inference_ms     = round(inference_ms, 2),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — returns 200 if models are loaded and ready."""
    engine = get_engine()
    return HealthResponse(
        status        = "ok",
        models_loaded = engine._loaded,
        n_features    = len(engine.feature_list),
        n_classes     = len(engine.le.classes_),
    )


@app.get("/info", response_model=InfoResponse, tags=["System"])
async def info():
    """Model metadata — classes, thresholds, verdict-to-action mapping."""
    engine = get_engine()
    import os
    return InfoResponse(
        model_type    = "RandomForestClassifier",
        anomaly_type  = "Autoencoder (reconstruction error)",
        n_features    = len(engine.feature_list),
        classes       = engine.le.classes_.tolist(),
        ae_threshold  = float(os.getenv("AE_THRESHOLD", "0.003863")),
        rf_conf_high  = float(os.getenv("RF_CONF_HIGH",  "0.70")),
        verdict_actions = VERDICT_ACTIONS,
    )


@app.post("/predict", response_model=VerdictResponse, tags=["Inference"])
async def predict(request: FlowRequest):
    """
    Classify a single network flow.

    Returns a verdict with label, confidence, anomaly score, and
    recommended action for the Mitigation Engine.
    """
    engine = get_engine()
    t0     = time.perf_counter()

    try:
        verdict = engine.predict(request.features)
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    inference_ms = (time.perf_counter() - t0) * 1000
    log.info(
        f"[predict] verdict={verdict.verdict:7s}  label={verdict.label:30s}  "
        f"conf={verdict.confidence:.3f}  ms={inference_ms:.1f}"
    )

    return _verdict_to_response(verdict, request.flow_id, inference_ms)


@app.post("/predict/batch", response_model=BatchVerdictResponse, tags=["Inference"])
async def predict_batch(request: BatchFlowRequest):
    """
    Classify a batch of flows in a single call.
    More efficient than calling /predict N times.
    Maximum 1000 flows per request.
    """
    engine = get_engine()
    t0     = time.perf_counter()

    flow_dicts = [f.features for f in request.flows]
    flow_ids   = [f.flow_id  for f in request.flows]

    try:
        verdicts = engine.predict_batch(flow_dicts)
    except Exception as e:
        log.error(f"Batch inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {e}")

    total_ms = (time.perf_counter() - t0) * 1000
    per_ms   = total_ms / len(verdicts) if verdicts else 0

    results = [
        _verdict_to_response(v, flow_ids[i], per_ms)
        for i, v in enumerate(verdicts)
    ]

    counts = {v: sum(1 for r in results if r.verdict == v)
              for v in ("ATTACK", "SUSPECT", "ANOMALY", "BENIGN")}

    log.info(
        f"[batch] n={len(verdicts)}  "
        f"ATTACK={counts['ATTACK']}  SUSPECT={counts['SUSPECT']}  "
        f"ANOMALY={counts['ANOMALY']}  BENIGN={counts['BENIGN']}  "
        f"total_ms={total_ms:.1f}"
    )

    return BatchVerdictResponse(
        results           = results,
        total             = len(results),
        attack_count      = counts["ATTACK"],
        suspect_count     = counts["SUSPECT"],
        anomaly_count     = counts["ANOMALY"],
        benign_count      = counts["BENIGN"],
        total_inference_ms= round(total_ms, 2),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting ML Engine API on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "api:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )