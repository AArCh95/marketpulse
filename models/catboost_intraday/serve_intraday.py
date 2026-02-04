#!/usr/bin/env python3
# D:\AARCH\models\catboost_intraday\serve.py
# FastAPI service for 5-minute CatBoost predictions

import json
from typing import List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from catboost import CatBoostClassifier

from config_intraday import (
    MODEL_PATH, FEATURE_ORDER_PATH, CLASS_NAMES, 
    MARGIN_CUTOFF, PROB_GAP, SHORT_MIN_PROB, CAT_FEATURES
)
from features_intraday import rows_to_frame

# ---------- Load model ----------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun train_intraday.py first.")

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))
print(f"[LOAD] Model loaded: {MODEL_PATH}")

# Load feature order
if FEATURE_ORDER_PATH.exists():
    with open(FEATURE_ORDER_PATH) as f:
        EXPECTED_FEATURES = json.load(f)
else:
    EXPECTED_FEATURES = None
    print("[WARN] Feature order file not found, using model's internal order.")

# ---------- API schemas ----------

class InferRow(BaseModel):
    """Single stock inference request."""
    symbol: str = Field(..., description="Stock ticker")
    ts: str = Field(..., description="UTC timestamp (ISO format)")

class InferRequest(BaseModel):
    """Batch inference request."""
    rows: List[InferRow] = Field(..., description="List of stocks to score")

class PredictionOutput(BaseModel):
    """Single prediction result."""
    symbol: str
    ts: str
    decision: str  # 'long', 'short', 'hold'
    probs: Dict[str, float]  # {p_up, p_down, p_flat}
    margin: float  # Max prob - second max
    reasoning: str  # Human-readable explanation

# ---------- Decision logic ----------

def make_decision(probs: np.ndarray) -> tuple[str, float, str]:
    """
    Convert model probabilities to trading decision.
    
    Returns: (decision, margin, reasoning)
    """
    p_down, p_flat, p_up = probs[0], probs[1], probs[2]
    
    # Margin = highest prob - second highest
    sorted_probs = sorted([p_up, p_down, p_flat], reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    
    # Decision rules
    if margin < MARGIN_CUTOFF:
        return "hold", margin, f"Low conviction (margin={margin:.3f} < {MARGIN_CUTOFF})"
    
    if p_up - p_down < PROB_GAP and p_down - p_up < PROB_GAP:
        return "hold", margin, f"Probs too close (gap={abs(p_up - p_down):.3f} < {PROB_GAP})"
    
    if p_up > p_down and p_up > p_flat:
        if margin >= MARGIN_CUTOFF:
            return "long", margin, f"Bullish signal (p_up={p_up:.3f}, margin={margin:.3f})"
        else:
            return "hold", margin, f"Bullish but weak margin"
    
    if p_down > p_up and p_down > p_flat:
        if p_down >= SHORT_MIN_PROB and margin >= MARGIN_CUTOFF:
            return "short", margin, f"Bearish signal (p_down={p_down:.3f}, margin={margin:.3f})"
        else:
            return "hold", margin, f"Bearish but insufficient confidence"
    
    return "hold", margin, f"Flat dominates (p_flat={p_flat:.3f})"


# ---------- API endpoints ----------

app = FastAPI(
    title="CatBoost Intraday 5m Predictor",
    description="5-minute forward return predictions for high-frequency trading",
    version="1.0.0"
)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_loaded": model is not None,
        "features": len(EXPECTED_FEATURES) if EXPECTED_FEATURES else "unknown"
    }

@app.post("/predict", response_model=List[PredictionOutput])
def predict(request: InferRequest) -> List[PredictionOutput]:
    """
    Batch inference endpoint.
    
    Input: List of {symbol, ts} dicts
    Output: List of predictions with decisions and probabilities
    """
    if not request.rows:
        raise HTTPException(status_code=400, detail="Empty request")
    
    # Convert to feature frame
    rows_dict = [r.dict() for r in request.rows]
    
    try:
        X = rows_to_frame(rows_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature hydration failed: {e}")
    
    if X.empty:
        raise HTTPException(status_code=500, detail="No valid features produced")
    
    # Predict probabilities
    try:
        probs = model.predict_proba(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    # Build outputs
    results = []
    for i, row in enumerate(request.rows):
        p = probs[i]
        decision, margin, reasoning = make_decision(p)
        
        results.append(PredictionOutput(
            symbol=row.symbol,
            ts=row.ts,
            decision=decision,
            probs={
                "p_up": float(p[2]),
                "p_down": float(p[0]),
                "p_flat": float(p[1]),
            },
            margin=float(margin),
            reasoning=reasoning
        ))
    
    return results

@app.post("/score_batch")
def score_batch(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    N8N-compatible batch scoring endpoint.
    
    Accepts n8n format:
      [
        {"mode": "infer", "schema_version": 1, "rows": [{"symbol": "AAPL", ...}]},
        {"mode": "infer", "schema_version": 1, "rows": [{"symbol": "MSFT", ...}]}
      ]
    
    Or simplified format:
      {"rows": [{"symbol": "AAPL", "ts": "..."}, ...]}
    
    Returns: {rows: [{symbol, prediction: {up, down, flat}, decision, ...}, ...]}
    """
    # Normalize input - handle n8n's array-of-jobs format
    rows = []
    
    if isinstance(payload, list):
        # N8N format: array of job objects
        for job in payload:
            if isinstance(job, dict) and "rows" in job:
                rows.extend(job["rows"])
            elif isinstance(job, dict):
                # Direct row object
                rows.append(job)
    elif isinstance(payload, dict):
        if "rows" in payload:
            # Simplified format: {rows: [...]}
            rows = payload["rows"]
        elif "payload" in payload:
            # Another common format: {payload: [...]}
            return score_batch(payload["payload"])
        else:
            # Single row?
            rows = [payload]
    else:
        raise HTTPException(status_code=400, detail="Invalid payload format")
    
    if not rows:
        return {"rows": []}
    
    # Convert to feature frame
    try:
        X = rows_to_frame(rows)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature hydration failed: {e}")
    
    # Predict
    try:
        probs = model.predict_proba(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    
    # Build n8n-compatible output
    output_rows = []
    for i, input_row in enumerate(rows):
        p = probs[i]
        decision, margin, reasoning = make_decision(p)
        
        output_rows.append({
            "symbol": input_row.get("symbol"),
            "ts": input_row.get("ts") or input_row.get("asof_utc"),
            "prediction": {
                "up": float(p[2]),
                "down": float(p[0]),
                "flat": float(p[1]),
            },
            "decision": decision,
            "margin": float(margin),
            "reasoning": reasoning,
            # Pass through input features for debugging
            "hydrated": {k: v for k, v in input_row.items() if k not in ["rows"]},
        })
    
    return {"rows": output_rows}

# ---------- Run server ----------

if __name__ == "__main__":
    import uvicorn
    
    # Read port from env or default to 8001 (different from daily model's 8000)
    import os
    port = int(os.getenv("PORT", 8001))
    
    print(f"\n{'='*60}")
    print(f"Starting CatBoost Intraday 5m API on port {port}")
    print(f"Model: {MODEL_PATH}")
    print(f"Features: {len(EXPECTED_FEATURES) if EXPECTED_FEATURES else 'N/A'}")
    print(f"{'='*60}\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
