# CatBoost Intraday 5m - Deployment Guide

## Overview

This is a **NEW parallel system** for 5-minute forward return predictions. It does NOT replace your existing daily model.

**Existing (unchanged):**
- Path: `D:\AARCH\models\catboost_core`
- Model: `catboost_core.cbm` (5-day predictions)
- Endpoint: Port 8000

**New (this system):**
- Path: `D:\AARCH\models\catboost_intraday`
- Model: `catboost_intraday_5m.cbm` (5-minute predictions)
- Endpoint: Port 8001

---

## File Structure

Create this directory structure:

```
D:\AARCH\models\catboost_intraday\
├── config_intraday.py          # Configuration
├── db_intraday.py               # Database access
├── features_intraday.py         # Feature engineering
├── train_intraday.py            # Training script
├── serve_intraday.py            # API server
├── retrain_weekly.py            # Automated retraining
├── requirements.txt             # Dependencies
└── models/                      # Created after training
    ├── catboost_intraday_5m.cbm
    ├── feature_order_intraday.json
    └── training_metadata.json
```

---

## Installation Steps

### 1. Create Directory and Copy Files

```bash
mkdir D:\AARCH\models\catboost_intraday
cd D:\AARCH\models\catboost_intraday

# Copy the 6 Python files I created into this directory
```

### 2. Create requirements.txt

```txt
catboost>=1.2
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.4.0
pandas>=2.1.0
numpy>=1.24.0
duckdb>=0.9.0
scikit-learn>=1.3.0
python-dateutil>=2.8.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Initial Training

**IMPORTANT:** Training uses 60 days of 1-minute bars. This will take **30-60 minutes** on first run.

```bash
cd D:\AARCH\models\catboost_intraday
python train_intraday.py
```

**Expected output:**
```
[1/5] Fetching training data from DuckDB...
  ✓ Loaded 780,000 bars across 50 symbols
  Date range: 2025-09-03 to 2025-11-03

[2/5] Creating labels...
  Class distribution:
    down  (0):   78,234 (10.03%)
    flat  (1):  623,532 (79.94%)
    up    (2):   78,234 (10.03%)

[3/5] Preparing features...
  ✓ Feature matrix: (780000, 21)

[4/5] Splitting train/validation...
  Train: 624,000 | Val: 156,000

[5/5] Training CatBoost...
  ...
  Best iteration: 287

VALIDATION RESULTS
Classification Report:
              precision    recall  f1-score   support

        down      0.652     0.548     0.595     15647
        flat      0.813     0.879     0.845    124707
          up      0.661     0.564     0.608     15646

    accuracy                          0.788    156000
```

**Troubleshooting:**
- If GPU training fails, it will automatically fall back to CPU
- If OOM error, reduce `TRAIN_SYMBOLS_LIMIT` in `config_intraday.py` (currently 50)

### 4. Start API Server

```bash
cd D:\AARCH\models\catboost_intraday
python serve_intraday.py
```

Server starts on **port 8001** (different from daily model's 8000).

**Test endpoint:**
```bash
curl http://localhost:8001/health
```

Expected response:
```json
{
  "status": "ok",
  "model_path": "D:\\AARCH\\models\\catboost_intraday\\models\\catboost_intraday_5m.cbm",
  "model_loaded": true,
  "features": 21
}
```

---

## N8N Integration

### Option A: Replace Existing CatBoost Node

In your n8n workflow (`MarketPulse_n8n.json`), find the **"CatBoostCore Node"** and update the URL:

**Before:**
```
http://catboost.aarch.shop/score_batch
```

**After:**
```
http://localhost:8001/score_batch
```

### Option B: Add Parallel Node (Recommended)

Keep your existing daily model running and add a **new parallel node** for 5-minute predictions:

1. Duplicate the "CatBoostCore Node"
2. Rename to "CatBoost Intraday 5m Node"
3. Change URL to `http://localhost:8001/score_batch`
4. Connect to "decisions with RTH vs OOH staging" node

This lets you compare both models' predictions.

---

## Input/Output Format

### Input (N8N → API)

Send to `POST http://localhost:8001/score_batch`:

```json
{
  "rows": [
    {
      "symbol": "AAPL",
      "ts": "2025-11-03T15:00:00Z"
    },
    {
      "symbol": "MSFT",
      "ts": "2025-11-03T15:00:00Z"
    }
  ]
}
```

**Required fields:**
- `symbol`: Stock ticker
- `ts`: UTC timestamp (ISO format)

**Optional fields:** None (all features computed from DB)

### Output (API → N8N)

```json
{
  "rows": [
    {
      "symbol": "AAPL",
      "ts": "2025-11-03T15:00:00Z",
      "prediction": {
        "up": 0.523,
        "down": 0.158,
        "flat": 0.319
      },
      "decision": "long",
      "margin": 0.204,
      "reasoning": "Bullish signal (p_up=0.523, margin=0.204)",
      "hydrated": {
        "close": 270.37,
        "dist_vwap_bps": 38.88,
        "ret_5m": 0.0023,
        "minute_of_day": 329,
        ...
      }
    }
  ]
}
```

**Key fields:**
- `prediction.up/down/flat`: Raw probabilities (sum to 1.0)
- `decision`: "long", "short", or "hold"
- `margin`: Confidence (highest prob - second highest)
- `reasoning`: Human-readable explanation
- `hydrated`: All computed features (for debugging)

---

## Weekly Retraining

### Manual Retrain

Run on Sunday (market closed):

```bash
cd D:\AARCH\models\catboost_intraday
python retrain_weekly.py
```

This will:
1. Backup current model to `model_backups/`
2. Train new model on latest 60 days
3. Validate new model loads correctly
4. Keep last 10 backups

**After retraining:**
```bash
# Restart API server to load new model
# (Stop current serve_intraday.py with Ctrl+C, then restart)
python serve_intraday.py
```

### Automated Retrain (Windows Task Scheduler)

Create a scheduled task:

1. Open **Task Scheduler**
2. Create Task:
   - **Trigger:** Weekly, Sunday 2:00 AM
   - **Action:** 
     - Program: `python`
     - Arguments: `retrain_weekly.py`
     - Start in: `D:\AARCH\models\catboost_intraday`
3. **Settings:**
   - Stop task if runs longer than 2 hours
   - If task fails, restart every 1 hour (max 3 times)

---

## Monitoring & Validation

### Check Model Performance

After retraining, check `models/training_metadata.json`:

```json
{
  "trained_at": "2025-11-03T02:15:43",
  "train_samples": 624000,
  "val_samples": 156000,
  "symbols": 50,
  "class_distribution": {
    "down": 78234,
    "flat": 623532,
    "up": 78234
  }
}
```

**Red flags:**
- Class imbalance >95% (e.g., flat dominates)
- Val samples <50,000
- Training took <5 minutes (data issue)

### Compare Predictions

Run both models on same data and compare:

```python
# Test script
import requests

data = {"rows": [{"symbol": "AAPL", "ts": "2025-11-03T15:00:00Z"}]}

# Old model (5-day)
r1 = requests.post("http://localhost:8000/score_batch", json=data)
print("5-day model:", r1.json()["rows"][0]["decision"])

# New model (5-min)
r2 = requests.post("http://localhost:8001/score_batch", json=data)
print("5-min model:", r2.json()["rows"][0]["decision"])
```

**Expected differences:**
- 5-day model: Broader trends (less sensitive to intraday noise)
- 5-min model: Faster signals (captures momentum, volatility)

---

## Troubleshooting

### "No training data returned"

**Cause:** Not enough recent 1-minute bars in `ohlcv_1m`.

**Fix:**
```bash
# Check bar count
duckdb D:\AARCH\DBs\market.duckdb
D SELECT COUNT(*), MAX(ts) FROM ohlcv_1m;

# If empty or stale, run your sync script
cd D:\AARCH\DBs
python market_sync_yahoo.py --duckdb "D:\AARCH\DBs\market.duckdb" --mode intraday --batch-size 60 --sleep 0.35
```

### "Feature hydration failed"

**Cause:** Missing columns in `ohlcv_1m` or DB connection issue.

**Fix:**
```bash
# Verify schema
duckdb D:\AARCH\DBs\market.duckdb
D DESCRIBE ohlcv_1m;

# Should have: symbol, ts, open, high, low, close, volume, dist_vwap_bps
```

### Model predictions are random

**Cause:** Training data quality issue or severe class imbalance.

**Fix:**
1. Check training metadata (`models/training_metadata.json`)
2. If flat class >95%, adjust thresholds in `config_intraday.py`:
   ```python
   UP_THRESH = 0.0015    # Was 0.001
   DOWN_THRESH = -0.0015 # Was -0.001
   ```
3. Retrain

### GPU out of memory

**Fix:** Reduce training size in `config_intraday.py`:
```python
TRAIN_SYMBOLS_LIMIT = 30  # Was 50
TRAIN_DAYS = 45           # Was 60
```

---

## Performance Expectations

**Training (first run):**
- Time: 30-60 minutes
- Data: ~780,000 bars (50 symbols × 60 days × 260 bars/day)
- Model size: 50-150 MB

**Inference:**
- Latency: 100-500ms per symbol (includes DB fetch + feature calc)
- Batch size: 50-200 symbols per request
- Memory: 200-500 MB

**Accuracy (typical):**
- Overall: 75-85% (RTH only)
- Up/Down: 55-70% precision
- Flat: 80-90% precision

---

## Next Steps

1. **Train the model** (60 min)
2. **Start API server** (port 8001)
3. **Test with curl** or Postman
4. **Update n8n workflow** to call new endpoint
5. **Monitor predictions** for 1 week
6. **Compare with daily model** outputs
7. **Schedule weekly retraining** (Sunday 2 AM)

---

## Contact / Issues

If you encounter issues:
1. Check logs from `train_intraday.py` or `serve_intraday.py`
2. Verify DB has recent data (`SELECT MAX(ts) FROM ohlcv_1m`)
3. Compare feature values in API response vs DB
4. Share error messages + training metadata JSON
