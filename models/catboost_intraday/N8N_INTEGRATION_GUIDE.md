# N8N Integration Guide - CatBoost Intraday 5m

## Overview

This guide shows how to integrate the new 5-minute CatBoost model into your existing MarketPulse n8n workflow.

---

## Current Architecture (BEFORE)

```
n8n Workflow: MarketPulse
│
├─ Generate Stock Watchlist (200 symbols)
├─ Alpaca/Finnhub Stock Price (real-time)
├─ SET THRESHOLD (filter ≥2.5% moves)
├─ Gather News (Polygon, Finnhub)
├─ FinBERT Sentiment Analysis
├─ Prepare Data For Catboost
│   └─ Builds payload with:
│       - asof_utc, session, close_d, high_d, low_d
│       - ret_1d, ret_5d, beta_60d
│       - news features, sector/market context
│       - RSI, MACD (from DB)
│
└─ CatBoostCore Node (CURRENT MODEL)
    ├─ URL: http://catboost.aarch.shop/score_batch
    ├─ Predicts: 5-DAY forward returns
    ├─ Uses: Daily bars (ohlcv_1d)
    └─ Output: {up, down, flat} probabilities
```

**Problem:** Model trained on 5-day horizon but used for minute-scale decisions.

---

## New Architecture (AFTER)

```
n8n Workflow: MarketPulse
│
├─ [SAME] Generate Stock Watchlist
├─ [SAME] Alpaca/Finnhub Stock Price
├─ [SAME] SET THRESHOLD (≥2.5% filter)
│
├─ [NEW] Prepare Data For CatBoost Intraday
│   └─ Simplified payload:
│       - symbol (required)
│       - ts (UTC timestamp, required)
│       - All other features auto-computed from ohlcv_1m
│
└─ [NEW] CatBoost Intraday 5m Node
    ├─ URL: http://localhost:8001/score_batch
    ├─ Predicts: 5-MINUTE forward returns
    ├─ Uses: 1-minute bars (ohlcv_1m)
    └─ Output: {up, down, flat} + decision + margin
```

---

## Key Differences

| Aspect | Old Model (Daily) | New Model (5-Minute) |
|--------|------------------|----------------------|
| **Horizon** | 5 days | 5 minutes |
| **Data Source** | `ohlcv_1d` (daily bars) | `ohlcv_1m` (minute bars) |
| **Predictions** | Swing trading signals | High-frequency signals |
| **Update Frequency** | Once per day | Every minute |
| **Features** | Daily OHLC, RSI, MACD | Intraday: VWAP, realized vol, minute-of-day |
| **N8N Payload** | 25+ features | 2 features (symbol, ts) |
| **Latency** | ~50ms per symbol | ~200ms per symbol |
| **Training Time** | 1-2 minutes | 30-60 minutes |
| **Model Size** | 5 MB | 50-150 MB |

---

## Option 1: Replace Existing Node (Simplest)

**Use this if:** You want to fully transition to 5-minute predictions.

### Steps:

1. **Find the CatBoostCore Node** in your workflow
2. **Update URL**:
   - Before: `http://catboost.aarch.shop/score_batch`
   - After: `http://localhost:8001/score_batch`

3. **Simplify the input payload** (optional but recommended):
   
   **Before (complex - 25+ fields):**
   ```javascript
   // "Prepare Data For Catboost" node
   return {
       mode: "infer",
       schema_version: 1,
       rows: [{
           symbol: "AAPL",
           asof_utc: "2025-11-03T15:00:00Z",
           session: "2025-11-03",
           close_d: 270.37,
           ret_1d: 0.012,
           ret_5d: 0.034,
           high_d: 277.32,
           low_d: 269.16,
           beta_60d: 1.23,
           mkt_ret_1d: 0.005,
           sector_ret_1d: 0.008,
           // ... 15+ more fields
       }]
   };
   ```
   
   **After (simplified - 2 fields):**
   ```javascript
   // New "Prepare Data For Catboost Intraday" node
   return {
       rows: [{
           symbol: "AAPL",
           ts: new Date().toISOString()  // Current UTC time
       }]
   };
   ```

4. **Update downstream nodes** to use new response format:
   
   **Old response:**
   ```json
   {
     "rows": [{
       "symbol": "AAPL",
       "up": 0.45,
       "down": 0.15,
       "flat": 0.40
     }]
   }
   ```
   
   **New response:**
   ```json
   {
     "rows": [{
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
         ...
       }
     }]
   }
   ```

5. **Update "decisions with RTH vs OOH staging" node**:
   
   Change references from:
   ```javascript
   $json.up        → $json.prediction.up
   $json.down      → $json.prediction.down
   $json.flat      → $json.prediction.flat
   ```

---

## Option 2: Parallel Nodes (Recommended for Testing)

**Use this if:** You want to compare both models' predictions before fully switching.

### Steps:

1. **Duplicate the CatBoostCore node** → rename to "CatBoost Intraday 5m"
2. **Change URL** in new node to `http://localhost:8001/score_batch`
3. **Add a Merge node** after both models:
   ```
   ┌─ CatBoostCore (daily) ──┐
   │                          ├─→ Merge ─→ Compare Decisions
   └─ CatBoost Intraday 5m ──┘
   ```

4. **Create comparison node** (JavaScript):
   ```javascript
   // Compare daily vs 5-minute predictions
   const daily = $input.item(0).json;
   const intraday = $input.item(1).json;
   
   return {
       symbol: daily.symbol,
       daily_decision: daily.decision,
       intraday_decision: intraday.decision,
       agreement: daily.decision === intraday.decision,
       daily_margin: daily.margin,
       intraday_margin: intraday.margin,
       use_prediction: intraday.decision  // Prefer 5-minute for now
   };
   ```

5. **Route to existing "decisions with RTH vs OOH staging" node**

---

## Minimal N8N Node Configuration

### Input Node: "Prepare Intraday Payload"

**Type:** Code (JavaScript)

```javascript
// Simplest possible input for 5-minute model
const items = $input.all();

return items.map(item => ({
    json: {
        symbol: item.json.symbol,
        ts: new Date().toISOString()  // Current time
    }
}));
```

### API Call Node: "CatBoost Intraday 5m"

**Type:** HTTP Request

- **Method:** POST
- **URL:** `http://localhost:8001/score_batch`
- **Body:** `{{ $json }}`
- **Content Type:** `application/json`

### Output Processing: "Parse Predictions"

**Type:** Code (JavaScript)

```javascript
// Extract key fields for downstream use
const prediction = $json;

return {
    json: {
        symbol: prediction.symbol,
        ts: prediction.ts,
        
        // Probabilities (0-1 scale)
        p_up: prediction.prediction.up,
        p_down: prediction.prediction.down,
        p_flat: prediction.prediction.flat,
        
        // Decision (long/short/hold)
        decision: prediction.decision,
        
        // Confidence
        margin: prediction.margin,
        reasoning: prediction.reasoning,
        
        // Debug info
        features: prediction.hydrated
    }
};
```

---

## Testing the Integration

### 1. Test API Directly (Postman/cURL)

```bash
curl -X POST http://localhost:8001/score_batch \
  -H "Content-Type: application/json" \
  -d '{
    "rows": [
      {"symbol": "AAPL", "ts": "2025-11-03T15:00:00Z"},
      {"symbol": "MSFT", "ts": "2025-11-03T15:00:00Z"}
    ]
  }'
```

Expected response:
```json
{
  "rows": [
    {
      "symbol": "AAPL",
      "ts": "2025-11-03T15:00:00Z",
      "prediction": {"up": 0.52, "down": 0.16, "flat": 0.32},
      "decision": "long",
      "margin": 0.20,
      "reasoning": "Bullish signal (p_up=0.520, margin=0.200)"
    },
    ...
  ]
}
```

### 2. Test in N8N Workflow

1. **Disable** the "Send Market Alert Email" node (avoid spamming)
2. **Enable** the new CatBoost Intraday node
3. **Execute manually** on 5-10 stocks
4. **Check output**:
   - All symbols have predictions?
   - Decisions make sense (not all "hold")?
   - Margins are reasonable (0.1-0.5)?

### 3. Compare with Old Model (Side-by-Side)

Run both models on same input and check:

```javascript
// Comparison node
const daily = $json.daily_prediction;
const intraday = $json.intraday_prediction;

console.log(`Symbol: ${daily.symbol}`);
console.log(`Daily (5d):    ${daily.decision} (margin ${daily.margin.toFixed(3)})`);
console.log(`Intraday (5m): ${intraday.decision} (margin ${intraday.margin.toFixed(3)})`);

// Expect: Intraday has more "long"/"short" (less "hold") due to higher frequency
```

---

## Troubleshooting

### Error: "Connection refused" on port 8001

**Cause:** API server not running.

**Fix:**
```bash
cd D:\AARCH\models\catboost_intraday
python serve_intraday.py
```

### Error: "Feature hydration failed"

**Cause:** Missing data in `ohlcv_1m` for requested symbol/time.

**Fix:**
1. Check if symbol exists in DB:
   ```sql
   SELECT COUNT(*) FROM ohlcv_1m WHERE symbol = 'AAPL';
   ```
2. Run market sync if no data:
   ```bash
   python market_sync_yahoo.py --mode intraday
   ```

### All predictions are "hold"

**Cause:** Model confidence is low (margins < threshold).

**Possible reasons:**
1. **Training data quality issue** (check `training_metadata.json`)
2. **Market is flat** (low volatility day)
3. **Thresholds too strict** (adjust `MARGIN_CUTOFF` in `config_intraday.py`)

**Fix:**
```python
# config_intraday.py
MARGIN_CUTOFF = 0.10  # Lower from 0.15 (more signals)
PROB_GAP = 0.05       # Lower from 0.08 (more signals)
```

Retrain after changing thresholds.

### Predictions are very different from daily model

**This is expected!** The 5-minute model sees:
- Intraday momentum (ret_5m, rv_30m)
- VWAP distance (dist_vwap_bps)
- Time-of-day effects (minute_of_day_pct)

The daily model sees:
- Multi-day trends (ret_5d)
- Daily RSI/MACD
- Overnight gaps

**Both are correct for their respective horizons.**

---

## Performance Tuning

### Reduce Latency

If predictions are too slow (>500ms per symbol):

1. **Batch requests** (send 50-100 symbols at once):
   ```javascript
   // N8N: Aggregate items before API call
   const batch = $input.all().map(i => ({
       symbol: i.json.symbol,
       ts: new Date().toISOString()
   }));
   
   return [{ json: { rows: batch } }];
   ```

2. **Pre-compute features** (run `materialize_intraday.py` every minute):
   ```bash
   # Windows Task Scheduler: Every 1 minute during market hours
   python materialize_intraday.py
   ```
   Then update `features_intraday.py` to read from `features_intraday_1m` table.

3. **Increase DB connection pool** (if using multiple workers).

### Increase Signal Rate

If model produces too few "long"/"short" signals:

1. **Lower thresholds** (`config_intraday.py`):
   ```python
   UP_THRESH = 0.0005      # From 0.001
   DOWN_THRESH = -0.0005   # From -0.001
   ```

2. **Lower decision cutoffs** (`config_intraday.py`):
   ```python
   MARGIN_CUTOFF = 0.10    # From 0.15
   ```

3. **Retrain** with new config:
   ```bash
   python train_intraday.py
   ```

---

## Migration Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Copy files to `D:\AARCH\models\catboost_intraday\`
- [ ] Run validation script (`python validate_setup.py`)
- [ ] Train model (`python train_intraday.py` - wait 30-60 min)
- [ ] Start API server (`python serve_intraday.py`)
- [ ] Test API with curl/Postman
- [ ] Update n8n workflow (Option 1 or 2)
- [ ] Test n8n workflow on 5-10 stocks
- [ ] Compare predictions with old model (1 week)
- [ ] Deploy to production
- [ ] Schedule weekly retraining (Sunday 2 AM)

---

## Rollback Plan

If the new model doesn't work:

1. **Revert n8n workflow** to old CatBoostCore URL
2. **Stop intraday API** (`Ctrl+C` on `serve_intraday.py`)
3. **Keep files** in `D:\AARCH\models\catboost_intraday\` for future retry
4. **Report issues** with:
   - Training logs
   - API error messages
   - Sample predictions (expected vs actual)

---

## Next Steps After Deployment

1. **Monitor for 1 week** alongside old model
2. **Compare accuracy** (track predictions vs actual 5-minute returns)
3. **Tune thresholds** based on signal quality
4. **Increase training data** to 90 days if needed
5. **Add more symbols** to watchlist (currently limited to 50)
6. **Optimize feature engineering** (add more intraday indicators)

---

## Support

If you need help:
1. Run `python validate_setup.py` and share output
2. Check `serve_intraday.py` logs for errors
3. Share `models/training_metadata.json` contents
4. Provide sample prediction output vs expected behavior
