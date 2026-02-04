# Quick Reference Card - CatBoost Intraday 5m

## 🚀 Deployment Steps (In Order)

### 1. Pre-Flight Check (2 minutes)
```bash
cat D:\AARCH\DBs\CURRENT_SNAPSHOT.txt
# Must show: D:\AARCH\DBs\market_snap_*.duckdb

cd D:\AARCH\models\catboost_intraday
python validate_setup.py
# Must show: ✓ All checks passed!
```

### 2. Train Model (30-60 minutes, ONE TIME)
```bash
cd D:\AARCH\models\catboost_intraday
python train_intraday.py
# Watch for: [DB] Using snapshot: ...
```

### 3. Start API (Runs Forever)
```bash
python serve_intraday.py
# API runs on: http://localhost:8001
```

### 4. Test API
```bash
curl http://localhost:8001/health
# Should return: {"status": "ok"}

curl -X POST http://localhost:8001/score_batch \
  -H "Content-Type: application/json" \
  -d '[{"mode":"infer","rows":[{"symbol":"AAPL","asof_utc":"2025-11-03T15:00:00Z"}]}]'
# Should return: {"rows": [{"symbol": "AAPL", "decision": "long/short/hold", ...}]}
```

### 5. Update N8N
**Option A (Simple):** Change URL in existing CatBoostCore node
- From: `http://catboost.aarch.shop/score_batch`
- To: `http://localhost:8001/score_batch`

**Option B (Safe):** Add new parallel node for 1 week comparison

---

## 📁 Files & Locations

**Deployment Directory:**
```
D:\AARCH\models\catboost_intraday\
├── config_intraday.py          # Config
├── db_intraday.py               # DB access (uses snapshot)
├── features_intraday.py         # Feature engineering
├── train_intraday.py            # Training script
├── serve_intraday.py            # API server (port 8001)
├── retrain_weekly.py            # Weekly retrain
├── requirements.txt             # Dependencies
└── validate_setup.py            # Pre-flight checks
```

**Database:**
```
D:\AARCH\DBs\
├── market.duckdb                # Live DB (written by sync script)
├── CURRENT_SNAPSHOT.txt         # Pointer to snapshot
└── market_snap_*.duckdb         # Read-only snapshot
```

---

## 🔧 Critical Commands

### Check Which DB is Being Used:
```python
python -c "from db_intraday import get_conn; conn = get_conn()"
# Should output: [DB] Using snapshot: D:\AARCH\DBs\market_snap_*.duckdb
# NOT: WARNING: Using primary DB
```

### Create New Snapshot (Before Weekly Retrain):
```bash
cd D:\AARCH\DBs
duckdb market.duckdb -c "EXPORT DATABASE 'market_snap_20251110_020000.duckdb'"
echo "D:\AARCH\DBs\market_snap_20251110_020000.duckdb" > CURRENT_SNAPSHOT.txt
```

### Weekly Retrain (Sunday 2 AM):
```bash
cd D:\AARCH\models\catboost_intraday
python retrain_weekly.py
# Then restart serve_intraday.py
```

---

## ⚠️ Troubleshooting

### "Connection refused" on port 8001
```bash
# Check if API is running
netstat -an | findstr 8001

# Start API if not running
cd D:\AARCH\models\catboost_intraday
python serve_intraday.py
```

### "No training data returned"
```bash
# Check ohlcv_1m has data
duckdb D:\AARCH\DBs\market_snap_*.duckdb
D SELECT COUNT(*), MAX(ts) FROM ohlcv_1m;
# Should show rows from last 24 hours

# If empty, run sync script
cd D:\AARCH\DBs
python market_sync_yahoo.py --mode intraday --batch-size 60 --sleep 0.35
```

### "Feature hydration failed"
```bash
# Check dist_vwap_bps is populated
duckdb D:\AARCH\DBs\market_snap_*.duckdb
D SELECT COUNT(*) FROM ohlcv_1m WHERE dist_vwap_bps IS NOT NULL;
# Should be > 0

# If not, run update script
cd D:\AARCH\DBs
python update_features.py
```

### All predictions are "hold"
```python
# Lower thresholds in config_intraday.py:
MARGIN_CUTOFF = 0.10  # From 0.15
PROB_GAP = 0.05       # From 0.08

# Then retrain
python train_intraday.py
```

---

## 📊 Expected Performance

**Training:**
- Time: 30-60 minutes (first run)
- Data: ~780,000 bars (60 days × 50 symbols)
- Model: 50-150 MB file

**Inference:**
- Latency: 100-500ms per symbol
- Batch: 50-200 symbols per request

**Accuracy:**
- Overall: 75-85%
- Up/Down: 55-70% precision
- Flat: 80-90% precision

---

## 🎯 What Model Uses vs Ignores

### ✅ USES (from ohlcv_1m):
- `close`, `high`, `low`, `open` (current bar)
- `dist_vwap_bps` (pre-computed by update_features.py)
- `volume`
- **Computed:** ret_5m, ret_15m, ret_30m
- **Computed:** rv_5m, rv_15m, rv_30m
- **Computed:** vol_ratio_5m
- **Computed:** minute_of_day, hour_of_day
- **Computed:** mkt_ret_5m (from SPY 1m bars)

### ❌ IGNORES (from n8n payload / ohlcv_1d):
- `close_d`, `high_d`, `low_d` (daily, too slow)
- `ret_1d`, `ret_5d` (daily returns, too slow)
- `rsi_14`, `macd_line` (daily indicators, too slow)
- `beta_60d` (not relevant for 5-minute)
- All other daily metrics from n8n

**Why?** Daily indicators update once per day. We need minute-level indicators that update every minute.

---

## 📅 Maintenance Schedule

**Daily (Automatic - Already Running):**
- `market_sync_yahoo.py --mode intraday` (populates ohlcv_1m)
- `market_sync_yahoo.py --mode daily` (populates ohlcv_1d, once per day)
- `update_features.py` (computes dist_vwap_bps)

**Weekly (Manual or Scheduled):**
1. Sunday 1:00 AM: Create new snapshot
2. Sunday 2:00 AM: Run `retrain_weekly.py`
3. Sunday 3:00 AM: Restart `serve_intraday.py`

**Monthly:**
- Review `models/training_metadata.json`
- Check class distribution (flat shouldn't be >95%)
- Tune thresholds if needed

---

## 🆘 Emergency Rollback

If new model doesn't work:

```bash
# 1. Stop new API
# (Ctrl+C on serve_intraday.py)

# 2. Revert n8n workflow
# Change URL back to: http://catboost.aarch.shop/score_batch

# 3. Keep files for investigation
# Don't delete D:\AARCH\models\catboost_intraday\
```

---

## 📞 Support Checklist

If you need help, provide:

1. **Validation output:**
   ```bash
   python validate_setup.py > validation_output.txt
   ```

2. **Training logs:**
   ```bash
   python train_intraday.py > training_logs.txt
   ```

3. **API logs:**
   ```bash
   # Copy console output from serve_intraday.py
   ```

4. **Sample prediction:**
   ```bash
   curl -X POST http://localhost:8001/score_batch \
     -H "Content-Type: application/json" \
     -d '[{"mode":"infer","rows":[{"symbol":"AAPL","asof_utc":"2025-11-03T15:00:00Z"}]}]' \
     > sample_prediction.json
   ```

5. **Training metadata:**
   ```bash
   cat models/training_metadata.json
   ```

---

## ✅ Success Criteria (After 1 Week)

Deploy to production if:
- ✅ Predictions vary (not all "hold")
- ✅ Margins are 0.1-0.5 range
- ✅ Decisions align with actual 5-minute moves
- ✅ API latency <500ms per symbol
- ✅ No crashes or errors in logs
- ✅ Snapshot pointer working correctly

Keep investigating if:
- ❌ >90% predictions are "hold"
- ❌ Margins always <0.05
- ❌ API errors or crashes
- ❌ Training fails

---

## 🎓 Key Concepts

**Snapshot Pointer System:**
- `CURRENT_SNAPSHOT.txt` → points to read-only snapshot
- Avoids lock conflicts with live `market.duckdb`
- Must update pointer before weekly retrain

**Why No Daily Data (ohlcv_1d)?**
- Daily RSI/MACD update once per day → too slow
- 5-minute model needs minute-level indicators
- Using daily data = temporal mismatch

**N8N Payload:**
- Sends 25+ fields (daily data)
- Model uses only 2: `symbol`, `ts`
- Recomputes everything from ohlcv_1m

**Two Models Coexist:**
- Old: `catboost_core` (5-day, daily bars, port 8000)
- New: `catboost_intraday` (5-min, minute bars, port 8001)
- Both can run simultaneously

---

## 📖 Documentation

- `DEPLOYMENT_GUIDE.md` - Full installation guide
- `N8N_INTEGRATION_GUIDE.md` - N8N workflow integration
- `TECHNICAL_DESIGN_DECISIONS.md` - Why we made certain choices
- `CRITICAL_FIXES_APPLIED.md` - What was fixed based on your feedback
- `README_COMPLETE_SOLUTION.md` - Executive summary

---

**Print this card and keep it handy during deployment! 📌**
