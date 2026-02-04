# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

AARCH is an automated algorithmic trading system that combines:
- **Market data ingestion** from Yahoo Finance (daily + 1-minute intraday bars)
- **ML-based prediction models** (CatBoost for price movement, FinBERT/Ollama for sentiment)
- **Trading execution** via Alpaca API
- **Real-time dashboards** for monitoring positions and signals

The system operates on a **snapshot-based architecture** to avoid file-locking conflicts on Windows: ETL processes write to live DuckDB files, then publish read-only snapshots that downstream services consume.

## Core Architecture

### Data Flow Pipeline

```
Yahoo Finance → market_sync_yahoo.py → DuckDB (live) → Snapshot (immutable)
                                          ↓
                                    Model Training
                                          ↓
                                    Inference APIs ← n8n workflows
                                          ↓
                                    Trader Server → Alpaca
```

### Key Components

1. **DBs/** - Database and ETL scripts
   - `market.duckdb`: Market OHLCV data (daily + 1-minute bars)
   - `trader.duckdb`: Trading state (positions, PnL, trade history)
   - `signals.duckdb`: ML model predictions
   - `market_sync_yahoo.py`: Main ETL script with DAILY and INTRADAY modes
   - Snapshot pointers: `CURRENT_SNAPSHOT.txt` (market), `CURRENT_SNAPSHOT_TRADER.txt` (trader)

2. **models/** - ML models (each with train.py, serve.py, config.py pattern)
   - `catboost_core/`: Daily model (5-day horizon, NOT suitable for intraday)
   - `catboost_intraday/`: Intraday 5-minute model (correct for minute-scale trading)
   - `FinBert/`: Financial sentiment from FinBERT transformers
   - `Sent_an_ollama/`: Sentiment analysis using local Ollama LLMs

3. **trader/** - Trading execution and monitoring
   - `server.py`: FastAPI service handling webhooks, position management, Alpaca orders
   - `trader_dashboard.py`: Streamlit dashboard for portfolio monitoring

4. **dashboard/** - Market signal visualization
   - `app.py`: Streamlit dashboard for viewing ML predictions
   - `db_app.py`: Direct database query interface

### Database Schemas

**ohlcv_1d** (daily bars):
- Columns: symbol, date, open, high, low, close, volume, rsi_14, macd_line, macd_signal, macd_hist
- Updated: Once daily via `run_daily.ps1`

**ohlcv_1m** (1-minute bars):
- Columns: symbol, ts, open, high, low, close, volume, dist_vwap_bps
- Updated: Continuously during market hours via `run_intraday.ps1`

**features_intraday_1m** (materialized intraday features):
- Rolling technical indicators computed from ohlcv_1m
- Used by catboost_core for fast feature hydration

**positions** (trader.duckdb):
- Current holdings: symbol, qty, avg_entry, market_value, unrealized_pl
- Snapshots taken after each trade for historical tracking

**signals** (signals.duckdb):
- Model predictions: symbol, ts, pred_class (up/down/flat), p_up, p_down, p_flat, p_margin

## Common Commands

### Database Operations

```bash
# Open market database
cd D:\AARCH\DBs
duckdb market.duckdb

# Open trader database
duckdb trader.duckdb

# Run daily ETL (once before market open)
powershell -File run_daily.ps1

# Run intraday ETL (continuous during market hours)
powershell -File run_intraday.ps1

# Manual ETL invocation
python market_sync_yahoo.py --duckdb "D:\AARCH\DBs\market.duckdb" --mode daily
python market_sync_yahoo.py --duckdb "D:\AARCH\DBs\market.duckdb" --mode intraday --batch-size 60 --sleep 0.35
```

### Model Training

Each model follows the pattern: configure `.env` → run `train.py` → serve via `serve.py`

```bash
# Train intraday model (60 days of 1-minute bars, ~30-60 min)
cd D:\AARCH\models\catboost_intraday
python train_intraday.py

# Train daily model (lighter, ~1-2 min)
cd D:\AARCH\models\catboost_core
python train.py

# Weekly automated retraining (intraday model)
cd D:\AARCH\models\catboost_intraday
python retrain_weekly.py
```

### Serving Models (APIs)

```bash
# CatBoost Core (port 8000) - daily model
cd D:\AARCH\models\catboost_core
python serve.py

# CatBoost Intraday (port 8001) - 5-minute model
cd D:\AARCH\models\catboost_intraday
python serve_intraday.py

# FinBERT (port 8080)
cd D:\AARCH\models\FinBert
python app.py

# Sentiment Ollama (port 8003)
cd D:\AARCH\models\Sent_an_ollama
python sentiment_api.py

# Trader server (port 8002)
cd D:\AARCH\trader
python server.py
```

### Dashboards

```bash
# Market signals dashboard
cd D:\AARCH\dashboard
streamlit run app.py

# Trader portfolio dashboard
cd D:\AARCH\trader
streamlit run trader_dashboard.py

# Database query interface
cd D:\AARCH\dashboard
streamlit run db_app.py
```

## Important Patterns and Conventions

### Snapshot-Based Database Access

**Problem**: Windows file-locking prevents simultaneous read/write to DuckDB.

**Solution**: Writers create timestamped snapshots; readers consume via pointer files.

```python
# Reading from snapshot (catboost_core/serve.py example)
from intraday_features import resolve_db_path

DB_PATH = resolve_db_path()  # Reads CURRENT_SNAPSHOT.txt → returns immutable snapshot path
conn = duckdb.connect(DB_PATH, read_only=True)
```

**Snapshot Creation** (market_sync_yahoo.py):
1. Write to live `market.duckdb`
2. Copy to `market_snap_<timestamp>.duckdb`
3. Update `CURRENT_SNAPSHOT.txt` with snapshot path
4. Old snapshots auto-cleaned (keep last 10)

### Model Training Workflow

All CatBoost models follow this structure:

```
config.py          # Paths, hyperparameters, feature definitions
db.py / db_*.py    # Database connectors, training data queries
features.py / features_*.py  # Feature engineering functions
train.py / train_*.py        # Training script (outputs .cbm + feature_order.json)
serve.py / serve_*.py        # FastAPI inference server
```

**Critical**: `catboost_core` (daily model) and `catboost_intraday` (5-minute model) are **separate systems**. The daily model predicts 5-day returns and is NOT suitable for minute-scale trading. Use `catboost_intraday` for intraday signals.

### Feature Engineering

**Daily features** (catboost_core):
- Source: `ohlcv_1d` table
- Indicators: RSI(14), MACD, distance to 52-week high/low
- News: sentiment scores from FinBERT/Ollama APIs
- Market context: beta_60d, sector_ret_1d, mkt_ret_1d

**Intraday features** (catboost_intraday):
- Source: `ohlcv_1m` table
- Momentum: ret_1m, ret_5m, ret_15m, ret_30m
- Volatility: rv_5m, rv_15m, rv_30m (realized volatility)
- VWAP: dist_vwap_bps (basis points from VWAP)
- Time: minute_of_day, hour_of_day (categorical)
- Volume: vol_ratio_5m, log_volume

**Materialization**: `update_features.py` pre-computes expensive features into `features_intraday_1m` table for fast serving.

### Trading Decision Logic

**trader/server.py** receives signals from n8n workflows (orchestration tool) and executes trades:

1. **Signal Validation**:
   - RTH check (09:30-16:00 ET)
   - Staleness check (asof_utc within max_stale_sec)
   - VWAP distance gate (|dist_vwap_bps| < threshold)
   - Margin gate (p_margin > min_margin)

2. **Position Sizing**:
   - Dynamic sizing based on signal confidence (p_margin)
   - Risk limits: MAX_WEIGHT_PCT per position
   - Take-profit: TP_PCT (default 3%)
   - Stop-loss: SL_PCT (default 2%)

3. **Order Execution**:
   - Uses Alpaca Trading API (paper or live mode)
   - Market orders with TimeInForce.DAY
   - Logs all decisions to trader.duckdb

### Configuration via .env Files

Each component has its own `.env` (never committed). Key variables:

**Trader** (trader/.env):
```
APCA_API_KEY_ID=<alpaca_key>
APCA_API_SECRET_KEY=<alpaca_secret>
ALPACA_PAPER=true
TRADER_WEBHOOK_TOKEN=<webhook_auth>
USD_PER_TRADE=100
PCT_PER_TRADE=2%
TP_PCT=3%
SL_PCT=2%
MAX_WEIGHT_PCT=0.4%
```

**Models** (models/*/​.env):
```
API_KEY=<model_api_key>
DB_SNAPSHOT_POINTER=D:\AARCH\DBs\CURRENT_SNAPSHOT.txt
SIGNALS_DB=D:\AARCH\DBs\signals.duckdb
```

**Sentiment** (models/Sent_an_ollama/.env):
```
OLLAMA_MODEL=llama3.2
RELEVANCE_THRESHOLD=0.6
```

## Deployment Notes

### Windows Task Scheduler Jobs

1. **Daily ETL**: 06:00 CR time (08:00 ET premarket)
   - Action: `powershell.exe -File D:\AARCH\DBs\run_daily.ps1`

2. **Intraday ETL**: 07:25 CR time (09:25 ET, before open)
   - Action: `powershell.exe -File D:\AARCH\DBs\run_intraday.ps1`
   - Runs continuously until manually stopped

3. **Weekly Model Retrain**: Sunday 02:00 ET
   - Action: `python D:\AARCH\models\catboost_intraday\retrain_weekly.py`

### Service Dependencies

- **Ollama** (for sentiment analysis): Must be running on localhost:11434
  ```bash
  ollama serve
  ollama pull llama3.2  # or llama3.1, mistral, phi3
  ```

- **Python Environment**: Python 3.13 at `C:\Users\Aaron\AppData\Local\Programs\Python\Python313\python.exe`

- **Virtual Environments**: Each model/service has its own venv/env (not committed)

### n8n Integration

External orchestration tool calls model APIs via webhooks. Key endpoints:

- `POST /score_batch` - CatBoost models (core + intraday)
- `POST /analyze` - Sentiment APIs
- `POST /webhook/signal` - Trader server (receives signals)

## File Locations and Paths

**Databases**: `D:\AARCH\DBs\`
- Prefer absolute paths in configs (Windows environment)
- Snapshot pointers allow location transparency

**Models**: `D:\AARCH\models\<model_name>\`
- Trained artifacts in `models/` subdirectory (e.g., `catboost_core.cbm`)
- Feature order JSON alongside model file

**Logs**: `D:\AARCH\DBs\logs\`
- Daily: `daily_<YYYYMMDD_HHMMSS>.log`
- Intraday: `intraday_<YYYYMMDD_HHMMSS>.log`

## Special Considerations

### Temporal Horizon Mismatch

**Critical**: Do NOT use `catboost_core` (daily model) for intraday trading decisions. It predicts 5-day forward returns, creating a fundamental temporal mismatch when used for 5-minute signals. Use `catboost_intraday` instead, which predicts 5-minute returns from 1-minute bars.

See `models/catboost_intraday/README_COMPLETE_SOLUTION.md` for detailed explanation.

### Symbol Fixups

`BRK.B` → `BRK-B` (Yahoo Finance format)

Full symbol list in `market_sync_yahoo.py:SYMBOLS` (~250 stocks)

### Time Zones

- **Database timestamps**: UTC
- **Market hours**: America/New_York (ET)
- **Local time**: America/Costa_Rica (CR, UTC-6)
- RTH (Regular Trading Hours): 09:30-16:00 ET

### GPU Acceleration

CatBoost models configured for GPU training (`task_type="GPU"`). Falls back to CPU if unavailable. Expect 10x speedup with GPU for large intraday datasets.

### Class Imbalance Handling

CatBoost models use tilted class weights to boost DOWN (class 0) recall:
- `TILT_DOWN = 1.25` (adjustable in config.py)
- Prevents model from ignoring short signals in imbalanced datasets

### Performance Expectations

**Training**:
- Daily model: 1-2 minutes (small dataset)
- Intraday model: 30-60 minutes (780K bars for 60 days × 50 symbols)

**Inference**:
- Daily model: ~50ms per symbol
- Intraday model: ~200ms per symbol
- Sentiment APIs: 1-3 seconds per article (LLM-based)

**Database queries**:
- Snapshot read: <10ms for feature hydration
- Live write: 100-500ms per batch (UPSERT with indicators)
