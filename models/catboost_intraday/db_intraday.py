# D:\AARCH\models\catboost_intraday\db.py
# DuckDB access for 5-minute intraday CatBoost model
# CRITICAL: Always reads from SNAPSHOT (via CURRENT_SNAPSHOT.txt pointer)

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Union
from datetime import datetime, timezone, timedelta

import duckdb
import pandas as pd

# Import from config (with fallbacks)
try:
    from config_intraday import (
        DUCKDB_PATH, HORIZON_MIN, TRAIN_DAYS, RTH_START, RTH_END, TRAIN_SYMBOLS_LIMIT
    )
except ImportError:
    DUCKDB_PATH = r"D:\AARCH\DBs\market.duckdb"
    HORIZON_MIN = 5
    TRAIN_DAYS = 60
    RTH_START = "09:30"
    RTH_END = "16:00"
    TRAIN_SYMBOLS_LIMIT = 50

# Snapshot pointer (same directory as primary DB)
SNAPSHOT_POINTER = Path(DUCKDB_PATH).parent / "CURRENT_SNAPSHOT.txt"


# ---------- Snapshot resolution ----------

def _resolve_db_path() -> Path:
    """
    Resolve DB path with priority:
      1. Snapshot pointer file (CURRENT_SNAPSHOT.txt)
      2. Primary DB (market.duckdb)
    
    CRITICAL: Always prefer snapshot to avoid locks with live writers.
    """
    # Try snapshot pointer first
    if SNAPSHOT_POINTER.exists():
        try:
            raw = SNAPSHOT_POINTER.read_text(encoding="utf-8").strip().strip('"').strip("'")
            if raw:
                snapshot_path = Path(raw)
                if snapshot_path.exists() and snapshot_path.is_file():
                    return snapshot_path
        except Exception as e:
            print(f"[WARN] Failed to read snapshot pointer: {e}")
    
    # Fallback to primary DB
    return Path(DUCKDB_PATH)


def _connect_ro(db_path: Path) -> duckdb.DuckDBPyConnection:
    """
    Open DuckDB read-only; if locked, copy to temp and open that.
    """
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except Exception:
        # Last resort: copy to temp
        tmp = Path(tempfile.gettempdir()) / f"{db_path.stem}_ro_{os.getpid()}.duckdb"
        try:
            shutil.copy2(db_path, tmp)
            return duckdb.connect(str(tmp), read_only=True)
        except Exception as e2:
            raise RuntimeError(f"Failed to open DB at {db_path}: {e2}") from e2


# ---------- Connection ----------

def get_conn() -> duckdb.DuckDBPyConnection:
    """
    Public entrypoint: returns read-only connection to SNAPSHOT (not live DB).
    Use with context manager: `with get_conn() as conn: ...`
    """
    db_path = _resolve_db_path()
    
    # Log which DB we're using (helps debugging)
    if "snap" in str(db_path).lower():
        print(f"[DB] Using snapshot: {db_path}")
    else:
        print(f"[DB] WARNING: Using primary DB (not snapshot): {db_path}")
    
    return _connect_ro(db_path)


# ---------- Time helpers ----------

def _to_utc_iso(dt_like: Union[str, datetime]) -> str:
    """Normalize to UTC ISO string."""
    if isinstance(dt_like, datetime):
        dt = dt_like
    else:
        s = str(dt_like).strip()
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            dt = datetime.utcnow().replace(tzinfo=timezone.utc)
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------- Fetch helpers ----------

def fetch_minute_row(conn: duckdb.DuckDBPyConnection, symbol: str, asof_utc: Union[str, datetime]):
    """
    Latest 1m row <= asof (UTC). Returns DataFrame with at most 1 row.
    """
    asof_iso = _to_utc_iso(asof_utc)
    q = """
        SELECT *
        FROM ohlcv_1m
        WHERE symbol = ? AND ts <= ?
        ORDER BY ts DESC
        LIMIT 1
    """
    return conn.execute(q, [symbol, asof_iso]).fetchdf()


def fetch_spy_context(conn: duckdb.DuckDBPyConnection, asof_utc: Union[str, datetime], lookback_min: int = 30):
    """
    Fetch SPY bars for the last `lookback_min` minutes before asof_utc.
    Used to compute mkt_ret_5m, mkt_ret_30m.
    """
    asof_iso = _to_utc_iso(asof_utc)
    q = f"""
        SELECT ts, close
        FROM ohlcv_1m
        WHERE symbol = 'SPY'
          AND ts <= CAST(? AS TIMESTAMPTZ)
          AND ts >= CAST(? AS TIMESTAMPTZ) - INTERVAL '{lookback_min} minutes'
        ORDER BY ts
    """
    return conn.execute(q, [asof_iso, asof_iso]).fetchdf()



# ---------- Training frame (5-minute horizon) ----------

def fetch_training_frame_5m(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Build training frame from ohlcv_1m for 5-minute forward return prediction.
    
    Features computed:
      - ret_1m, ret_5m, ret_15m, ret_30m (momentum)
      - range_pct = (high - low) / close
      - minute_of_day (0-389 for 09:30-16:00 ET)
      - minute_of_day_pct (normalized 0-1)
      - rv_5m, rv_15m, rv_30m (realized volatility)
      - vol_ratio_5m (volume vs 5m avg)
      - log_volume
      - hour_of_day (categorical: 9-15)
    
    Label:
      - fwd_ret_5m = (close[t+5] / close[t]) - 1
    
    Filters:
      - RTH only (09:30-16:00 ET)
      - Last TRAIN_DAYS days
      - Top TRAIN_SYMBOLS_LIMIT symbols by row count (for fast iteration)
    
    Returns:
      DataFrame with columns: [ts, symbol, <features>, fwd_ret_5m]
    """
    
    h = int(HORIZON_MIN)
    days = int(TRAIN_DAYS)
    sym_limit = int(TRAIN_SYMBOLS_LIMIT)
    rth_start = RTH_START
    rth_end = RTH_END
    
    # Step 1: Find top N symbols by bar count (for C1 fast iteration)
    q_symbols = f"""
        WITH recent AS (
            SELECT symbol, COUNT(*) AS bar_count
            FROM ohlcv_1m
            WHERE ts >= CURRENT_DATE - INTERVAL '{days} days'
              AND EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') * 60 
                  + EXTRACT(MINUTE FROM ts AT TIME ZONE 'America/New_York') 
                  BETWEEN 
                      (CAST(SPLIT_PART('{rth_start}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{rth_start}', ':', 2) AS INT))
                  AND 
                      (CAST(SPLIT_PART('{rth_end}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{rth_end}', ':', 2) AS INT))
            GROUP BY symbol
            ORDER BY bar_count DESC
            LIMIT {sym_limit}
        )
        SELECT symbol FROM recent
    """
    symbols = [r[0] for r in conn.execute(q_symbols).fetchall()]
    
    if not symbols:
        return pd.DataFrame()
    
    # Convert symbols list to SQL IN clause
    sym_list = ", ".join(f"'{s}'" for s in symbols)
    
    # Step 2: Build feature frame with window functions
    # Split into stages to avoid nested window functions
    q_features = f"""
    WITH base AS (
        SELECT
            symbol,
            ts,
            ts AT TIME ZONE 'America/New_York' AS ts_ny,
            open, high, low, close, volume,
            dist_vwap_bps
        FROM ohlcv_1m
        WHERE symbol IN ({sym_list})
          AND ts >= CURRENT_DATE - INTERVAL '{days} days'
          AND EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') * 60 
              + EXTRACT(MINUTE FROM ts AT TIME ZONE 'America/New_York') 
              BETWEEN 
                  (CAST(SPLIT_PART('{rth_start}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{rth_start}', ':', 2) AS INT))
              AND 
                  (CAST(SPLIT_PART('{rth_end}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{rth_end}', ':', 2) AS INT))
    ),
    log_returns AS (
        SELECT
            symbol, ts, open, high, low, close, volume, dist_vwap_bps,
            LN(close / NULLIF(LAG(close, 1) OVER w, 0)) AS log_ret
        FROM base
        WINDOW w AS (PARTITION BY symbol ORDER BY ts)
    ),
    features AS (
        SELECT
            symbol, ts,
            open, high, low, close, volume,
            dist_vwap_bps,
            
            -- Returns (momentum signals)
            close / NULLIF(LAG(close, 1) OVER w, 0) - 1.0 AS ret_1m,
            close / NULLIF(LAG(close, 5) OVER w, 0) - 1.0 AS ret_5m,
            close / NULLIF(LAG(close, 15) OVER w, 0) - 1.0 AS ret_15m,
            close / NULLIF(LAG(close, 30) OVER w, 0) - 1.0 AS ret_30m,
            
            -- Range (volatility proxy)
            (high - low) / NULLIF(close, 0) AS range_pct,
            
            -- Volume features
            LN(NULLIF(volume, 0) + 1) AS log_volume,
            volume / NULLIF(AVG(volume) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING), 0) AS vol_ratio_5m,
            
            -- Intraday position (minute of day: 0 = 09:30 ET, 389 = 15:59 ET)
            (EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') * 60 
             + EXTRACT(MINUTE FROM ts AT TIME ZONE 'America/New_York')
             - (CAST(SPLIT_PART('{rth_start}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{rth_start}', ':', 2) AS INT))
            ) AS minute_of_day,
            
            -- Normalized minute (0-1 scale)
            (EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') * 60 
             + EXTRACT(MINUTE FROM ts AT TIME ZONE 'America/New_York')
             - (CAST(SPLIT_PART('{rth_start}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{rth_start}', ':', 2) AS INT))
            ) / 390.0 AS minute_of_day_pct,
            
            -- Hour of day (categorical)
            EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') AS hour_of_day,
            
            -- Realized volatility (std of log returns - computed from log_returns CTE)
            STDDEV(log_ret) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 5 PRECEDING AND CURRENT ROW) AS rv_5m,
            STDDEV(log_ret) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 15 PRECEDING AND CURRENT ROW) AS rv_15m,
            STDDEV(log_ret) OVER (PARTITION BY symbol ORDER BY ts ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) AS rv_30m,
            
            -- Forward return (LABEL)
            LEAD(close, {h}) OVER w / NULLIF(close, 0) - 1.0 AS fwd_ret_{h}m
            
        FROM log_returns
        WINDOW w AS (PARTITION BY symbol ORDER BY ts)
    ),
    spy_context AS (
        SELECT
            ts,
            close / NULLIF(LAG(close, 5) OVER (ORDER BY ts), 0) - 1.0 AS mkt_ret_5m,
            close / NULLIF(LAG(close, 30) OVER (ORDER BY ts), 0) - 1.0 AS mkt_ret_30m
        FROM ohlcv_1m
        WHERE symbol = 'SPY'
          AND ts >= CURRENT_DATE - INTERVAL '{days} days'
          AND EXTRACT(HOUR FROM ts AT TIME ZONE 'America/New_York') * 60 
              + EXTRACT(MINUTE FROM ts AT TIME ZONE 'America/New_York') 
              BETWEEN 
                  (CAST(SPLIT_PART('{rth_start}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{rth_start}', ':', 2) AS INT))
              AND 
                  (CAST(SPLIT_PART('{rth_end}', ':', 1) AS INT) * 60 + CAST(SPLIT_PART('{rth_end}', ':', 2) AS INT))
    )
    SELECT
        f.symbol,
        f.ts,
        f.open,
        f.high,
        f.low,
        f.close,
        f.volume,
        f.dist_vwap_bps,
        f.ret_1m,
        f.ret_5m,
        f.ret_15m,
        f.ret_30m,
        f.range_pct,
        f.log_volume,
        f.vol_ratio_5m,
        f.minute_of_day,
        f.minute_of_day_pct,
        f.hour_of_day,
        f.rv_5m,
        f.rv_15m,
        f.rv_30m,
        f.fwd_ret_{h}m,
        spy.mkt_ret_5m,
        spy.mkt_ret_30m
    FROM features f
    LEFT JOIN spy_context spy ON f.ts = spy.ts
    WHERE f.fwd_ret_{h}m IS NOT NULL  -- Drop rows without forward return
      AND f.ret_1m IS NOT NULL        -- Drop first bar (no lag)
    ORDER BY f.symbol, f.ts
    """
    
    return conn.execute(q_features).fetchdf()


# ---------- Inference helper ----------

def fetch_latest_bars_for_inference(
    conn: duckdb.DuckDBPyConnection, 
    symbol: str, 
    asof_utc: Union[str, datetime],
    lookback_bars: int = 30
) -> pd.DataFrame:
    """
    Fetch last N bars for a symbol to compute features at inference time.
    Used by serve.py to calculate rolling features (rv_30m, vol_ratio_5m, etc.)
    """
    asof_iso = _to_utc_iso(asof_utc)
    q = f"""
        SELECT ts, open, high, low, close, volume, dist_vwap_bps
        FROM ohlcv_1m
        WHERE symbol = ?
          AND ts <= ?
        ORDER BY ts DESC
        LIMIT {lookback_bars}
    """
    df = conn.execute(q, [symbol, asof_iso]).fetchdf()
    return df.sort_values("ts")  # Oldest to newest for window calcs

 
