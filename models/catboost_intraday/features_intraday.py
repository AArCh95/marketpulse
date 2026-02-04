# D:\AARCH\models\catboost_intraday\features.py
# Feature engineering for 5-minute intraday predictions at inference time

import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime, timezone

from config_intraday import FEATURE_COLUMNS, NUM_FEATURES, CAT_FEATURES, ID_COLS
from db_intraday import get_conn, fetch_latest_bars_for_inference, fetch_spy_context

def _to_ny_time(ts_utc: pd.Timestamp) -> pd.Timestamp:
    """Convert UTC timestamp to NY time."""
    return ts_utc.tz_convert("America/New_York")


def _minute_of_day(ts_ny: pd.Timestamp) -> int:
    """
    Calculate minute of trading day (0 = 09:30 ET, 389 = 15:59 ET).
    Returns -1 if outside RTH.
    """
    hour = ts_ny.hour
    minute = ts_ny.minute
    
    # RTH: 09:30 - 16:00 ET
    if hour < 9 or (hour == 9 and minute < 30) or hour >= 16:
        return -1
    
    minutes_since_open = (hour - 9) * 60 + minute - 30
    return minutes_since_open


def compute_rolling_features(bars: pd.DataFrame) -> Dict[str, float]:
    """
    Compute rolling features from recent bars (for single inference point).
    
    Input: DataFrame with columns [ts, open, high, low, close, volume, dist_vwap_bps]
           Sorted oldest to newest (last row = current bar).
    
    Returns: Dict with computed features for the LAST bar.
    """
    if bars.empty or len(bars) < 2:
        # Not enough data for rolling calcs
        return {f: None for f in NUM_FEATURES}
    
    # Ensure numeric
    for col in ["open", "high", "low", "close", "volume"]:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    
    # Get current (last) bar
    curr = bars.iloc[-1]
    
    feats = {}
    
    # Current bar price features
    feats["close"] = float(curr["close"]) if pd.notna(curr["close"]) else None
    feats["high"] = float(curr["high"]) if pd.notna(curr["high"]) else None
    feats["low"] = float(curr["low"]) if pd.notna(curr["low"]) else None
    feats["open"] = float(curr["open"]) if pd.notna(curr["open"]) else None
    feats["volume"] = float(curr["volume"]) if pd.notna(curr["volume"]) else None
    feats["dist_vwap_bps"] = float(curr["dist_vwap_bps"]) if pd.notna(curr["dist_vwap_bps"]) else None
    
    # Range percentage
    if feats["high"] and feats["low"] and feats["close"]:
        feats["range_pct"] = (feats["high"] - feats["low"]) / feats["close"]
    else:
        feats["range_pct"] = None
    
    # Log volume
    if feats["volume"] and feats["volume"] > 0:
        feats["log_volume"] = float(np.log(feats["volume"] + 1))
    else:
        feats["log_volume"] = None
    
    # Momentum returns (need historical bars)
    closes = bars["close"].values
    
    def safe_return(lag: int) -> float:
        if len(closes) <= lag or closes[-lag-1] == 0:
            return None
        return float(closes[-1] / closes[-lag-1] - 1.0)
    
    feats["ret_1m"] = safe_return(1)
    feats["ret_5m"] = safe_return(5)
    feats["ret_15m"] = safe_return(15)
    feats["ret_30m"] = safe_return(30)
    
    # Volume ratio (current vs recent 5-bar avg)
    if len(bars) >= 6:
        recent_vol = bars["volume"].iloc[-6:-1].mean()
        if recent_vol > 0:
            feats["vol_ratio_5m"] = float(feats["volume"] / recent_vol) if feats["volume"] else None
        else:
            feats["vol_ratio_5m"] = None
    else:
        feats["vol_ratio_5m"] = None
    
    # Realized volatility (std of log returns)
    def realized_vol(window: int) -> float:
        if len(closes) <= window:
            return None
        log_rets = np.log(closes[-window:] / closes[-window-1:-1])
        log_rets = log_rets[np.isfinite(log_rets)]
        if len(log_rets) < 2:
            return None
        return float(np.std(log_rets))
    
    feats["rv_5m"] = realized_vol(5)
    feats["rv_15m"] = realized_vol(15)
    feats["rv_30m"] = realized_vol(30)
    
    # Intraday position features (computed from current timestamp)
    ts_utc = pd.to_datetime(curr["ts"], utc=True)
    ts_ny = _to_ny_time(ts_utc)
    
    mod = _minute_of_day(ts_ny)
    feats["minute_of_day"] = float(mod) if mod >= 0 else None
    feats["minute_of_day_pct"] = float(mod / 390.0) if mod >= 0 else None
    
    # Hour of day (categorical)
    feats["hour_of_day"] = str(ts_ny.hour) if 9 <= ts_ny.hour < 16 else None
    
    return feats


def compute_market_context(conn, ts_utc: pd.Timestamp) -> Dict[str, float]:
    """
    Compute SPY market returns for context features.
    """
    spy_bars = fetch_spy_context(conn, ts_utc, lookback_min=30)
    
    if spy_bars.empty or len(spy_bars) < 2:
        return {"mkt_ret_5m": None, "mkt_ret_30m": None}
    
    closes = spy_bars["close"].values
    
    def spy_return(lag: int) -> float:
        if len(closes) <= lag or closes[-lag-1] == 0:
            return None
        return float(closes[-1] / closes[-lag-1] - 1.0)
    
    return {
        "mkt_ret_5m": spy_return(5),
        "mkt_ret_30m": spy_return(30),
    }


def hydrate_single_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hydrate a single inference row with computed intraday features.
    
    N8N sends complex payloads with many pre-computed fields (ret_1d, close_d, etc.).
    We INTENTIONALLY IGNORE these and recompute everything from ohlcv_1m for consistency.
    
    Why ignore daily data from n8n?
    - N8N payload has daily closes (close_d), daily returns (ret_1d, ret_5d)
    - These are from ohlcv_1d (lagging, once per day)
    - Our 5-minute model needs INTRADAY features from ohlcv_1m (updated every minute)
    - Using daily data would introduce train/serve skew
    
    Required fields in input:
      - symbol (string)
      - ts or asof_utc (UTC ISO timestamp)
    
    Returns: row with all FEATURE_COLUMNS populated (or None for missing).
    """
    symbol = row.get("symbol")
    
    # N8N can send ts, asof_utc, or timestamp
    ts_str = row.get("ts") or row.get("asof_utc") or row.get("timestamp")
    
    if not symbol or not ts_str:
        raise ValueError(f"Row must have 'symbol' and 'ts' (or 'asof_utc'). Got: {list(row.keys())}")
    
    # Parse timestamp
    ts_utc = pd.to_datetime(ts_str, utc=True)
    
    with get_conn() as conn:
        # Fetch recent bars for rolling features
        bars = fetch_latest_bars_for_inference(conn, symbol, ts_utc, lookback_bars=35)
        
        if bars.empty:
            # No data available - return placeholder
            feats = {f: None for f in NUM_FEATURES}
            feats["symbol"] = symbol
            feats["hour_of_day"] = None
            return feats
        
        # Compute rolling features from bars
        feats = compute_rolling_features(bars)
        
        # Add market context
        mkt = compute_market_context(conn, ts_utc)
        feats.update(mkt)
        
        # Add categorical
        feats["symbol"] = symbol
        # hour_of_day already set in compute_rolling_features
    
    return feats


def rows_to_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of inference rows to feature DataFrame ready for CatBoost.
    
    Input: List of dicts with at minimum {symbol, ts}
    Output: DataFrame with columns = FEATURE_COLUMNS, properly typed.
    """
    hydrated = []
    for r in rows:
        try:
            h = hydrate_single_row(r)
            hydrated.append(h)
        except Exception as e:
            print(f"Warning: Failed to hydrate {r.get('symbol')}: {e}")
            # Add placeholder row
            h = {f: None for f in FEATURE_COLUMNS}
            h["symbol"] = r.get("symbol", "UNKNOWN")
            hydrated.append(h)
    
    df = pd.DataFrame(hydrated)
    
    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            if col in CAT_FEATURES:
                df[col] = None
            else:
                df[col] = np.nan
    
    # Type conversions
    for col in NUM_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("None", "unknown")
    
    # Return only model features (drop any extra columns)
    return df[FEATURE_COLUMNS]


def save_feature_order(path, order=None):
    """Persist feature order for model inference."""
    if order is None:
        order = FEATURE_COLUMNS
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(order), f, indent=2)
