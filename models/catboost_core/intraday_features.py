# intraday_features.py
from __future__ import annotations

import os
from typing import Optional, Tuple, Dict

import duckdb
import pandas as pd

# Timezone for reporting freshness metadata
NY_TZ = "America/New_York"

# Pointer file that contains the path to the CURRENT (active) DuckDB snapshot
SNAPSHOT_POINTER = r"D:\AARCH\DBs\CURRENT_SNAPSHOT.txt"

# Columns expected in the materialized snapshot table
SNAP_COLS = [
    "rsi_14", "macd_line", "macd_signal", "macd_hist", "dist_vwap_bps",
    "close_d", "mkt_ret_1d", "sector_ret_1d", "rel_sector_vs_mkt",
    "rv_30m", "dist_to_high_bps", "dist_to_low_bps", "minute_of_day_pct", "vol_zscore_today",
]


def resolve_db_path() -> str:
    """
    Resolve the active DuckDB path with this priority:
      1) Snapshot pointer file (CURRENT_SNAPSHOT.txt)
      2) DB_PATH environment variable
      3) Safe default: D:\\AARCH\\DBs\\market.duckdb  (only used if 1/2 missing)
    """
    # 1) Pointer file
    try:
        with open(SNAPSHOT_POINTER, "r", encoding="utf-8") as f:
            p = f.read().strip().strip('"').strip("'")
            if p:
                return p
    except Exception:
        pass

    # 2) Environment
    env_db = os.getenv("DB_PATH")
    if env_db:
        return env_db

    # 3) Default (won't be used if the pointer is maintained correctly)
    return r"D:\AARCH\DBs\market.duckdb"


def latest_snapshot_features(db_path: Optional[str], symbol: str) -> Tuple[Dict[str, Optional[float]], Optional[str], Dict]:
    """
    Snapshot-only reader.
    Returns (feats, session_date_str[NY], info) pulled from features_intraday_1m
    for the latest snapshot per symbol.
    """
    if not db_path:
        db_path = resolve_db_path()

    conn = duckdb.connect(db_path, read_only=True)
    try:
        q = """
        SELECT *
        FROM features_intraday_1m
        WHERE symbol = ?
        QUALIFY row_number() OVER (PARTITION BY symbol ORDER BY snapshot_ts_utc DESC) = 1
        """
        df = conn.execute(q, [symbol]).df()

        if df.empty:
            feats = {k: None for k in SNAP_COLS}
            info = {
                "latest_ts_utc": None,
                "latest_ts_ny": None,
                "stale_sec": None,
                "in_rth": False,
                "bars_today": 0,
                "enough_rsi": False,
                "enough_macd": False,
                "enough_vwap": False,
                "used_rth_only": True,
            }
            return feats, None, info

        r = df.iloc[0]

        # features payload
        feats: Dict[str, Optional[float]] = {}
        for k in SNAP_COLS:
            v = r.get(k)
            feats[k] = None if pd.isna(v) else float(v)

        # session (NY date) saved by the materializer
        session = str(r.get("session_ny")) if pd.notna(r.get("session_ny")) else None

        # freshness / meta
        last_ts_utc = pd.to_datetime(r.get("snapshot_ts_utc"), utc=True, errors="coerce")
        last_ts_ny = last_ts_utc.tz_convert(NY_TZ) if pd.notna(last_ts_utc) else None
        now_utc = pd.Timestamp.now(tz="UTC")
        stale_sec = int((now_utc - last_ts_utc).total_seconds()) if pd.notna(last_ts_utc) else None

        info = {
            "latest_ts_utc": last_ts_utc.isoformat() if pd.notna(last_ts_utc) else None,
            "latest_ts_ny": last_ts_ny.isoformat() if last_ts_ny is not None else None,
            "stale_sec": stale_sec,
            "in_rth": bool(r.get("in_rth")) if r.get("in_rth") is not None else False,
            "bars_today": int(r.get("bars_today")) if pd.notna(r.get("bars_today")) else 0,
            # by definition, the snapshot was produced after adequate bars/calcs
            "enough_rsi": True,
            "enough_macd": True,
            "enough_vwap": True,
            "used_rth_only": True,
        }
        return feats, session, info
    finally:
        conn.close()


# ------------------------------------------------------------------------------------
# Compatibility wrapper:
# Some callers may still import/expect compute_intraday_features(...).
# To enforce "snapshot-only" reads, we delegate to latest_snapshot_features(...)
# and ignore live-compute parameters.
# ------------------------------------------------------------------------------------
def compute_intraday_features(
    db_path: str,
    symbol: str,
    mkt_etf: str = "SPY",
    sector_etf: Optional[str] = None,
    minutes: int = 800,
    rth_only: bool = True,
) -> Tuple[Dict[str, Optional[float]], Optional[str], Dict]:
    """
    Snapshot-only compatibility wrapper.
    Ignores live-compute params and returns the latest materialized row for `symbol`.
    """
    return latest_snapshot_features(db_path, symbol)


__all__ = [
    "latest_snapshot_features",
    "compute_intraday_features",
    "resolve_db_path",
]
