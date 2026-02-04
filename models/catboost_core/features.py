import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from dateutil import parser as dtp

from config import FEATURE_ORDER, NUM_FEATURES, CAT_FEATURES, ID_COLS
from db import get_conn, fetch_daily_row, fetch_minute_row, _to_utc_iso

def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in FEATURE_ORDER:
        if c not in df.columns:
            df[c] = np.nan if c not in ID_COLS and c not in CAT_FEATURES else None
    return df[FEATURE_ORDER]

def hydrate_from_db(row: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing technicals from DuckDB (1d: rsi/macd; 1m: dist_vwap_bps)."""
    symbol = row["symbol"]
    asof_iso = _to_utc_iso(row["asof_utc"])
    with get_conn() as conn:
        d1 = fetch_daily_row(conn, symbol, asof_iso)
        if not d1.empty:
            for k in ["rsi_14","macd_line","macd_signal","macd_hist"]:
                if row.get(k) is None and pd.notna(d1.iloc[0].get(k)):
                    row[k] = float(d1.iloc[0][k])
        m1 = fetch_minute_row(conn, symbol, asof_iso)
        if not m1.empty and row.get("dist_vwap_bps") is None:
            val = m1.iloc[0].get("dist_vwap_bps")
            if pd.notna(val):
                row["dist_vwap_bps"] = float(val)
    return row

def rows_to_frame(jobs: List[Dict[str, Any]]) -> pd.DataFrame:
    # flatten rows across jobs
    rows: List[Dict[str, Any]] = []
    for job in jobs:
        for r in job["rows"]:
            rows.append(dict(r))
    df = pd.DataFrame(rows)
    # hydrate missing from DB
    df = df.apply(lambda s: pd.Series(hydrate_from_db(s.to_dict())), axis=1)
    # numeric casting
    _coerce_numeric(df, NUM_FEATURES)
    # enforce column order + placeholders
    df = _ensure_columns(df)
    # cat features as strings (None stays None)
    for c in CAT_FEATURES:
        df[c] = df[c].astype("string")
    return df

def save_feature_order(path, order=None):
    """
    Persist the EXACT inference column order.
    Default is FEATURE_COLUMNS (NUM+CAT only), NOT the ID-inclusive list.
    """
    import json
    from config import FEATURE_COLUMNS

    if order is None:
        order = FEATURE_COLUMNS

    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(order), f, indent=2)
