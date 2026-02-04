import os
import duckdb
import pandas as pd
from intraday_features import compute_intraday_features

# --- Snapshot pointer (highest priority) ---
SNAPSHOT_POINTER = r"D:\AARCH\DBs\CURRENT_SNAPSHOT.txt"

def resolve_db_path() -> str:
    """
    Resolve the active DuckDB path.
    Priority:
      1) Read from snapshot pointer file (CURRENT_SNAPSHOT.txt)
      2) DB_PATH environment variable
      3) Default to D:\\AARCH\\DBs\\market.duckdb
    """
    # 1) Pointer file
    try:
        with open(SNAPSHOT_POINTER, "r", encoding="utf-8") as f:
            snap = f.read().strip().strip('"').strip("'")
            if snap:
                return snap
    except Exception:
        pass

    # 2) Env var
    env_val = os.getenv("DB_PATH")
    if env_val:
        return env_val

    # 3) Default
    return r"D:\AARCH\DBs\market.duckdb"

DB_PATH = resolve_db_path()
MINUTES_LOOKBACK = int(os.getenv("INTRADAY_MINUTES", "800"))

def get_symbols(conn: duckdb.DuckDBPyConnection):
    """
    Prefer symbols from the materialized snapshot (features_intraday_1m).
    If empty (first run), fall back to recent symbols in base ohlcv_1m.
    """
    # Prefer snapshot-driven universe
    q_snap = "SELECT DISTINCT symbol FROM features_intraday_1m"
    syms = [r[0] for r in conn.execute(q_snap).fetchall()]
    if syms:
        return syms

    # Fallback if snapshot empty
    q_fallback = """
        SELECT DISTINCT symbol
        FROM ohlcv_1m
        WHERE ts > NOW() - INTERVAL 2 DAY
    """
    return [r[0] for r in conn.execute(q_fallback).fetchall()]

def upsert(conn: duckdb.DuckDBPyConnection, symbol: str, feats: dict, session, info: dict):
    """
    Upsert the computed intraday feature snapshot for the symbol at the current UTC time.
    Uses a registered DataFrame relation 'df' for MERGE to be explicit and robust.
    """
    snap_ts = pd.Timestamp.now(tz="UTC").to_pydatetime()
    row = {
        "symbol": symbol,
        "snapshot_ts_utc": snap_ts,
        "session_ny": pd.to_datetime(session).date() if session else None,
        "in_rth": info.get("in_rth"),
        "stale_sec": info.get("stale_sec"),
        "bars_today": info.get("bars_today"),
        # features
        "rsi_14": feats.get("rsi_14"),
        "macd_line": feats.get("macd_line"),
        "macd_signal": feats.get("macd_signal"),
        "macd_hist": feats.get("macd_hist"),
        "dist_vwap_bps": feats.get("dist_vwap_bps"),
        "close_d": feats.get("close_d"),
        "mkt_ret_1d": feats.get("mkt_ret_1d"),
        "sector_ret_1d": feats.get("sector_ret_1d"),
        "rel_sector_vs_mkt": feats.get("rel_sector_vs_mkt"),
        "rv_30m": feats.get("rv_30m"),
        "dist_to_high_bps": feats.get("dist_to_high_bps"),
        "dist_to_low_bps": feats.get("dist_to_low_bps"),
        "minute_of_day_pct": feats.get("minute_of_day_pct"),
        "vol_zscore_today": feats.get("vol_zscore_today"),
    }
    pdf = pd.DataFrame([row])

    # Register the DataFrame explicitly for MERGE
    conn.register("df", pdf)
    conn.execute("""
        MERGE INTO features_intraday_1m AS t
        USING df
        ON t.symbol = df.symbol AND t.snapshot_ts_utc = df.snapshot_ts_utc
        WHEN MATCHED THEN UPDATE SET
          session_ny=df.session_ny, in_rth=df.in_rth, stale_sec=df.stale_sec, bars_today=df.bars_today,
          rsi_14=df.rsi_14, macd_line=df.macd_line, macd_signal=df.macd_signal, macd_hist=df.macd_hist,
          dist_vwap_bps=df.dist_vwap_bps, close_d=df.close_d, mkt_ret_1d=df.mkt_ret_1d, sector_ret_1d=df.sector_ret_1d, rel_sector_vs_mkt=df.rel_sector_vs_mkt,
          rv_30m=df.rv_30m, dist_to_high_bps=df.dist_to_high_bps, dist_to_low_bps=df.dist_to_low_bps, minute_of_day_pct=df.minute_of_day_pct, vol_zscore_today=df.vol_zscore_today
        WHEN NOT MATCHED THEN INSERT *
    """)
    conn.unregister("df")

def run_once():
    # Open the DB resolved via the pointer (read/write)
    with duckdb.connect(DB_PATH) as conn:
        syms = get_symbols(conn)
        for s in syms:
            try:
                feats, session, info = compute_intraday_features(
                    DB_PATH, s, mkt_etf="SPY", sector_etf=None,
                    minutes=MINUTES_LOOKBACK, rth_only=True
                )
                upsert(conn, s, feats, session, info)
            except Exception as e:
                print("ERR", s, e)

if __name__ == "__main__":
    run_once()
