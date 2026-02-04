#!/usr/bin/env python3
# D:\AARCH\DBs\update_features.py
# Compute RSI(14), MACD(12,26,9), and dist_vwap_bps for the latest session(s)
# and upsert into features_daily / features_intraday in market.duckdb.

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timezone

DB_PATH = r"D:\AARCH\DBs\market.duckdb"

# ---------- helpers ----------
def rsi_14(close: pd.Series) -> float | None:
    s = close.dropna().astype(float)
    if len(s) < 15:
        return None
    delta = s.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain).rolling(14).mean()
    roll_dn = pd.Series(loss).rolling(14).mean().replace(0, np.nan)
    rs = roll_up / roll_dn
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return None if pd.isna(val) else float(round(val, 2))

def macd_vals(close: pd.Series, fast=12, slow=26, signal=9):
    s = close.dropna().astype(float)
    if len(s) < slow + signal + 5:
        return (None, None, None)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return (
        float(round(macd_line.iloc[-1], 6)),
        float(round(macd_signal.iloc[-1], 6)),
        float(round(macd_hist.iloc[-1], 6)),
    )

def dist_vwap_bps(df_1m: pd.DataFrame) -> float | None:
    if df_1m.empty:
        return None
    df = df_1m.dropna(subset=["close","volume","high","low"]).copy()
    if df.empty:
        return None
    tp = (df["high"] + df["low"] + df["close"]) / 3.0  # typical price
    v_sum = df["volume"].sum()
    if v_sum == 0 or pd.isna(v_sum):
        return None
    vwap = float((tp * df["volume"]).sum() / v_sum)
    last = float(df["close"].iloc[-1])
    if vwap == 0:
        return None
    return float(round((last / vwap - 1.0) * 10000.0, 2))  # basis points

# ---------- main ----------
con = duckdb.connect(DB_PATH)

# Pick the most recent DATEs we have for each frequency
last_daily_date = con.execute("SELECT max(date(ts)) FROM ohlcv_1d").fetchone()[0]
last_intraday_date = con.execute("SELECT max(date(ts)) FROM ohlcv_1m").fetchone()[0]

# ---- DAILY features for symbols that have data on last_daily_date ----
if last_daily_date is not None:
    syms = con.execute("""
        SELECT DISTINCT symbol FROM ohlcv_1d
        WHERE date(ts) = ?
    """, [last_daily_date]).df()["symbol"].tolist()

    for sym in syms:
        # get enough history for indicators (e.g., 200 bars)
        df = con.execute("""
            SELECT ts, close
            FROM ohlcv_1d
            WHERE symbol = ?
            ORDER BY ts
            LIMIT 100000
        """, [sym]).df()

        if df.empty:
            continue

        # compute indicators on the full history; take latest for the session
        rsi = rsi_14(df["close"])
        macd_line, macd_signal, macd_hist = macd_vals(df["close"])

        # UPSERT into features_daily (DuckDB: INSERT OR REPLACE honors PK)
        con.execute("""
            INSERT OR REPLACE INTO features_daily (symbol, session, rsi_14, macd_line, macd_signal, macd_hist, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, now())
        """, [sym, last_daily_date, rsi, macd_line, macd_signal, macd_hist])

# ---- INTRADAY feature for last_intraday_date (dist_vwap_bps) ----
if last_intraday_date is not None:
    syms_1m = con.execute("""
        SELECT DISTINCT symbol FROM ohlcv_1m
        WHERE date(ts) = ?
    """, [last_intraday_date]).df()["symbol"].tolist()

    for sym in syms_1m:
        df1m = con.execute("""
            SELECT ts, open, high, low, close, volume
            FROM ohlcv_1m
            WHERE symbol = ? AND date(ts) = ?
            ORDER BY ts
        """, [sym, last_intraday_date]).df()

        d_bps = dist_vwap_bps(df1m)

        con.execute("""
            INSERT OR REPLACE INTO features_intraday (symbol, session, dist_vwap_bps, updated_at)
            VALUES (?, ?, ?, now())
        """, [sym, last_intraday_date, d_bps])

print("Features updated:",
      "daily date =", last_daily_date,
      "| intraday date =", last_intraday_date)
