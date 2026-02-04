# D:\AARCH\dashboard\db_app.py
# Intraday viewer from DuckDB snapshots: candlesticks + VWAP + EMAs + volume
# + CatBoost signals overlay + daily leaderboard

import os
import glob
import json
from pathlib import Path
from datetime import datetime, date, time as dtime, timezone

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytz
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ------------------ Config ------------------
SNAPSHOT_DIR = os.getenv("DB_SNAPSHOT_DIR", r"D:\AARCH\DBs")
DB_PATH_FALLBACK = os.getenv("DB_PATH", r"D:\AARCH\DBs\market.duckdb")
SIGNALS_DB = os.getenv("SIGNALS_DB", r"D:\AARCH\DBs\signals.duckdb")
ET = pytz.timezone("America/New_York")

st.set_page_config(page_title="MarketPulse — Intraday DB", layout="wide")
st.title("📊 MarketPulse — Intraday desde DuckDB (Snapshots)")

# ------------------ Helpers ------------------

def to_et_naive(ts_utc):
    """Convert UTC tz-aware timestamps to ET-naive (works with Series or DatetimeIndex)."""
    if isinstance(ts_utc, pd.DatetimeIndex):
        return ts_utc.tz_convert(ET).tz_localize(None)
    s = pd.to_datetime(ts_utc, utc=True, errors="coerce")
    return s.dt.tz_convert(ET).dt.tz_localize(None)

def find_latest_snapshot() -> str:
    """
    Look for snapshot files like market_*.duckdb in SNAPSHOT_DIR.
    If none, fall back to DB_PATH_FALLBACK.
    """
    snap_dir = SNAPSHOT_DIR if os.path.isdir(SNAPSHOT_DIR) else str(Path(DB_PATH_FALLBACK).parent)
    candidates = sorted(
        glob.glob(os.path.join(snap_dir, "market_*.duckdb")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    if candidates:
        return candidates[0]
    return DB_PATH_FALLBACK

def rth_bounds_utc(session_et: date) -> tuple[datetime, datetime]:
    start_et = ET.localize(datetime.combine(session_et, dtime(9, 30)))
    end_et = ET.localize(datetime.combine(session_et, dtime(16, 0)))
    return start_et.astimezone(timezone.utc), end_et.astimezone(timezone.utc)

def _safe_json_loads(x):
    if x is None or (isinstance(x, str) and x.strip() == ""):
        return {}
    try:
        if isinstance(x, (dict, list)):  # already parsed
            return x
        return json.loads(x)
    except Exception:
        return {}
    
def add_vwap_robusto(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["vwap"] = np.nan
        return df

    df = df.sort_values("ts").copy()
    # VWAP crudo por sesión (PV/V acumulado)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    v = df["volume"].fillna(0.0)
    cum_pv = (tp * v).cumsum()
    cum_v = v.cumsum()
    vwap_calc = (cum_pv / cum_v.replace(0, np.nan)).astype(float)

    # VWAP reconstruido desde dist_vwap_bps (cuando esté disponible)
    vwap_from_dist = None
    if "dist_vwap_bps" in df.columns:
        with np.errstate(invalid="ignore"):
            vwap_from_dist = df["close"] / (1.0 + (df["dist_vwap_bps"].astype(float) / 10000.0))

    # Fusionar: usa reconstruido si existe; si no, el crudo; y rellena huecos con el crudo
    df["vwap"] = (vwap_from_dist if vwap_from_dist is not None else vwap_calc)
    if vwap_from_dist is not None:
        df["vwap"] = df["vwap"].where(df["vwap"].notna(), vwap_calc)

    return df


def _reasons_to_str(r):
    if r is None:
        return ""
    if isinstance(r, list):
        return ", ".join(map(str, r))
    # duckdb may give us a stringified array
    try:
        rr = json.loads(r)
        if isinstance(rr, list):
            return ", ".join(map(str, rr))
    except Exception:
        pass
    return str(r)

# ------------------ Cached connection/data ------------------
@st.cache_resource
def get_conn(db_path: str):
    # Resource cache for non-serializable objects (connections). Read-only to avoid locks.
    return duckdb.connect(db_path, read_only=True)

@st.cache_data(ttl=30)
def list_symbols(db_path: str) -> list[str]:
    try:
        con = get_conn(db_path)
        df = con.execute("SELECT DISTINCT symbol FROM ohlcv_1m ORDER BY symbol").df()
        return df["symbol"].tolist()
    except Exception as e:
        st.error(f"Error loading symbols: {e}")
        return []

@st.cache_data(ttl=15)
def last_session_for_symbol(db_path: str, symbol: str) -> date | None:
    try:
        con = get_conn(db_path)
        df = con.execute(
            "SELECT max(ts) AS max_ts FROM ohlcv_1m WHERE symbol = ?", [symbol]
        ).df()
        if df.empty or pd.isna(df.loc[0, "max_ts"]):
            return None
        ts_utc = pd.to_datetime(df.loc[0, "max_ts"], utc=True)
        return ts_utc.tz_convert(ET).date()
    except Exception:
        return None

@st.cache_data(ttl=15)
def load_intraday_rth(db_path: str, symbol: str, session_et: date) -> pd.DataFrame:
    try:
        t0_utc, t1_utc = rth_bounds_utc(session_et)
        con = get_conn(db_path)
        q = """
        SELECT symbol, ts, open, high, low, close, volume, dist_vwap_bps
        FROM ohlcv_1m
        WHERE symbol = ?
          AND ts BETWEEN ? AND ?
        ORDER BY ts
        """
        df = con.execute(q, [symbol, t0_utc, t1_utc]).df()
        if df.empty:
            return df

        # Ensure tz-aware UTC
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        # Compute or reconstruct VWAP
        if "dist_vwap_bps" not in df.columns or df["dist_vwap_bps"].isna().all():
            tp = (df["high"] + df["low"] + df["close"]) / 3.0
            v = df["volume"].fillna(0.0)
            cum_pv = (tp * v).cumsum()
            cum_v = v.cumsum().replace(0, np.nan)
            df["vwap"] = (cum_pv / cum_v).astype(float)
        else:
            with np.errstate(invalid="ignore"):
                df["vwap"] = df["close"] / (1.0 + (df["dist_vwap_bps"].astype(float) / 10000.0))

        # Overlays
        df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

        # ET-naive column for plotting
        df["ts"] = pd.to_datetime(df["ts"], utc=True)

        # VWAP robusto (reconstruido + crudo como fallback)
        df = add_vwap_robusto(df)

        # Overlays
        df["ema9"]  = df["close"].ewm(span=9,  adjust=False).mean()
        df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()

        # ET-naive column for plotting
        df["ts_plot"] = to_et_naive(df["ts"])
        df["ts_plot"] = _ensure_dt64ns(df["ts_plot"])   # <— add this line

        return df
    except Exception as e:
        st.error(f"Error loading intraday data: {e}")
        return pd.DataFrame()
    
def _ensure_dt64ns(x: pd.Series | pd.DatetimeIndex) -> pd.Series:
    s = pd.to_datetime(x, errors="coerce")            # supports Series/Index
    if isinstance(s, pd.DatetimeIndex):
        s = pd.Series(s)
    return s.astype("datetime64[ns]")

    
def _normalize_signals_ts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer asof_utc (may be 'Z' or ±HH:MM), else ts_utc.
    Produces:
      - event_ts_utc (tz-aware UTC)
      - ts_plot (ET-naive) for plotting
    """
    if df.empty:
        return df

    # Choose the best source per row
    src = df.get("asof_utc")
    if src is None:
        ts_src = df["ts_utc"].astype(str)
    else:
        # use asof_utc where non-empty, otherwise ts_utc
        ts_src = np.where(
            src.astype(str).str.len() > 0,
            src.astype(str),
            df["ts_utc"].astype(str)
        )

    # Parse to UTC; may return DatetimeIndex
    parsed = pd.to_datetime(ts_src, utc=True, errors="coerce")

    # Ensure it's a Series aligned to df.index
    if isinstance(parsed, pd.DatetimeIndex):
        event_ts_utc = pd.Series(parsed, index=df.index)
    else:
        event_ts_utc = parsed

    df["event_ts_utc"] = event_ts_utc
    df["ts_plot"] = to_et_naive(event_ts_utc)
    return df.sort_values("event_ts_utc")


# ---------- Signals (CatBoost) ----------
@st.cache_data(ttl=10)
def load_signals(db_path: str, symbol: str, session_et: date) -> pd.DataFrame:
    sig_db = os.getenv("SIGNALS_DB", r"D:\AARCH\DBs\signals.duckdb")
    if not os.path.exists(sig_db):
        return pd.DataFrame()

    try:
        con = duckdb.connect(sig_db, read_only=True)
        # Pull by session to keep it small; we’ll filter by time after normalization.
        session_str = session_et.strftime("%Y-%m-%d")
        df = con.execute("""
            SELECT ts_utc, asof_utc, symbol, session,
                   decision, p_up, p_down, p_flat, margin, reasons, features
            FROM signals
            WHERE symbol = ? AND session = ?
            ORDER BY ts_utc
        """, [symbol, session_str]).df()
        con.close()
        if df.empty:
            return df

        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df["ts_plot"] = to_et_naive(df["ts_utc"])
        df["ts_plot"] = _ensure_dt64ns(df["ts_plot"])   # <— add this line
        df = _normalize_signals_ts(df)

        # Keep only RTH window for the chart tab
        t0, t1 = rth_bounds_utc(session_et)
        df = df[(df["event_ts_utc"] >= t0) & (df["event_ts_utc"] <= t1)]
        return df
    except Exception as e:
        st.error(f"Error loading signals: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=7)
def load_signals_all(session_et: date, full_day: bool = True) -> pd.DataFrame:
    if not os.path.exists(SIGNALS_DB):
        return pd.DataFrame()
    try:
        con = duckdb.connect(SIGNALS_DB, read_only=True)
        session_str = session_et.strftime("%Y-%m-%d")
        df = con.execute("""
            SELECT ts_utc, asof_utc, symbol, session,
                   decision, p_up, p_down, p_flat, margin, reasons, features
            FROM signals
            WHERE session = ?
            ORDER BY ts_utc
        """, [session_str]).df()
        con.close()
        if df.empty:
            return df
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        df = _normalize_signals_ts(df)
        if not full_day:
            t0, t1 = rth_bounds_utc(session_et)
            df = df[(df["event_ts_utc"] >= t0) & (df["event_ts_utc"] <= t1)]
        return df
    except Exception as e:
        st.error(f"Error loading all signals: {e}")
        return pd.DataFrame()


    



def build_leaderboard(sig: pd.DataFrame) -> pd.DataFrame:
    if sig.empty:
        return pd.DataFrame(columns=[
            "symbol","last_ts","last_decision","p_up","p_down","p_flat",
            "margin","signals_today","avg_margin","max_margin","age_min","confidence"
        ])
    try:
        # latest by event time
        idx = sig.groupby("symbol")["event_ts_utc"].idxmax()
        last = sig.loc[idx, ["symbol","event_ts_utc","decision","p_up","p_down","p_flat","margin"]].rename(
            columns={"event_ts_utc":"last_ts","decision":"last_decision"}
        )
        agg = sig.groupby("symbol").agg(
            signals_today=("decision","size"),
            avg_margin=("margin","mean"),
            max_margin=("margin","max"),
        ).reset_index()
        out = last.merge(agg, on="symbol", how="left")
        nowu = datetime.now(timezone.utc)
        out["age_min"] = ((nowu - out["last_ts"]).dt.total_seconds() / 60.0).round(1)
        out["confidence"] = out[["p_up","p_down","p_flat"]].max(axis=1)
        return out.sort_values(["margin","confidence"], ascending=[False, False]).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error building leaderboard: {e}")
        return pd.DataFrame()


# ------------------ Plotting ------------------
from pandas import merge_asof

def make_candles(
    df: pd.DataFrame,
    symbol: str,
    session_et: date,
    show_vwap: bool = True,
    show_ema: bool = True,
    signals: pd.DataFrame | None = None,
):
    """
    Render candles + VWAP/EMAs and (optionally) signal markers.
    Ensures both candle and signal timestamps are datetime64[ns] and sorted
    before merge_asof to avoid dtype errors.
    """
    try:
        def _to_ns(x):
            s = pd.to_datetime(x, errors="coerce")
            # handle DatetimeIndex or Series
            if isinstance(s, pd.DatetimeIndex):
                s = pd.Series(s)
            return s.astype("datetime64[ns]")

        # ---- Prep candles (ensure ts_plot exists & is ns) ----
        df = df.copy()
        if "ts_plot" not in df.columns:
            if "ts" in df.columns:
                df["ts"] = pd.to_datetime(df["ts"], utc=True)
                # fall back to ET-naive like elsewhere in the app
                df["ts_plot"] = df["ts"].dt.tz_convert(ET).dt.tz_localize(None)
            else:
                raise ValueError("DataFrame is missing 'ts_plot' and 'ts'.")
        df["ts_plot"] = _to_ns(df["ts_plot"])
        df = df.sort_values("ts_plot")

        # ---- Figure skeleton ----
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.72, 0.28], vertical_spacing=0.04,
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        )

        # Candles
        fig.add_trace(
            go.Candlestick(
                x=df["ts_plot"], open=df["open"], high=df["high"],
                low=df["low"], close=df["close"],
                name=f"{symbol} {session_et} (1m)",
            ),
            row=1, col=1,
        )

        # VWAP / EMAs
        if show_vwap and "vwap" in df.columns and not df["vwap"].isna().all():
            fig.add_trace(
                go.Scatter(x=df["ts_plot"], y=df["vwap"], mode="lines", name="VWAP"),
                row=1, col=1,
            )
        if show_ema:
            if "ema9" in df.columns:
                fig.add_trace(
                    go.Scatter(x=df["ts_plot"], y=df["ema9"], mode="lines", name="EMA(9)"),
                    row=1, col=1,
                )
            if "ema21" in df.columns:
                fig.add_trace(
                    go.Scatter(x=df["ts_plot"], y=df["ema21"], mode="lines", name="EMA(21)"),
                    row=1, col=1,
                )

        # Volume
        if "volume" in df.columns:
            fig.add_trace(
                go.Bar(x=df["ts_plot"], y=df["volume"], name="Volume"),
                row=2, col=1,
            )

        # ---- Overlay signals (snap to nearest candle) ----
        if signals is not None and not signals.empty:
            sig = signals[["ts_plot", "decision", "p_up", "p_down", "p_flat", "margin"]].copy()
            sig["ts_plot"] = _to_ns(sig["ts_plot"])
            sig = sig.dropna(subset=["ts_plot"]).sort_values("ts_plot")

            base = df[["ts_plot", "close"]].dropna().sort_values("ts_plot")

            snap = pd.merge_asof(
                sig, base, on="ts_plot",
                direction="nearest",
                tolerance=pd.Timedelta("2min"),
                allow_exact_matches=True,
            ).dropna(subset=["close"])

            # Color by decision
            color_map = {
                "up": "#16a34a", "down": "#dc2626", "flat": "#64748b",
                "long": "#16a34a", "short": "#dc2626",
                "queue_long": "#10b981", "queue_short": "#ef4444",
            }
            colors = [color_map.get(str(d), "#22c55e") for d in snap["decision"].astype(str)]

            hover = (
                "Time (ET): %{x}<br>"
                "Decision: %{customdata[0]}<br>"
                "p_up: %{customdata[1]:.3f} | p_down: %{customdata[2]:.3f} | p_flat: %{customdata[3]:.3f}<br>"
                "margin: %{customdata[4]:.3f}"
            )

            fig.add_trace(
                go.Scatter(
                    x=snap["ts_plot"],
                    y=snap["close"],
                    mode="markers",
                    name="Signals",
                    marker=dict(size=9, symbol="circle", line=dict(width=0.5, color="#111"), color=colors),
                    customdata=snap[["decision", "p_up", "p_down", "p_flat", "margin"]].values,
                    hovertemplate=hover,
                ),
                row=1, col=1,
            )

        fig.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            xaxis_rangeslider_visible=False,
            title=f"{symbol} — {session_et} (RTH 09:30–16:00 ET)",
        )
        return fig

    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return go.Figure()



# ------------------ UI ------------------
tab_chart, tab_board, tab_raw = st.tabs(["📈 Chart", "🏆 Leaderboard", "🗃️ Raw DB info"])

with tab_chart:
    topL, topR = st.columns([0.7, 0.3])
    with topL:
        db_path = find_latest_snapshot()
        st.caption(f"📦 DB snapshot: `{db_path}`  |  🔍 Signals DB: `{SIGNALS_DB}`")
    with topR:
        if st.button("🔄 Refrescar todo"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    symbols = list_symbols(db_path)
    if not symbols:
        st.error("No se encontraron símbolos en ohlcv_1m. ¿La DB/snapshot existe y tiene intradía?")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns([1.1, 1.0, 0.9, 0.9, 1.1])
    with c1:
        symbol = st.selectbox("Símbolo", symbols, index=0)
    with c2:
        default_session = last_session_for_symbol(db_path, symbol) or date.today()
        session_et = st.date_input("Sesión (ET)", value=default_session)
    with c3:
        show_vwap = st.checkbox("VWAP", value=True)
    with c4:
        show_ema = st.checkbox("EMA(9/21)", value=True)
    with c5:
        include_ah = st.checkbox("AH/PM (señales fuera RTH)", value=True)

    df = load_intraday_rth(db_path, symbol, session_et)
    if df.empty:
        st.warning("No hay datos para esa sesión/símbolo (RTH).")
        st.stop()

    # KPIs - Added safe float conversion
    try:
        o = float(df["open"].iloc[0])
        c = float(df["close"].iloc[-1])
        chg = (c / o - 1.0) * 100.0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Open", f"{o:.2f}")
        k2.metric("Last", f"{c:.2f}", f"{chg:+.2f}%")
        k3.metric("Máx/ Mín", f"{df['high'].max():.2f} / {df['low'].min():.2f}")
        k4.metric("Vol (sum)", f"{float(df['volume'].fillna(0).sum()):,.0f}")
    except (ValueError, IndexError) as e:
        st.error(f"Error calculating metrics: {e}")

    # Load and merge signals if requested
    signals = None
    if include_ah:
        signals = load_signals(db_path, symbol, session_et)

    fig = make_candles(df, symbol, session_et, show_vwap, show_ema, signals)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Fuente: ohlcv_1m (1m) · RTH 09:30—16:00 ET · VWAP intradía calculado o reconstruido desde dist_vwap_bps.")

with tab_board:
    st.subheader("🏆 Leaderboard — señales del día (todas)")
    session_for_board = st.date_input("Sesión (ET) para leaderboard", value=date.today(), key="board_session")
    full_day = st.checkbox("Usar día completo (no solo RTH)", value=True, key="board_full_day")

    sig_all = load_signals_all(session_for_board, full_day=full_day)
    if sig_all.empty:
        st.info("No hay señales en signals.duckdb para este día.")
    else:
        lb = build_leaderboard(sig_all)
        if lb.empty:
            st.warning("No se pudo construir el leaderboard.")
        else:
            # Filters - Added safety checks
            f1, f2, f3 = st.columns([1,1,1])
            with f1:
                max_margin = float(lb["margin"].max()) if not lb.empty and not lb["margin"].isna().all() else 1.0
                min_margin = st.slider("Min margin", 0.0, max_margin, 0.0, 0.01)
            with f2:
                decisions = st.multiselect("Decisiones", ["up","down","flat"], default=["up","down","flat"])
            with f3:
                max_age_val = float(lb["age_min"].max()) if not lb.empty and not lb["age_min"].isna().all() else 1.0
                max_age_val = max(1.0, max_age_val)
                max_age = st.slider("Max age (min)", 0.0, max_age_val, max_age_val, 1.0)

            filt = (lb["margin"] >= min_margin) & (lb["last_decision"].isin(decisions)) & (lb["age_min"] <= max_age)
            show = lb.loc[filt].copy()

            st.dataframe(
                show[["symbol","last_decision","margin","confidence","p_up","p_down","p_flat","signals_today","avg_margin","max_margin","last_ts","age_min"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "last_decision": st.column_config.Column("last_decision", help="Última decisión (por símbolo)"),
                    "margin": st.column_config.NumberColumn("margin", format="%.3f"),
                    "confidence": st.column_config.NumberColumn("confidence", format="%.3f", help="max(p_up, p_down, p_flat)"),
                    "p_up": st.column_config.NumberColumn("p_up", format="%.3f"),
                    "p_down": st.column_config.NumberColumn("p_down", format="%.3f"),
                    "p_flat": st.column_config.NumberColumn("p_flat", format="%.3f"),
                    "signals_today": st.column_config.NumberColumn("signals_today", help="Conteo de señales del día"),
                    "avg_margin": st.column_config.NumberColumn("avg_margin", format="%.3f"),
                    "max_margin": st.column_config.NumberColumn("max_margin", format="%.3f"),
                    "last_ts": st.column_config.DatetimeColumn("last_ts"),
                    "age_min": st.column_config.NumberColumn("age_min", format="%.1f", help="Minutos desde la última señal"),
                }
            )

            # Daily KPIs - Added safety checks
            try:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total señales (día)", f"{len(sig_all):,}")
                up_ct = (sig_all["decision"] == "up").sum()
                dn_ct = (sig_all["decision"] == "down").sum()
                fl_ct = (sig_all["decision"] == "flat").sum()
                k2.metric("up / down / flat", f"{up_ct} / {dn_ct} / {fl_ct}")
                k3.metric("Avg margin", f"{sig_all['margin'].mean():.3f}")
                k4.metric("Última señal (UTC)", sig_all["ts_utc"].max().strftime("%H:%M:%S"))
            except Exception as e:
                st.error(f"Error displaying KPIs: {e}")

with tab_raw:
    st.subheader("🗃️ Info de base de datos")
    db_path = find_latest_snapshot()
    st.write("Snapshot detectado:", db_path)
    if os.path.exists(SIGNALS_DB):
        st.write("Signals DB:", SIGNALS_DB)
    else:
        st.warning("Signals DB no existe en la ruta especificada.")