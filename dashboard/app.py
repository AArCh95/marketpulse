import os
import time
import json
import pytz
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime, time as dt_time, timezone, timedelta
from dotenv import load_dotenv

from api_client import score_batch, normalize_rows, load_results_from_url, load_json_file

load_dotenv()

st.set_page_config(page_title="MarketPulse — Panel", layout="wide")

# --------------------------
# Sidebar: Controles
# --------------------------
st.sidebar.title("⚙️ Controles")

mode = st.sidebar.radio(
    "Fuente de datos",
    ["A) Llamar API con payload local", "B) Subir/Pegar respuesta API", "C) URL JSON (opcional)"],
    index=0
)

# Umbrales/Gates
st.sidebar.markdown("### Umbrales / Gates")
max_stale_sec = int(st.sidebar.number_input("Max stale (segundos)", min_value=60, max_value=3600, value=600, step=60))
max_vwap_bps = float(st.sidebar.number_input("Máx |dist_vwap_bps|", min_value=0.0, max_value=1000.0, value=50.0, step=5.0))
min_margin = float(st.sidebar.number_input("Mín. p_margin para señal", min_value=0.0, max_value=1.0, value=0.10, step=0.01))

st.sidebar.markdown("---")
do_heatmap = st.sidebar.checkbox("Mostrar Heatmap por sector", value=True)
top_k = int(st.sidebar.number_input("Top señales a mostrar", min_value=5, max_value=200, value=30, step=5))

st.sidebar.markdown("---")
st.sidebar.caption("Zona horaria: America/Costa_Rica (CR)")

# --------------------------
# Helpers
# --------------------------
CR_TZ = pytz.timezone("America/Costa_Rica")
ET_TZ = pytz.timezone("America/New_York")

def now_utc():
    return datetime.now(timezone.utc)

def in_rth(now: datetime | None = None) -> bool:
    """Rango básico RTH (NYSE) 09:30–16:00 ET."""
    if now is None:
        now = datetime.now(ET_TZ)
    else:
        now = now.astimezone(ET_TZ)
    t = now.time()
    return dt_time(9, 30) <= t <= dt_time(16, 0)

def eval_reasons(row: pd.Series) -> list[str]:
    rs = []
    # RTH gate
    if not in_rth():
        rs.append("not_in_RTH")
    # Freshness gate
    ts = pd.to_datetime(row.get("asof_utc"), utc=True, errors="coerce")
    if pd.isna(ts):
        rs.append("stale_data")
    else:
        age = (now_utc() - ts).total_seconds()
        if age > max_stale_sec:
            rs.append("stale_data")
    # VWAP gate (si tenemos el feature)
    dvb = row.get("dist_vwap_bps")
    if pd.notna(dvb):
        try:
            if abs(float(dvb)) > max_vwap_bps:
                rs.append("away_from_VWAP")
        except Exception:
            pass
    # Margen mínimo
    pm = row.get("p_margin", 0.0)
    try:
        if float(pm) < min_margin:
            rs.append("low_margin")
    except Exception:
        rs.append("low_margin")
    return rs

def decide(row: pd.Series) -> str:
    reasons = eval_reasons(row)
    if len(reasons) == 0:
        # Acción según clase
        cls = row.get("pred_class")
        if cls == "up":
            return "buy"
        elif cls == "down":
            return "sell"
    return "hold"

# --------------------------
# Carga de datos según modo
# --------------------------
resp = None

if mode.startswith("A)"):
    st.sidebar.markdown("#### Payload local → /score_batch")
    payload_file = st.sidebar.text_input("Ruta payload JSON", value=str(os.path.join(os.getcwd(), "sample_payload.json")))
    colA1, colA2 = st.sidebar.columns([1,1])
    with colA1:
        do_call = st.button("Llamar API", type="primary")
    with colA2:
        st.write("")

    if do_call:
        try:
            payload = load_json_file(payload_file)
            resp = score_batch(payload)
            st.success("✅ API respondió OK")
        except Exception as e:
            st.error(f"❌ Error llamando API: {e}")

elif mode.startswith("B)"):
    st.sidebar.markdown("#### Pegar/Subir respuesta de /score_batch")
    up = st.sidebar.file_uploader("Subir JSON", type=["json"])
    txt = st.sidebar.text_area("...o pegar JSON aquí", height=200)
    if up:
        try:
            resp = json.load(up)
            st.success("✅ JSON subido OK")
        except Exception as e:
            st.error(f"❌ Error leyendo archivo: {e}")
    elif txt.strip():
        try:
            resp = json.loads(txt)
            st.success("✅ JSON pegado OK")
        except Exception as e:
            st.error(f"❌ JSON inválido: {e}")

else:
    st.sidebar.markdown("#### URL con JSON de resultados")
    default_url = os.getenv("RESULTS_JSON_URL", "")
    url = st.sidebar.text_input("URL", value=default_url)
    if st.sidebar.button("Cargar desde URL", type="primary"):
        try:
            resp = load_results_from_url(url)
            st.success("✅ Descargado OK")
        except Exception as e:
            st.error(f"❌ Error al descargar: {e}")

if resp is None:
    st.info("Carga datos con alguno de los modos en la barra lateral.")
    st.stop()

# --------------------------
# Normalización
# --------------------------
try:
    df, rows = normalize_rows(resp)
except Exception as e:
    st.error(f"❌ Error normalizando respuesta: {e}")
    st.stop()

if df.empty:
    st.warning("Respuesta vacía (sin filas).")
    st.stop()

# Decisiones + razones
df = df.copy()
df["decision"] = df.apply(decide, axis=1)
df["reasons"] = df.apply(lambda r: ", ".join(eval_reasons(r)), axis=1)

# KPIs
n = len(df)
n_buy = int((df["decision"] == "buy").sum())
n_sell = int((df["decision"] == "sell").sum())
n_hold = n - n_buy - n_sell

c1, c2, c3, c4 = st.columns(4)
c1.metric("Símbolos", n)
c2.metric("BUY", n_buy)
c3.metric("SELL", n_sell)
c4.metric("HOLD", n_hold)

st.markdown("---")

# Heatmap por sector (promedio p_up)
if do_heatmap:
    df_hm = df.copy()
    df_hm["sector"] = df_hm["sector"].fillna("Unknown")
    hm = (
        df_hm.groupby("sector", as_index=False)["p_up"]
        .mean(numeric_only=True)
        .rename(columns={"p_up": "avg_p_up"})
    )
    fig = px.imshow(
        hm.pivot_table(index="sector", values="avg_p_up", aggfunc="mean"),
        color_continuous_scale="Blues",
        aspect="auto",
        title="Promedio p_up por Sector"
    )
    st.plotly_chart(fig, use_container_width=True)

# Top señales por margen (filtradas a decisiones != hold si quieres)
top = (
    df.sort_values("p_margin", ascending=False)
      .head(top_k)
      .reset_index(drop=True)
)

st.subheader("Top señales por margen")
st.dataframe(
    top[[
        "symbol","pred_class","p_up","p_down","p_flat","p_margin",
        "sector","industry","asof_utc","dist_vwap_bps","rsi_14","decision","reasons"
    ]],
    use_container_width=True,
    hide_index=True
)

# Detalle por símbolo (cards)
st.markdown("---")
st.subheader("Detalle por símbolo")

def card(row: pd.Series):
    st.markdown(f"### {row['symbol']} · {row.get('sector','')} · {row.get('industry','')}")
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.write(f"**Pred:** {row.get('pred_class')}")
        st.write(f"**Decision:** {row.get('decision')}")
        st.write(f"**p_up:** {row.get('p_up'):.3f}")
        st.write(f"**p_margin:** {row.get('p_margin'):.3f}")
    with c2:
        st.write(f"**VWAP (bps):** {row.get('dist_vwap_bps')}")
        st.write(f"**RSI14:** {row.get('rsi_14')}")
        st.write(f"**asof_utc:** {row.get('asof_utc')}")
    with c3:
        st.write(f"**Razones/Gates:** {row.get('reasons')}")
    st.markdown("---")

for _, r in top.iterrows():
    card(r)
