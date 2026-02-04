import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("CATBOOST_API_URL", "http://127.0.0.1:8500")
API_KEY = os.getenv("CATBOOST_API_KEY", "")

HEADERS = {"Content-Type": "application/json"}
if API_KEY:
    HEADERS["X-AARCH-Key"] = API_KEY


def score_batch(payload: dict | list) -> dict:
    """
    Llama al endpoint /score_batch con un payload (lista de jobs o un job).
    Devuelve el JSON de respuesta como dict.
    """
    url = f"{API_URL}/score_batch"
    r = requests.post(url, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def normalize_rows(resp: dict) -> tuple[pd.DataFrame, list[dict]]:
    """
    Toma la respuesta de /score_batch y:
      - Devuelve un DataFrame con 'hydrated' + 'prediction'
      - Y también la lista cruda de rows (para mostrar detalles)
    """
    rows = resp.get("rows", [])
    hyd = [r.get("hydrated", {}) for r in rows]
    pred = [r.get("prediction", {}) for r in rows]
    df_h = pd.DataFrame(hyd)
    df_p = pd.DataFrame(pred).rename(columns={"prediction": "pred_class"})
    df = pd.concat([df_h, df_p], axis=1)

    # margen simple: p_up - max(p_down, p_flat)
    for col in ("p_up", "p_down", "p_flat"):
        if col not in df:
            df[col] = 0.0
    df["p_margin"] = df["p_up"].fillna(0) - df[["p_down", "p_flat"]].fillna(0).max(axis=1)

    # asegurar columnas clave existan
    for c in ["symbol", "sector", "industry", "asof_utc", "dist_vwap_bps", "rsi_14"]:
        if c not in df:
            df[c] = None

    return df, rows


def load_results_from_url(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
