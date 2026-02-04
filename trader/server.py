import os, json, hashlib, time, shutil, asyncio
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import logging
import duckdb
import pytz
from fastapi import FastAPI, Request, Header, HTTPException
from dotenv import load_dotenv

# --- Alpaca (alpaca-py) ---
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.models import Position

load_dotenv()

# ------------ robust env parsing helpers ------------
def _env_raw(key: str) -> Optional[str]:
    v = os.getenv(key)
    if v is None:
        return None
    return v.split("#", 1)[0].strip()

def _env_bool(key: str, default: bool = False) -> bool:
    v = _env_raw(key)
    return default if v is None else v.lower() in {"1", "true", "yes", "y", "on"}

def _env_num(key: str, default: Optional[float] = None) -> Optional[float]:
    v = _env_raw(key)
    if v is None:
        return default
    if v.endswith("%"):
        try:
            return float(v[:-1].strip()) / 100.0
        except Exception:
            return default
    try:
        return float(v)
    except Exception:
        return default

def _env_int(key: str, default: Optional[int] = None) -> Optional[int]:
    v = _env_raw(key)
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default

def _pct_value(s: Optional[str], default: float) -> float:
    """
    Convert env string to fraction:
      - "2%"   -> 0.02
      - "2"    -> 0.02   (treat >1 as percent)
      - "0.02" -> 0.02
      - None/parse-fail -> default
    Strips inline comments automatically via _env_raw().
    """
    if s is None:
        return default
    s = s.strip()
    try:
        if s.endswith("%"):
            return float(s[:-1].strip()) / 100.0
        v = float(s)
        return v / 100.0 if v > 1 else v
    except Exception:
        return default

def _mask(s: Optional[str], keep: int = 4) -> str:
    if not s:
        return "None"
    if len(s) <= keep:
        return "*" * len(s)
    return s[:keep] + "*" * max(0, len(s) - (keep + 2)) + s[-2:]

# ------------ Config (env) ------------
DB_PATH          = _env_raw("TRADER_DB") or r"D:\AARCH\DBs\trader.duckdb"
SIGNALS_DB       = _env_raw("SIGNALS_DB") or r"D:\AARCH\DBs\signals.duckdb"
API_TOKEN        = _env_raw("TRADER_WEBHOOK_TOKEN") or "change-me"

ALPACA_KEY       = _env_raw("APCA_API_KEY_ID") or _env_raw("ALPACA_KEY_ID")
ALPACA_SECRET    = _env_raw("APCA_API_SECRET_KEY") or _env_raw("ALPACA_SECRET_KEY")
ALPACA_PAPER     = _env_bool("ALPACA_PAPER", True)

# --- Fallback unit sizing ---
USD_PER_TRADE    = float(_env_num("USD_PER_TRADE", 100.0))
PCT_PER_TRADE    = _pct_value(_env_raw("PCT_PER_TRADE"), 0.02)

# --- Risk / TP / SL ---
TP_PCT           = _pct_value(_env_raw("TP_PCT"), 0.03)
SL_PCT           = _pct_value(_env_raw("SL_PCT"), 0.02)
ALLOW_SHORTS     = _env_bool("ALLOW_SHORTS", True)

# --- Dynamic sizing knobs ---
MAX_WEIGHT_PCT   = _pct_value(_env_raw("MAX_WEIGHT_PCT"), 0.004)   # default 0.4% (first day)
MIN_DELTA_WEIGHT = _pct_value(_env_raw("MIN_DELTA_WEIGHT"), 0.001) # 0.1%
MIN_NOTIONAL     = float(_env_num("MIN_NOTIONAL", 1.00))

# --- Batch & pacing ---
MAX_NAMES_PER_BATCH = int(_env_int("MAX_NAMES_PER_BATCH", 0) or 0)
ORDER_THROTTLE_MS   = int(_env_int("ORDER_THROTTLE_MS", 0) or 0)

# --- Freshness & cooldowns ---
STALE_SEC_MAX           = int(_env_int("STALE_SEC_MAX", 0) or 0)              # 0 = disabled
PER_SYMBOL_COOLDOWN_MIN = int(_env_int("PER_SYMBOL_COOLDOWN_MIN", 0) or 0)
DEDUPE_WINDOW_MIN       = int(_env_int("DEDUPE_WINDOW_MIN", 0) or 0)          # reserved (not used)
SELL_BAN_MIN            = int(_env_int("SELL_BAN_MIN", 0) or 0)               # 0 = disabled

# --- Daily spend cap ---
DAILY_SPEND_CAP = (_env_raw("DAILY_SPEND_CAP") or "").strip()  # "", "50", "50$", "20%", "0.2"
DAILY_SPEND_TZ  = _env_raw("DAILY_SPEND_TZ") or "America/New_York"

# ------------ Snapshot config ------------
SNAP_DIR        = _env_raw("TRADER_SNAPSHOT_DIR") or os.path.join(os.path.dirname(DB_PATH), "snapshots")
SNAP_KEEP       = int(_env_int("TRADER_SNAPSHOT_KEEP", 12) or 12)
SNAP_POINTER    = _env_raw("TRADER_SNAPSHOT_POINTER") or os.path.join(os.path.dirname(DB_PATH), "CURRENT_SNAPSHOT_TRADER.txt")
SNAP_INTERVAL   = int(_env_int("TRADER_SNAPSHOT_INTERVAL_SEC", 0) or 0)   # 0 = disabled
SNAP_ON_INGEST  = _env_bool("TRADER_SNAPSHOT_ON_INGEST", True)

# ------------ Sell ban config ------------
SELL_BAN_FILE   = r"D:\AARCH\trader\sell_bans.json"

# ------------ logging setup & config dump ------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("server")

def _log_effective_config():
    cfg = {
        # Trading mode / connectivity
        "ALPACA_PAPER": ALPACA_PAPER,
        "APCA_API_KEY_ID": _mask(ALPACA_KEY),
        "APCA_API_SECRET_KEY": _mask(ALPACA_SECRET),

        # Risk / sizing (fractions; e.g., 0.03 = 3%)
        "TP_PCT": TP_PCT,
        "SL_PCT": SL_PCT,
        "ALLOW_SHORTS": ALLOW_SHORTS,

        # Budget & pacing
        "DAILY_SPEND_CAP": DAILY_SPEND_CAP or "(disabled)",
        "DAILY_SPEND_TZ": DAILY_SPEND_TZ,
        "MAX_NAMES_PER_BATCH": MAX_NAMES_PER_BATCH,
        "ORDER_THROTTLE_MS": ORDER_THROTTLE_MS,

        # Cooldown / freshness
        "PER_SYMBOL_COOLDOWN_MIN": PER_SYMBOL_COOLDOWN_MIN,
        "SELL_BAN_MIN": SELL_BAN_MIN,
        "STALE_SEC_MAX": STALE_SEC_MAX,

        # Sizing caps & fallbacks
        "MAX_WEIGHT_PCT": MAX_WEIGHT_PCT,   # 0.004 = 0.4%
        "MIN_DELTA_WEIGHT": MIN_DELTA_WEIGHT,
        "MIN_NOTIONAL": MIN_NOTIONAL,
        "PCT_PER_TRADE": PCT_PER_TRADE,     # 0.02 = 2% of cash
        "USD_PER_TRADE": USD_PER_TRADE,

        # Snapshots
        "TRADER_SNAPSHOT_DIR": SNAP_DIR,
        "TRADER_SNAPSHOT_KEEP": SNAP_KEEP,
        "TRADER_SNAPSHOT_POINTER": SNAP_POINTER,
        "TRADER_SNAPSHOT_INTERVAL_SEC": SNAP_INTERVAL,
        "TRADER_SNAPSHOT_ON_INGEST": SNAP_ON_INGEST,
    }
    width = max(len(k) for k in cfg)
    log.info("=== EFFECTIVE TRADER CONFIG ===")
    for k, v in cfg.items():
        log.info(f"{k.ljust(width)} = {v}")
    log.info("================================")

# ------------ Alpaca ------------
if not ALPACA_KEY or not ALPACA_SECRET:
    raise RuntimeError("Missing Alpaca credentials (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")

trading_client: TradingClient = TradingClient(
    api_key=ALPACA_KEY, secret_key=ALPACA_SECRET, paper=ALPACA_PAPER
)

ET = pytz.timezone("America/New_York")
app = FastAPI(title="AARCH Trader")

# print config every time the server boots
@app.on_event("startup")
def _print_config_on_startup():
    _log_effective_config()

# snapshot loop (optional)
if SNAP_INTERVAL > 0:
    @app.on_event("startup")
    async def _start_snapshotter():
        async def _loop():
            while True:
                try:
                    _publish_snapshot()
                except Exception as e:
                    log.warning("snapshot error: %s", e)
                await asyncio.sleep(SNAP_INTERVAL)
        asyncio.create_task(_loop())

# ------------ Helpers ------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def _auth_or_401(x_api_key: Optional[str]):
    if API_TOKEN and (x_api_key != API_TOKEN):
        raise HTTPException(401, "Unauthorized")

def _hash_alert(payload: Dict[str, Any]) -> str:
    m = hashlib.sha256()
    m.update(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    return m.hexdigest()[:16]

def _ensure_schema():
    con = duckdb.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS alerts(
            id TEXT PRIMARY KEY,
            received_ts TIMESTAMPTZ DEFAULT now(),
            session TEXT,
            symbol TEXT,
            body JSON,
            decision TEXT,
            direction TEXT,
            in_rth BOOLEAN,
            change BOOLEAN,
            notify BOOLEAN,
            price DOUBLE,
            queued_for_open BOOLEAN,
            queued_consumed BOOLEAN
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS orders(
            id TEXT PRIMARY KEY,
            created_ts TIMESTAMPTZ DEFAULT now(),
            symbol TEXT,
            side TEXT,
            qty BIGINT,
            notional DOUBLE,
            order_type TEXT,
            tif TEXT,
            tp_pct DOUBLE,
            sl_pct DOUBLE,
            alpaca_order_id TEXT,
            status TEXT,
            client_order_id TEXT
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS open_queue(
            id TEXT PRIMARY KEY,
            enqueued_ts TIMESTAMPTZ DEFAULT now(),
            symbol TEXT,
            side TEXT,
            notional DOUBLE,
            qty BIGINT,
            tif TEXT,
            body JSON
        );
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS positions(
            snapshot_ts TIMESTAMPTZ,
            symbol TEXT,
            qty DOUBLE,
            avg_entry DOUBLE,
            market_price DOUBLE,
            market_value DOUBLE,
            unrealized_pl DOUBLE,
            unrealized_plpc DOUBLE,
            realized_pl_day DOUBLE
        );
    """)
    con.close()

def _publish_snapshot() -> Dict[str, Any]:
    try:
        os.makedirs(SNAP_DIR, exist_ok=True)
        if not os.path.exists(DB_PATH):
            return {"ok": False, "error": f"missing_primary:{DB_PATH}"}
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        snap_path = os.path.join(SNAP_DIR, f"trader_{ts_str}.duckdb")
        last_err = None
        for attempt in range(3):
            try:
                shutil.copy2(DB_PATH, snap_path)
                last_err = None
                break
            except Exception as e:
                last_err = e
                time.sleep(0.25 * (attempt + 1))
        if last_err:
            return {"ok": False, "error": f"copy_failed:{last_err}"}
        tmp_ptr = SNAP_POINTER + ".tmp"
        with open(tmp_ptr, "w", encoding="utf-8") as f:
            f.write(snap_path)
        os.replace(tmp_ptr, SNAP_POINTER)
        snaps = sorted(Path(SNAP_DIR).glob("trader_*.duckdb"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in snaps[SNAP_KEEP:]:
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass
        return {"ok": True, "snapshot": snap_path, "kept": min(len(snaps), SNAP_KEEP)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def _calc_qty(notional: float, price: Optional[float]) -> int:
    if notional <= 0 or not price:
        return 0
    return max(1, int(notional // price))

def _extract(j: Dict[str, Any]) -> Tuple[str, str, str, bool, Optional[float], bool, bool, str, Optional[float]]:
    sym        = (j.get("symbol") or j.get("ticker") or "").upper()
    model      = j.get("model") or {}
    decision   = (model.get("decision") or j.get("decision") or "").lower()
    direction  = (j.get("direction") or ("up" if decision in ("long","queue_long") else "down" if decision in ("short","queue_short") else "")).lower()
    basics     = j.get("basics") or {}
    price      = basics.get("close_d") or (j.get("price") or {}).get("current")
    chg  = j.get("change")
    if chg is None:
        chg = j.get("Change")
    flags = {"change": bool(chg or (j.get("price") or {}).get("change")), "notify": bool(j.get("notify"))}
    session    = basics.get("alertType") or j.get("session") or "RTH"
    in_rth     = bool(j.get("in_rth", session == "RTH"))
    usd = j.get("usd") or j.get("notional")
    return sym, decision, direction, in_rth, (float(price) if price is not None else None), bool(flags["change"]), bool(flags["notify"]), session, (float(usd) if usd is not None else None)

def _parse_daily_cap(cap_raw: str, cash_available: float) -> float:
    if not cap_raw:
        return 0.0
    s = cap_raw.strip().replace("$", "")
    try:
        if s.endswith("%"):
            pct = float(s.strip("%")) / 100.0
            return max(0.0, pct * cash_available)
        v = float(s)
        if v <= 1.0:  # treat "0.2" as 20%
            return max(0.0, v * cash_available)
        return max(0.0, v)
    except Exception:
        return 0.0
    
def should_exit(decision: str, intent: str) -> bool:
    if intent in ("exit","short","queue_short"): 
        return True
    if decision in ("short","queue_short") and not ALLOW_SHORTS:
        return True
    return False


def _et_day_bounds(now_utc: datetime, tz_name: str) -> Tuple[str, str]:
    tz = pytz.timezone(tz_name)
    now_local = now_utc.astimezone(tz)
    start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc).isoformat()
    end_utc   = end_local.astimezone(timezone.utc).isoformat()
    return start_utc, end_utc

# ------------ Sell ban helpers ------------
def _load_sell_bans() -> Dict[str, str]:
    """Load sell bans from JSON file. Returns {symbol: banned_until_iso}"""
    if not os.path.exists(SELL_BAN_FILE):
        return {}
    try:
        with open(SELL_BAN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_sell_bans(bans: Dict[str, str]):
    """Save sell bans to JSON file with automatic cleanup of expired entries."""
    now = utcnow()
    # Clean expired bans before saving
    cleaned = {sym: until for sym, until in bans.items()
               if datetime.fromisoformat(until.replace("Z", "+00:00")) > now}
    try:
        with open(SELL_BAN_FILE, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, indent=2)
    except Exception as e:
        log.warning(f"Failed to save sell bans: {e}")

def _is_symbol_banned(symbol: str) -> bool:
    """Check if symbol is currently banned from buying."""
    if SELL_BAN_MIN <= 0:
        return False
    bans = _load_sell_bans()
    if symbol not in bans:
        return False
    try:
        banned_until = datetime.fromisoformat(bans[symbol].replace("Z", "+00:00"))
        is_banned = utcnow() < banned_until
        if is_banned:
            log.info(f"{symbol} is currently banned until {banned_until}")
        return is_banned
    except Exception:
        return False

def _ban_symbol_after_sell(symbol: str):
    """Record a sell ban for the symbol."""
    if SELL_BAN_MIN <= 0:
        log.warning(f"Ban not recorded for {symbol}: SELL_BAN_MIN={SELL_BAN_MIN} (disabled)")
        return
    bans = _load_sell_bans()
    banned_until = (utcnow() + timedelta(minutes=SELL_BAN_MIN)).isoformat()
    bans[symbol] = banned_until
    log.info(f"Banning {symbol} from re-buy until {banned_until}")
    _save_sell_bans(bans)

# ------------ API ------------
@app.post("/submit_open_queue")
async def api_submit_open_queue(x_api_key: Optional[str] = Header(None)):
    _auth_or_401(x_api_key)
    return submit_opg_from_queue()

@app.post("/snapshot")
async def api_snapshot(x_api_key: Optional[str] = Header(None)):
    _auth_or_401(x_api_key)
    return _publish_snapshot()

@app.post("/ingest")
async def ingest(request: Request, x_api_key: Optional[str] = Header(None)):
    _auth_or_401(x_api_key)

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON")

    raw_items: List[Dict[str, Any]] = body if isinstance(body, list) else [body]
    _ensure_schema()
    con = duckdb.connect(DB_PATH)
    results: List[Dict[str, Any]] = []

    # --- account snapshot (once) for sizing ---
    try:
        acct = trading_client.get_account()
        def _f(x, d=0.0):
            try: return float(x)
            except Exception: return d
        buying_power    = _f(getattr(acct, "buying_power", 0))
        portfolio_value = _f(getattr(acct, "portfolio_value", 0))
        cash_available  = _f(getattr(acct, "cash", 0))
    except Exception:
        buying_power, portfolio_value, cash_available = 0.0, 0.0, 0.0

    # --- Daily spend cap (BUY only) ---
    spend_cap_usd = _parse_daily_cap(DAILY_SPEND_CAP, cash_available)
    start_utc, end_utc = _et_day_bounds(utcnow(), DAILY_SPEND_TZ)
    spent_today = 0.0
    if spend_cap_usd > 0:
        try:
            q = """
                SELECT coalesce(SUM(notional),0) FROM orders
                WHERE side='buy' AND created_ts BETWEEN ? AND ?
            """
            spent_today = float(con.execute(q, [start_utc, end_utc]).fetchone()[0] or 0.0)
        except Exception:
            spent_today = 0.0
    remaining_cap = max(0.0, spend_cap_usd - spent_today)

    # --- MAX_NAMES_PER_BATCH: keep top-K by opportunity_score if provided ---
    def _score(x: Dict[str, Any]) -> float:
        try: return float(x.get("opportunity_score", 0.0))
        except Exception: return 0.0
    items = sorted(raw_items, key=_score, reverse=True)
    if MAX_NAMES_PER_BATCH > 0 and len(items) > MAX_NAMES_PER_BATCH:
        items = items[:MAX_NAMES_PER_BATCH]

    # --- Iterate items ---
    for j in items:
        # Stale skip
        if STALE_SEC_MAX > 0:
            try:
                if int(j.get("stale_sec", 0)) > STALE_SEC_MAX:
                    results.append({"ok": True, "symbol": j.get("symbol"), "action": "skipped_stale"})
                    continue
            except Exception:
                pass

        sym, decision, direction, in_rth, price, change, notify, session, usd_override = _extract(j)
        if not sym:
            results.append({"ok": False, "error": "missing_symbol"})
            continue
        if not change and j.get("Change") is not None:
            change = bool(j.get("Change"))

        # Idempotency: hash full payload (you can switch to a stable key if you want stronger dedupe)
        rec_id = _hash_alert(j)
        already = con.execute("SELECT 1 FROM alerts WHERE id = ?", [rec_id]).fetchone()
        if already:
            results.append({"ok": True, "symbol": sym, "action": "duplicate"})
            continue

        queued_for_open = (not in_rth) and decision in ("queue_long", "queue_short", "long", "short")
        con.execute("""
            INSERT INTO alerts(id, received_ts, session, symbol, body, decision, direction,
                               in_rth, change, notify, price, queued_for_open, queued_consumed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            rec_id, utcnow().isoformat(), session, sym, json.dumps(j), decision, direction,
            in_rth, change, notify, (float(price) if price is not None else None),
            queued_for_open, False
        ])

        # Gate
        if not bool(change or notify):
            results.append({"ok": True, "symbol": sym, "action": "ignored_no_change"})
            continue

        # Per-symbol cooldown (skip if recent same-side order)
        if PER_SYMBOL_COOLDOWN_MIN > 0:
            try:
                since_utc = (utcnow() - timedelta(minutes=PER_SYMBOL_COOLDOWN_MIN)).isoformat()
                side_check = "buy" if decision in ("long","queue_long") else "sell" if decision in ("short","queue_short") else None
                if side_check:
                    q = """
                        SELECT 1 FROM orders
                        WHERE symbol=? AND side=? AND created_ts >= ?
                        LIMIT 1
                    """
                    recent = con.execute(q, [sym, side_check, since_utc]).fetchone()
                    if recent:
                        results.append({"ok": True, "symbol": sym, "action": "cooldown_skip"})
                        continue
            except Exception:
                pass

        # Dynamic hints
        intent_raw = (j.get("intent") or "accumulate").lower()
        intent = "exit" if intent_raw in ("short", "queue_short") else intent_raw


        max_w_env = MAX_WEIGHT_PCT

        try:
            max_w = float(j.get("max_weight_pct", max_w_env))
        except Exception:
            max_w = max_w_env
        max_w = max(0.0, min(max_w, max_w_env))  # clamp to env

        # Parse target weight safely
        tgt_w_raw = j.get("target_weight_pct")
        try:
            tgt_w = float(tgt_w_raw) if tgt_w_raw is not None else None
        except Exception:
            tgt_w = None

        try:
            # ---- LONG / ACCUMULATE (fractional BUY, DAY) ----
            if decision in ("long", "queue_long"):
                side = "buy"

                # Check sell ban before allowing any BUY
                if _is_symbol_banned(sym):
                    results.append({"ok": True, "symbol": sym, "action": "sell_ban_active"})
                    continue

                if intent in ("exit", "short", "queue_short"):
                    # Exit any existing long using notional=market_value
                    cur_mv = 0.0
                    try:
                        for p in trading_client.get_all_positions():
                            if p.symbol.upper() == sym:
                                cur_mv = float(p.market_value or 0.0)
                                break
                    except Exception:
                        pass
                    if cur_mv <= 0:
                        results.append({"ok": True, "symbol": sym, "action": "no_position_to_exit"})
                        continue
                    # Daily cap: SELL does not count toward spend cap
                    tif_exit = TimeInForce.DAY if in_rth else TimeInForce.OPG
                    req = MarketOrderRequest(
                        symbol=sym, notional=round(abs(cur_mv), 2),
                        side=OrderSide.SELL, time_in_force=tif_exit
                    )
                    order = trading_client.submit_order(req)
                    tif_str = "DAY" if in_rth else "OPG"
                    con.execute("""
                        INSERT INTO orders(id, symbol, side, qty, notional, order_type, tif, tp_pct, sl_pct,
                                           alpaca_order_id, status, client_order_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [rec_id, sym, "sell", 0, abs(cur_mv), "market", tif_str, TP_PCT, SL_PCT,
                          getattr(order, "id", None), "submitted", getattr(order, "client_order_id", None)])
                    _ban_symbol_after_sell(sym)
                    results.append({"ok": True, "symbol": sym, "action": f"exit_all_{tif_str.lower()}"})
                    if ORDER_THROTTLE_MS > 0:
                        await asyncio.sleep(ORDER_THROTTLE_MS / 1000.0)
                    continue

                # Determine notional from target weight or fallbacks
                notional = None
                if tgt_w is not None and portfolio_value > 0:
                    # Desired weight cannot exceed payload cap OR env cap
                    desired = max(0.0, min(tgt_w, max_w))

                    # Current weight from live positions
                    cur_mv = 0.0
                    try:
                        for p in trading_client.get_all_positions():
                            if p.symbol.upper() == sym:
                                cur_mv = float(p.market_value or 0.0)
                                break
                    except Exception:
                        pass

                    cur_w = abs(cur_mv) / portfolio_value if portfolio_value > 0 else 0.0

                    # Hard guard: if already at/above desired (or above env cap), skip
                    if cur_w >= desired or cur_w >= MAX_WEIGHT_PCT:
                        results.append({
                            "ok": True, "symbol": sym, "action": "at_or_above_target",
                            "current_w": round(cur_w, 6), "target_w": round(desired, 6)
                        })
                        continue

                    delta_w = desired - cur_w
                    if delta_w <= MIN_DELTA_WEIGHT:
                        results.append({
                            "ok": True, "symbol": sym, "action": "below_min_delta",
                            "current_w": round(cur_w, 6), "target_w": round(desired, 6),
                            "min_delta": MIN_DELTA_WEIGHT
                        })
                        continue

                    # Convert delta weight to notional and clamp
                    notional = round(delta_w * portfolio_value, 2)

                    # Never allow per-name notional to push weight past env cap
                    max_additional = max(0.0, (MAX_WEIGHT_PCT - cur_w) * portfolio_value)
                    notional = min(notional, max_additional)

                    # Cap by available cash
                    if cash_available > 0:
                        notional = min(notional, cash_available)

                if notional is None:
                    # fallback sizing (kept as-is)
                    if usd_override is not None:
                        notional = round(max(MIN_NOTIONAL, float(usd_override)), 2)
                    else:
                        pct = PCT_PER_TRADE
                        base_cash = cash_available
                        notional = round(max(MIN_NOTIONAL, base_cash * pct), 2)


                # Apply daily spend cap
                if spend_cap_usd > 0:
                    if remaining_cap <= 0:
                        results.append({"ok": True, "symbol": sym, "action": "daily_cap_reached"})
                        continue
                    if notional > remaining_cap:
                        notional = round(remaining_cap, 2)

                if notional < MIN_NOTIONAL:
                    results.append({"ok": False, "symbol": sym, "error": "insufficient_cash_or_small_delta", "notional": notional})
                    continue

                # Submit fractional BUY (DAY)
                req = MarketOrderRequest(
                    symbol=sym,
                    notional=round(notional, 2),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
                order = trading_client.submit_order(req)
                con.execute("""
                    INSERT INTO orders(id, symbol, side, qty, notional, order_type, tif, tp_pct, sl_pct,
                                       alpaca_order_id, status, client_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    rec_id, sym, side, 0, float(notional), "market", "DAY", TP_PCT, SL_PCT,
                    getattr(order, "id", None), "submitted", getattr(order, "client_order_id", None)
                ])
                results.append({"ok": True, "symbol": sym, "action": f"order_{side}_DAY", "notional": float(notional)})

                # Update local daily remaining for this request
                if spend_cap_usd > 0:
                    remaining_cap = max(0.0, remaining_cap - float(notional))

                if ORDER_THROTTLE_MS > 0:
                    await asyncio.sleep(ORDER_THROTTLE_MS / 1000.0)

            # ---- SHORT / EXIT or open short (qty-based) ----
            elif decision in ("short", "queue_short"):
                if intent == "exit" or not ALLOW_SHORTS:
                    cur_mv = 0.0
                    try:
                        for p in trading_client.get_all_positions():
                            if p.symbol.upper() == sym:
                                cur_mv = float(p.market_value or 0.0)
                                break
                    except Exception:
                        pass
                    if cur_mv <= 0:
                        results.append({"ok": True, "symbol": sym, "action": "no_position_to_exit"})
                        continue
                    tif_exit = TimeInForce.DAY if in_rth else TimeInForce.OPG
                    req = MarketOrderRequest(
                        symbol=sym, notional=round(abs(cur_mv), 2),
                        side=OrderSide.SELL, time_in_force=tif_exit
                    )
                    order = trading_client.submit_order(req)
                    tif_str = "DAY" if in_rth else "OPG"
                    con.execute("""
                        INSERT INTO orders(id, symbol, side, qty, notional, order_type, tif, tp_pct, sl_pct,
                                           alpaca_order_id, status, client_order_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [rec_id, sym, "sell", 0, abs(cur_mv), "market", tif_str, TP_PCT, SL_PCT,
                          getattr(order, "id", None), "submitted", getattr(order, "client_order_id", None)])
                    _ban_symbol_after_sell(sym)
                    results.append({"ok": True, "symbol": sym, "action": f"exit_all_{tif_str.lower()}"})
                else:
                    tif = "DAY" if in_rth else "OPG"
                    if price is None or price <= 0:
                        results.append({"ok": False, "symbol": sym, "error": "price_missing_for_qty"})
                        continue
                    notional_eff = float(usd_override) if usd_override is not None else USD_PER_TRADE
                    qty = max(1, int(notional_eff // price))
                    req = MarketOrderRequest(
                        symbol=sym, qty=qty,
                        side=OrderSide.SELL, time_in_force=(TimeInForce.DAY if in_rth else TimeInForce.OPG)
                    )
                    order = trading_client.submit_order(req)
                    con.execute("""
                        INSERT INTO orders(id, symbol, side, qty, notional, order_type, tif, tp_pct, sl_pct,
                                           alpaca_order_id, status, client_order_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [
                        rec_id, sym, "sell", qty, float(notional_eff), "market", tif, TP_PCT, SL_PCT,
                        getattr(order, "id", None), "submitted", getattr(order, "client_order_id", None)
                    ])
                    if not in_rth and tif == "OPG":
                        con.execute("""
                            INSERT OR REPLACE INTO open_queue(id, symbol, side, notional, qty, tif, body)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, [rec_id, sym, "sell", float(notional_eff), qty, tif, json.dumps(j)])
                    results.append({"ok": True, "symbol": sym, "action": f"order_sell_{tif}", "qty": qty})

                if ORDER_THROTTLE_MS > 0:
                    await asyncio.sleep(ORDER_THROTTLE_MS / 1000.0)

            else:
                results.append({"ok": True, "symbol": sym, "action": f"no_trade_decision:{decision}"})

        except Exception as e:
            results.append({"ok": False, "symbol": sym, "error": str(e)})

    con.close()
    if SNAP_ON_INGEST:
        _ = _publish_snapshot()
    return {"ok": True, "n": len(results), "results": results}

# ------------ OPG queue trigger ------------
def submit_opg_from_queue() -> Dict[str, Any]:
    _ensure_schema()
    con = duckdb.connect(DB_PATH)

    rows = con.execute("""
        SELECT id, symbol, side, notional, qty, tif, body
        FROM open_queue
    """).fetchall()

    if not rows:
        con.close()
        return {"ok": True, "inserted": 0, "skipped": 0, "remaining": 0}

    inserted = 0
    skipped  = 0

    for id_, sym, side, notional, qty, tif, body_json in rows:
        # Guard against double-submit
        already = con.execute("SELECT 1 FROM orders WHERE id = ? LIMIT 1", [id_]).fetchone()
        if already:
            con.execute("UPDATE alerts SET queued_consumed = TRUE WHERE id = ?", [id_])
            con.execute("DELETE FROM open_queue WHERE id = ?", [id_])
            skipped += 1
            continue

        # Check sell ban for BUY orders
        if side == "buy" and _is_symbol_banned(sym):
            con.execute("UPDATE alerts SET queued_consumed = TRUE WHERE id = ?", [id_])
            con.execute("DELETE FROM open_queue WHERE id = ?", [id_])
            skipped += 1
            continue

        try:
            res = trading_client.submit_order(MarketOrderRequest(
                symbol=sym,
                qty=int(qty),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.OPG,
            ))
            con.execute("""
                INSERT INTO orders(id, symbol, side, qty, notional, order_type, tif, tp_pct, sl_pct,
                                   alpaca_order_id, status, client_order_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                id_, sym, side, int(qty), float(notional), "market", "OPG", TP_PCT, SL_PCT,
                res.id, "submitted", getattr(res, "client_order_id", None)
            ])
            con.execute("UPDATE alerts SET queued_consumed = TRUE WHERE id = ?", [id_])
            con.execute("DELETE FROM open_queue WHERE id = ?", [id_])
            inserted += 1
        except Exception:
            pass

    remaining = con.execute("SELECT COUNT(*) FROM open_queue").fetchone()[0]
    con.close()
    return {"ok": True, "inserted": inserted, "skipped": skipped, "remaining": int(remaining)}

# ------------ Positions snapshot ------------
def _row_from_position(p) -> Dict[str, Any]:
    def f(x, d=None):
        try:
            return float(x)
        except Exception:
            return d

    def get(obj, name, d=None):
        return getattr(obj, name, d)

    realized_day = get(p, "realized_pl_day", None)  # usually not present
    if realized_day is None:
        realized_day = get(p, "unrealized_intraday_pl", None)  # best available proxy

    return dict(
        symbol=get(p, "symbol"),
        qty=f(get(p, "qty")),
        avg_entry=f(get(p, "avg_entry_price")),
        market_price=f(get(p, "current_price")),
        market_value=f(get(p, "market_value")),
        unrealized_pl=f(get(p, "unrealized_pl")),
        unrealized_plpc=f(get(p, "unrealized_plpc")),
        realized_pl_day=f(realized_day, None),
    )

def snapshot_positions() -> Dict[str, Any]:
    _ensure_schema()
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    if not positions:
        return {"ok": True, "inserted": 0}

    now_ts = utcnow().isoformat()
    rows = []
    for p in positions:
        d = _row_from_position(p)
        rows.append((
            now_ts, d["symbol"], d["qty"], d["avg_entry"], d["market_price"], d["market_value"],
            d["unrealized_pl"], d["unrealized_plpc"], d["realized_pl_day"]
        ))

    con = duckdb.connect(DB_PATH)
    con.executemany("""
        INSERT INTO positions (
            snapshot_ts, symbol, qty, avg_entry, market_price, market_value,
            unrealized_pl, unrealized_plpc, realized_pl_day
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.close()
    return {"ok": True, "inserted": len(rows)}

@app.post("/sync_positions")
async def api_sync_positions(x_api_key: Optional[str] = Header(None)):
    _auth_or_401(x_api_key)
    return snapshot_positions()

@app.get("/get_positions")
async def api_get_positions(x_api_key: Optional[str] = Header(None)):
    """
    Returns the latest snapshot of open positions from DuckDB.
    Used by n8n to retrieve current positions for review logic.
    
    Returns JSON:
    {
        "ok": true,
        "count": 5,
        "snapshot_ts": "2025-11-04T10:30:00+00:00",
        "positions": [
            {
                "symbol": "AAPL",
                "qty": 10.0,
                "avg_entry": 150.25,
                "market_price": 152.30,
                "market_value": 1523.00,
                "unrealized_pl": 20.50,
                "unrealized_plpc": 1.36,
                "realized_pl_day": 5.00
            },
            ...
        ]
    }
    """
    _auth_or_401(x_api_key)
    _ensure_schema()
    
    try:
        con = duckdb.connect(DB_PATH, read_only=True)
        
        # Get the latest snapshot timestamp
        ts_result = con.execute("""
            SELECT MAX(snapshot_ts) as latest_ts
            FROM positions
        """).fetchone()
        
        if not ts_result or ts_result[0] is None:
            con.close()
            return {
                "ok": True,
                "count": 0,
                "snapshot_ts": None,
                "positions": []
            }
        
        latest_ts = ts_result[0]
        
        # Get all positions from the latest snapshot
        rows = con.execute("""
            SELECT 
                symbol,
                qty,
                avg_entry,
                market_price,
                market_value,
                unrealized_pl,
                unrealized_plpc,
                realized_pl_day
            FROM positions
            WHERE snapshot_ts = ?
            ORDER BY symbol
        """, [latest_ts]).fetchall()
        
        con.close()
        
        # Convert to list of dicts
        positions = []
        for row in rows:
            positions.append({
                "symbol": row[0],
                "qty": float(row[1]) if row[1] is not None else 0.0,
                "avg_entry": float(row[2]) if row[2] is not None else 0.0,
                "market_price": float(row[3]) if row[3] is not None else 0.0,
                "market_value": float(row[4]) if row[4] is not None else 0.0,
                "unrealized_pl": float(row[5]) if row[5] is not None else 0.0,
                "unrealized_plpc": float(row[6]) if row[6] is not None else 0.0,
                "realized_pl_day": float(row[7]) if row[7] is not None else None
            })
        
        return {
            "ok": True,
            "count": len(positions),
            "snapshot_ts": latest_ts.isoformat() if hasattr(latest_ts, 'isoformat') else str(latest_ts),
            "positions": positions
        }
        
    except Exception as e:
        log.error(f"Error in /get_positions: {str(e)}")
        return {
            "ok": False,
            "error": str(e),
            "count": 0,
            "positions": []
        }
