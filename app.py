"""
Flask web application per analisi gamma exposure 0DTE
"""
from flask import Flask, render_template, request, jsonify
import os
import time
import csv
import io
import json
import urllib.request
import urllib.parse
from typing import Any, Dict, Optional
import datetime as _dt
import tempfile
import pdfplumber
import pandas as pd
from werkzeug.utils import secure_filename
import re
import importlib.util
import sys

# Optional: load local .env for development (no-op if not installed / not present).
try:  # pragma: no cover
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover
    MongoClient = None

_PYMUPDF_AVAILABLE = importlib.util.find_spec("fitz") is not None
_RUNTIME_PYTHON = sys.executable
_IN_VENV = getattr(sys, "base_prefix", sys.prefix) != sys.prefix
try:
    _APP_BUILD = int(os.path.getmtime(__file__))
except Exception:
    _APP_BUILD = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


_MONGO_CLIENT: Optional["MongoClient"] = None
_MONGO_COLLECTION = None


def _get_mongo_collection():
    """Return Mongo collection for pressure points or None if not configured/available."""

    global _MONGO_CLIENT, _MONGO_COLLECTION
    if _MONGO_COLLECTION is not None:
        return _MONGO_COLLECTION

    if MongoClient is None:
        return None

    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None

    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_PRESSURE_COLLECTION") or "pressure_points").strip()

    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]
        # Unique by second; updates within the same second will overwrite.
        try:
            coll.create_index("ts", unique=True)
        except Exception:
            pass

        # TTL to avoid unbounded growth (keep more than 8h).
        # Requires a datetime field.
        try:
            coll.create_index("created_at", expireAfterSeconds=60 * 60 * 36)
        except Exception:
            pass
        _MONGO_COLLECTION = coll
        return _MONGO_COLLECTION
    except Exception:
        return None


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test_path = os.path.join(path, ".__write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False


def get_upload_folder() -> str:
    """Return a writable folder for uploads.

    Vercel/AWS Lambda filesystems are read-only except for /tmp.
    """

    env_folder = (os.getenv("UPLOAD_FOLDER") or "").strip()
    candidates = [p for p in [env_folder, "uploads"] if p]

    tmp_base = tempfile.gettempdir() or "/tmp"
    candidates.append(os.path.join(tmp_base, "uploads"))

    for folder in candidates:
        if _is_writable_dir(folder):
            return folder

    # Last resort: /tmp
    return tmp_base


app.config['UPLOAD_FOLDER'] = get_upload_folder()


_SP500_PRICE_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_ES_PRICE_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_NVDA_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_SPY_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_MSFT_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_SPX_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_XSP_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_AAPL_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_GOOG_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_AMZN_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


def _parse_pdf_number(value: object) -> float:
    """Parse numeric strings found in PDFs.

    Handles both:
    - US style: 1,234.56
    - EU style: 1.234,56
    - Thousand separators only: 1,234 or 1.234
    """

    raw = ("" if value is None else str(value)).strip()
    if not raw or raw.lower() in {"none", "nan", ""}:
        return 0.0

    raw = raw.replace("\u00a0", "").replace(" ", "")
    raw = raw.replace("$", "")

    negative = False
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw[1:-1]

    # Keep only digits, separators and sign
    raw = re.sub(r"[^0-9,\.\-]", "", raw)
    if not raw or raw in {"-", ".", ","}:
        return 0.0

    has_dot = "." in raw
    has_comma = "," in raw

    try:
        if has_dot and has_comma:
            # Decide decimal separator as the rightmost of the two.
            if raw.rfind(",") > raw.rfind("."):
                # EU: '.' thousands, ',' decimal
                raw = raw.replace(".", "")
                raw = raw.replace(",", ".")
            else:
                # US: ',' thousands, '.' decimal
                raw = raw.replace(",", "")
        elif has_dot:
            # If dot-groups look like thousands (e.g. 1.234 or 12.345.678), remove dots.
            if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", raw):
                raw = raw.replace(".", "")
        elif has_comma:
            # If comma-groups look like thousands, remove commas; else treat comma as decimal.
            if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", raw):
                raw = raw.replace(",", "")
            else:
                raw = raw.replace(",", ".")

        out = float(raw)
        return -out if negative else out
    except Exception:
        return 0.0


def _fetch_stooq_latest_close(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches the latest Stooq CSV row for a symbol (no API key).

    Note: Stooq exposes OHLCV fields; the app uses the `Close` column as the
    latest available quote. This is commonly delayed/indicative (not CME real-time).

    Returns a dict with keys: symbol, price, date, time, source.
    """

    url = f"https://stooq.com/q/l/?s={urllib.parse.quote(symbol)}&f=sd2t2ohlcv&h&e=csv"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            raw = response.read().decode("utf-8", errors="replace")

        reader = csv.DictReader(io.StringIO(raw))
        row = next(reader, None)
        if not row:
            return None

        close_val = (row.get("Close") or "").strip()
        if not close_val or close_val.upper() in {"N/D", "NA", "NULL"}:
            return None

        return {
            "symbol": (row.get("Symbol") or symbol).strip(),
            "price": float(close_val),
            "date": (row.get("Date") or "").strip(),
            "time": (row.get("Time") or "").strip(),
            "source": "stooq",
        }
    except Exception:
        return None


def _fetch_nasdaq_json(url: str, referer: str) -> Optional[Dict[str, Any]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception:
        return None


def _parse_nasdaq_month_day(text: str, now: Optional[_dt.date] = None) -> Optional[_dt.date]:
    """Parse strings like 'Jan 2' into a concrete date near 'now'."""

    if not text:
        return None

    now = now or _dt.date.today()
    cleaned = str(text).strip()

    # Typical format: 'Jan 2' or 'Jan 02'
    try:
        dt = _dt.datetime.strptime(f"{cleaned} {now.year}", "%b %d %Y").date()
    except Exception:
        return None

    # If it ended up in the past (e.g. around year rollover), bump to next year.
    if dt < now:
        try:
            dt = _dt.datetime.strptime(f"{cleaned} {now.year + 1}", "%b %d %Y").date()
        except Exception:
            return None

    return dt


def _get_nasdaq_stock_snapshot_cached(
    symbol: str,
    cache: Dict[str, Any],
    max_age_seconds: int = 60,
) -> Optional[Dict[str, Any]]:
    """Generic Nasdaq option-chain snapshot for a US stock symbol."""

    now_ts = time.time()
    cached = cache.get("value")
    fetched_at = float(cache.get("fetched_at") or 0.0)
    if cached and (now_ts - fetched_at) <= max_age_seconds:
        return cached

    sym = (symbol or "").strip().upper()
    if not sym:
        return None

    referer = f"https://www.nasdaq.com/market-activity/stocks/{sym.lower()}/option-chain"
    url = f"https://api.nasdaq.com/api/quote/{urllib.parse.quote(sym)}/option-chain?assetclass=stocks"
    payload = _fetch_nasdaq_json(url, referer=referer)
    if not payload:
        return None

    data = payload.get("data") or {}
    table = data.get("table") or {}
    rows = table.get("rows") or []

    last_trade_raw = (data.get("lastTrade") or "").strip()
    last_sale_price = None
    last_sale_time = None
    if last_trade_raw:
        m = re.search(r"\$\s*([0-9][0-9,\.]+)", last_trade_raw)
        if m:
            last_sale_price = _parse_pdf_number(m.group(1))
        m2 = re.search(r"\(\s*AS\s+OF\s+([^\)]+)\)", last_trade_raw, re.IGNORECASE)
        if m2:
            last_sale_time = m2.group(1).strip()
        else:
            last_sale_time = last_trade_raw

    today = _dt.date.today()
    expiry_candidates: Dict[str, _dt.date] = {}
    for row in rows:
        exp = (row.get("expiryDate") or "").strip()
        if not exp:
            continue
        parsed = _parse_nasdaq_month_day(exp, now=today)
        if parsed:
            expiry_candidates[exp] = parsed

    if not expiry_candidates:
        return None

    nearest_exp_label, nearest_exp_date = sorted(expiry_candidates.items(), key=lambda kv: kv[1])[0]

    strikes: list[float] = []
    calls: list[float] = []
    puts: list[float] = []
    gammas: list[float] = []

    for row in rows:
        if (row.get("expiryDate") or "").strip() != nearest_exp_label:
            continue

        strike = _parse_pdf_number(row.get("strike"))
        if strike <= 0:
            continue

        call_oi = _parse_pdf_number(row.get("c_Openinterest"))
        put_oi = _parse_pdf_number(row.get("p_Openinterest"))
        gamma_exposure = (call_oi - put_oi) * 1000

        strikes.append(float(strike))
        calls.append(float(call_oi))
        puts.append(float(put_oi))
        gammas.append(float(gamma_exposure))

    if not strikes:
        return None

    df = pd.DataFrame({
        "Strike": strikes,
        "Call_OI": calls,
        "Put_OI": puts,
        "Gamma_Exposure": gammas,
    }).sort_values("Strike").reset_index(drop=True)

    results = analyze_0dte(df, current_price=float(last_sale_price) if last_sale_price else None)
    snapshot: Dict[str, Any] = {
        "symbol": sym,
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    if isinstance(results, dict):
        snapshot.update(results)

    cache["value"] = snapshot
    cache["fetched_at"] = now_ts
    return snapshot


def get_aapl_snapshot_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    return _get_nasdaq_stock_snapshot_cached("AAPL", _AAPL_SNAPSHOT_CACHE, max_age_seconds=max_age_seconds)


def get_goog_snapshot_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    return _get_nasdaq_stock_snapshot_cached("GOOG", _GOOG_SNAPSHOT_CACHE, max_age_seconds=max_age_seconds)


def get_amzn_snapshot_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    return _get_nasdaq_stock_snapshot_cached("AMZN", _AMZN_SNAPSHOT_CACHE, max_age_seconds=max_age_seconds)


def get_nvda_snapshot_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch NVDA last price + option-chain derived gamma flip for the nearest expiry."""

    now_ts = time.time()
    cached = _NVDA_SNAPSHOT_CACHE.get("value")
    fetched_at = float(_NVDA_SNAPSHOT_CACHE.get("fetched_at") or 0.0)
    if cached and (now_ts - fetched_at) <= max_age_seconds:
        return cached

    referer = "https://www.nasdaq.com/market-activity/stocks/nvda/option-chain"
    url = "https://api.nasdaq.com/api/quote/NVDA/option-chain?assetclass=stocks"
    payload = _fetch_nasdaq_json(url, referer=referer)
    if not payload:
        return None

    data = payload.get("data") or {}
    table = data.get("table") or {}
    rows = table.get("rows") or []

    last_trade_raw = (data.get("lastTrade") or "").strip()
    last_sale_price = None
    last_sale_time = None
    if last_trade_raw:
        m = re.search(r"\$\s*([0-9][0-9,\.]+)", last_trade_raw)
        if m:
            last_sale_price = _parse_pdf_number(m.group(1))
        m2 = re.search(r"\(\s*AS\s+OF\s+([^\)]+)\)", last_trade_raw, re.IGNORECASE)
        if m2:
            last_sale_time = m2.group(1).strip()
        else:
            # Keep the raw string if it doesn't match expected formatting.
            last_sale_time = last_trade_raw

    # Determine nearest expiry present in the table (strings like 'Jan 2')
    today = _dt.date.today()
    expiry_candidates: Dict[str, _dt.date] = {}
    for row in rows:
        exp = (row.get("expiryDate") or "").strip()
        if not exp:
            continue
        parsed = _parse_nasdaq_month_day(exp, now=today)
        if parsed:
            expiry_candidates[exp] = parsed

    if not expiry_candidates:
        return None

    # Choose the closest expiry date.
    nearest_exp_label, nearest_exp_date = sorted(expiry_candidates.items(), key=lambda kv: kv[1])[0]

    strikes: list[float] = []
    calls: list[float] = []
    puts: list[float] = []
    gammas: list[float] = []

    for row in rows:
        if (row.get("expiryDate") or "").strip() != nearest_exp_label:
            continue

        strike = _parse_pdf_number(row.get("strike"))
        # Some rows are group/header rows (strike missing)
        if strike <= 0:
            continue

        call_oi = _parse_pdf_number(row.get("c_Openinterest"))
        put_oi = _parse_pdf_number(row.get("p_Openinterest"))
        gamma_exposure = (call_oi - put_oi) * 1000

        strikes.append(float(strike))
        calls.append(float(call_oi))
        puts.append(float(put_oi))
        gammas.append(float(gamma_exposure))

    if not strikes:
        return None

    df = pd.DataFrame({
        "Strike": strikes,
        "Call_OI": calls,
        "Put_OI": puts,
        "Gamma_Exposure": gammas,
    }).sort_values("Strike").reset_index(drop=True)

    results = analyze_0dte(df, current_price=float(last_sale_price) if last_sale_price else None)
    snapshot: Dict[str, Any] = {
        "symbol": "NVDA",
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    if isinstance(results, dict):
        snapshot.update(results)

    _NVDA_SNAPSHOT_CACHE["value"] = snapshot
    _NVDA_SNAPSHOT_CACHE["fetched_at"] = now_ts
    return snapshot


def get_spy_snapshot_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch SPY last price + option-chain derived gamma flip for the nearest expiry."""

    now_ts = time.time()
    cached = _SPY_SNAPSHOT_CACHE.get("value")
    fetched_at = float(_SPY_SNAPSHOT_CACHE.get("fetched_at") or 0.0)
    if cached and (now_ts - fetched_at) <= max_age_seconds:
        return cached

    # SPY is an ETF on Nasdaq; use the ETF option chain endpoint.
    referer = "https://www.nasdaq.com/market-activity/etf/spy/option-chain"
    url = "https://api.nasdaq.com/api/quote/SPY/option-chain?assetclass=etf"
    payload = _fetch_nasdaq_json(url, referer=referer)
    if not payload:
        # Fallback: some environments may require the 'stocks' assetclass.
        referer = "https://www.nasdaq.com/market-activity/stocks/spy/option-chain"
        url = "https://api.nasdaq.com/api/quote/SPY/option-chain?assetclass=stocks"
        payload = _fetch_nasdaq_json(url, referer=referer)
        if not payload:
            return None

    data = payload.get("data") or {}
    table = data.get("table") or {}
    rows = table.get("rows") or []

    last_trade_raw = (data.get("lastTrade") or "").strip()
    last_sale_price = None
    last_sale_time = None
    if last_trade_raw:
        m = re.search(r"\$\s*([0-9][0-9,\.]+)", last_trade_raw)
        if m:
            last_sale_price = _parse_pdf_number(m.group(1))
        m2 = re.search(r"\(\s*AS\s+OF\s+([^\)]+)\)", last_trade_raw, re.IGNORECASE)
        if m2:
            last_sale_time = m2.group(1).strip()
        else:
            last_sale_time = last_trade_raw

    # Determine nearest expiry present in the table (strings like 'Jan 2')
    today = _dt.date.today()
    expiry_candidates: Dict[str, _dt.date] = {}
    for row in rows:
        exp = (row.get("expiryDate") or "").strip()
        if not exp:
            continue
        parsed = _parse_nasdaq_month_day(exp, now=today)
        if parsed:
            expiry_candidates[exp] = parsed

    if not expiry_candidates:
        return None

    nearest_exp_label, nearest_exp_date = sorted(expiry_candidates.items(), key=lambda kv: kv[1])[0]

    strikes: list[float] = []
    calls: list[float] = []
    puts: list[float] = []
    gammas: list[float] = []

    for row in rows:
        if (row.get("expiryDate") or "").strip() != nearest_exp_label:
            continue

        strike = _parse_pdf_number(row.get("strike"))
        if strike <= 0:
            continue

        call_oi = _parse_pdf_number(row.get("c_Openinterest"))
        put_oi = _parse_pdf_number(row.get("p_Openinterest"))
        gamma_exposure = (call_oi - put_oi) * 1000

        strikes.append(float(strike))
        calls.append(float(call_oi))
        puts.append(float(put_oi))
        gammas.append(float(gamma_exposure))

    if not strikes:
        return None

    df = pd.DataFrame({
        "Strike": strikes,
        "Call_OI": calls,
        "Put_OI": puts,
        "Gamma_Exposure": gammas,
    }).sort_values("Strike").reset_index(drop=True)

    results = analyze_0dte(df, current_price=float(last_sale_price) if last_sale_price else None)
    snapshot: Dict[str, Any] = {
        "symbol": "SPY",
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    if isinstance(results, dict):
        snapshot.update(results)

    _SPY_SNAPSHOT_CACHE["value"] = snapshot
    _SPY_SNAPSHOT_CACHE["fetched_at"] = now_ts
    return snapshot


def get_msft_snapshot_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch MSFT last price + option-chain derived gamma flip for the nearest expiry."""

    now_ts = time.time()
    cached = _MSFT_SNAPSHOT_CACHE.get("value")
    fetched_at = float(_MSFT_SNAPSHOT_CACHE.get("fetched_at") or 0.0)
    if cached and (now_ts - fetched_at) <= max_age_seconds:
        return cached

    referer = "https://www.nasdaq.com/market-activity/stocks/msft/option-chain"
    url = "https://api.nasdaq.com/api/quote/MSFT/option-chain?assetclass=stocks"
    payload = _fetch_nasdaq_json(url, referer=referer)
    if not payload:
        return None

    data = payload.get("data") or {}
    table = data.get("table") or {}
    rows = table.get("rows") or []

    last_trade_raw = (data.get("lastTrade") or "").strip()
    last_sale_price = None
    last_sale_time = None
    if last_trade_raw:
        m = re.search(r"\$\s*([0-9][0-9,\.]+)", last_trade_raw)
        if m:
            last_sale_price = _parse_pdf_number(m.group(1))
        m2 = re.search(r"\(\s*AS\s+OF\s+([^\)]+)\)", last_trade_raw, re.IGNORECASE)
        if m2:
            last_sale_time = m2.group(1).strip()
        else:
            last_sale_time = last_trade_raw

    today = _dt.date.today()
    expiry_candidates: Dict[str, _dt.date] = {}
    for row in rows:
        exp = (row.get("expiryDate") or "").strip()
        if not exp:
            continue
        parsed = _parse_nasdaq_month_day(exp, now=today)
        if parsed:
            expiry_candidates[exp] = parsed

    if not expiry_candidates:
        return None

    nearest_exp_label, nearest_exp_date = sorted(expiry_candidates.items(), key=lambda kv: kv[1])[0]

    strikes: list[float] = []
    calls: list[float] = []
    puts: list[float] = []
    gammas: list[float] = []

    for row in rows:
        if (row.get("expiryDate") or "").strip() != nearest_exp_label:
            continue

        strike = _parse_pdf_number(row.get("strike"))
        if strike <= 0:
            continue

        call_oi = _parse_pdf_number(row.get("c_Openinterest"))
        put_oi = _parse_pdf_number(row.get("p_Openinterest"))
        gamma_exposure = (call_oi - put_oi) * 1000

        strikes.append(float(strike))
        calls.append(float(call_oi))
        puts.append(float(put_oi))
        gammas.append(float(gamma_exposure))

    if not strikes:
        return None

    df = pd.DataFrame({
        "Strike": strikes,
        "Call_OI": calls,
        "Put_OI": puts,
        "Gamma_Exposure": gammas,
    }).sort_values("Strike").reset_index(drop=True)

    results = analyze_0dte(df, current_price=float(last_sale_price) if last_sale_price else None)
    snapshot: Dict[str, Any] = {
        "symbol": "MSFT",
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    if isinstance(results, dict):
        snapshot.update(results)

    _MSFT_SNAPSHOT_CACHE["value"] = snapshot
    _MSFT_SNAPSHOT_CACHE["fetched_at"] = now_ts
    return snapshot


def get_spx_snapshot_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch SPX last price + option-chain derived gamma flip for the nearest expiry.

    Nasdaq may not expose SPX option chains reliably; when unavailable, falls back
    to SPY option chain as a proxy.
    """

    now_ts = time.time()
    cached = _SPX_SNAPSHOT_CACHE.get("value")
    fetched_at = float(_SPX_SNAPSHOT_CACHE.get("fetched_at") or 0.0)
    if cached and (now_ts - fetched_at) <= max_age_seconds:
        return cached

    payload = None
    # Try likely Nasdaq index endpoints.
    candidates = [
        (
            "https://www.nasdaq.com/market-activity/index/spx/option-chain",
            "https://api.nasdaq.com/api/quote/SPX/option-chain?assetclass=index",
        ),
        (
            "https://www.nasdaq.com/market-activity/index/spx/option-chain",
            "https://api.nasdaq.com/api/quote/SPX/option-chain?assetclass=indexes",
        ),
        (
            "https://www.nasdaq.com/market-activity/index/spx/option-chain",
            "https://api.nasdaq.com/api/quote/SPX/option-chain?assetclass=stocks",
        ),
    ]
    for referer, url in candidates:
        payload = _fetch_nasdaq_json(url, referer=referer)
        if payload:
            break

    if not payload:
        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "SPX"
        snapshot["note"] = "Proxy (SPY option chain) used when SPX unavailable"
        _SPX_SNAPSHOT_CACHE["value"] = snapshot
        _SPX_SNAPSHOT_CACHE["fetched_at"] = now_ts
        return snapshot

    data = payload.get("data") or {}
    table = data.get("table") or {}
    rows = table.get("rows") or []

    last_trade_raw = (data.get("lastTrade") or "").strip()
    last_sale_price = None
    last_sale_time = None
    if last_trade_raw:
        m = re.search(r"\$\s*([0-9][0-9,\.]+)", last_trade_raw)
        if m:
            last_sale_price = _parse_pdf_number(m.group(1))
        m2 = re.search(r"\(\s*AS\s+OF\s+([^\)]+)\)", last_trade_raw, re.IGNORECASE)
        if m2:
            last_sale_time = m2.group(1).strip()
        else:
            last_sale_time = last_trade_raw

    today = _dt.date.today()
    expiry_candidates: Dict[str, _dt.date] = {}
    for row in rows:
        exp = (row.get("expiryDate") or "").strip()
        if not exp:
            continue
        parsed = _parse_nasdaq_month_day(exp, now=today)
        if parsed:
            expiry_candidates[exp] = parsed

    if not expiry_candidates:
        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "SPX"
        snapshot["note"] = "Proxy (SPY option chain) used when SPX expiries unavailable"
        _SPX_SNAPSHOT_CACHE["value"] = snapshot
        _SPX_SNAPSHOT_CACHE["fetched_at"] = now_ts
        return snapshot

    nearest_exp_label, nearest_exp_date = sorted(expiry_candidates.items(), key=lambda kv: kv[1])[0]

    strikes: list[float] = []
    calls: list[float] = []
    puts: list[float] = []
    gammas: list[float] = []

    for row in rows:
        if (row.get("expiryDate") or "").strip() != nearest_exp_label:
            continue

        strike = _parse_pdf_number(row.get("strike"))
        if strike <= 0:
            continue

        call_oi = _parse_pdf_number(row.get("c_Openinterest"))
        put_oi = _parse_pdf_number(row.get("p_Openinterest"))
        gamma_exposure = (call_oi - put_oi) * 1000

        strikes.append(float(strike))
        calls.append(float(call_oi))
        puts.append(float(put_oi))
        gammas.append(float(gamma_exposure))

    if not strikes:
        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "SPX"
        snapshot["note"] = "Proxy (SPY option chain) used when SPX strikes unavailable"
        _SPX_SNAPSHOT_CACHE["value"] = snapshot
        _SPX_SNAPSHOT_CACHE["fetched_at"] = now_ts
        return snapshot

    df = pd.DataFrame({
        "Strike": strikes,
        "Call_OI": calls,
        "Put_OI": puts,
        "Gamma_Exposure": gammas,
    }).sort_values("Strike").reset_index(drop=True)

    results = analyze_0dte(df, current_price=float(last_sale_price) if last_sale_price else None)
    snapshot: Dict[str, Any] = {
        "symbol": "SPX",
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    if isinstance(results, dict):
        snapshot.update(results)

    _SPX_SNAPSHOT_CACHE["value"] = snapshot
    _SPX_SNAPSHOT_CACHE["fetched_at"] = now_ts
    return snapshot


def get_xsp_snapshot_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch XSP last price + option-chain derived gamma flip for the nearest expiry.

    If Nasdaq does not provide XSP chains, falls back to SPY option chain as proxy.
    """

    now_ts = time.time()
    cached = _XSP_SNAPSHOT_CACHE.get("value")
    fetched_at = float(_XSP_SNAPSHOT_CACHE.get("fetched_at") or 0.0)
    if cached and (now_ts - fetched_at) <= max_age_seconds:
        return cached

    payload = None
    candidates = [
        (
            "https://www.nasdaq.com/market-activity/etf/xsp/option-chain",
            "https://api.nasdaq.com/api/quote/XSP/option-chain?assetclass=etf",
        ),
        (
            "https://www.nasdaq.com/market-activity/stocks/xsp/option-chain",
            "https://api.nasdaq.com/api/quote/XSP/option-chain?assetclass=stocks",
        ),
    ]
    for referer, url in candidates:
        payload = _fetch_nasdaq_json(url, referer=referer)
        if payload:
            break

    if not payload:
        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "XSP"
        snapshot["note"] = "Proxy (SPY option chain) used when XSP unavailable"
        _XSP_SNAPSHOT_CACHE["value"] = snapshot
        _XSP_SNAPSHOT_CACHE["fetched_at"] = now_ts
        return snapshot

    data = payload.get("data") or {}
    table = data.get("table") or {}
    rows = table.get("rows") or []

    last_trade_raw = (data.get("lastTrade") or "").strip()
    last_sale_price = None
    last_sale_time = None
    if last_trade_raw:
        m = re.search(r"\$\s*([0-9][0-9,\.]+)", last_trade_raw)
        if m:
            last_sale_price = _parse_pdf_number(m.group(1))
        m2 = re.search(r"\(\s*AS\s+OF\s+([^\)]+)\)", last_trade_raw, re.IGNORECASE)
        if m2:
            last_sale_time = m2.group(1).strip()
        else:
            last_sale_time = last_trade_raw

    today = _dt.date.today()
    expiry_candidates: Dict[str, _dt.date] = {}
    for row in rows:
        exp = (row.get("expiryDate") or "").strip()
        if not exp:
            continue
        parsed = _parse_nasdaq_month_day(exp, now=today)
        if parsed:
            expiry_candidates[exp] = parsed

    if not expiry_candidates:
        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "XSP"
        snapshot["note"] = "Proxy (SPY option chain) used when XSP expiries unavailable"
        _XSP_SNAPSHOT_CACHE["value"] = snapshot
        _XSP_SNAPSHOT_CACHE["fetched_at"] = now_ts
        return snapshot

    nearest_exp_label, nearest_exp_date = sorted(expiry_candidates.items(), key=lambda kv: kv[1])[0]

    strikes: list[float] = []
    calls: list[float] = []
    puts: list[float] = []
    gammas: list[float] = []

    for row in rows:
        if (row.get("expiryDate") or "").strip() != nearest_exp_label:
            continue

        strike = _parse_pdf_number(row.get("strike"))
        if strike <= 0:
            continue

        call_oi = _parse_pdf_number(row.get("c_Openinterest"))
        put_oi = _parse_pdf_number(row.get("p_Openinterest"))
        gamma_exposure = (call_oi - put_oi) * 1000

        strikes.append(float(strike))
        calls.append(float(call_oi))
        puts.append(float(put_oi))
        gammas.append(float(gamma_exposure))

    if not strikes:
        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "XSP"
        snapshot["note"] = "Proxy (SPY option chain) used when XSP strikes unavailable"
        _XSP_SNAPSHOT_CACHE["value"] = snapshot
        _XSP_SNAPSHOT_CACHE["fetched_at"] = now_ts
        return snapshot

    df = pd.DataFrame({
        "Strike": strikes,
        "Call_OI": calls,
        "Put_OI": puts,
        "Gamma_Exposure": gammas,
    }).sort_values("Strike").reset_index(drop=True)

    results = analyze_0dte(df, current_price=float(last_sale_price) if last_sale_price else None)
    snapshot: Dict[str, Any] = {
        "symbol": "XSP",
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    if isinstance(results, dict):
        snapshot.update(results)

    _XSP_SNAPSHOT_CACHE["value"] = snapshot
    _XSP_SNAPSHOT_CACHE["fetched_at"] = now_ts
    return snapshot


def get_sp500_price_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _SP500_PRICE_CACHE.get("value")
    fetched_at = float(_SP500_PRICE_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    # Prefer the index; fall back to SPY as a proxy if the index is unavailable.
    for symbol in ("^spx", "spy.us"):
        data = _fetch_stooq_latest_close(symbol)
        if data:
            if symbol != "^spx":
                data["note"] = "Proxy (SPY) used when ^SPX unavailable"
            _SP500_PRICE_CACHE["value"] = data
            _SP500_PRICE_CACHE["fetched_at"] = now
            return data

    return None


def get_es_price_cached(max_age_seconds: int = 5) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _ES_PRICE_CACHE.get("value")
    fetched_at = float(_ES_PRICE_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    # ES continuous future on Stooq.
    data = _fetch_stooq_latest_close("es.f")
    if not data:
        return None

    data["instrument"] = "ES Futures"
    data["note"] = "Stooq es.f (continuous); quote may be delayed"
    _ES_PRICE_CACHE["value"] = data
    _ES_PRICE_CACHE["fetched_at"] = now
    return data

def extract_0dte_data(pdf_path: str) -> pd.DataFrame:
    """Estrae solo i dati 0DTE dal PDF Open Interest Matrix."""

    # Fast/robust path: coordinate-based extraction via PyMuPDF.
    # Some PDFs cause pdfplumber table detection to be very slow or incomplete.
    df = _extract_dte_pair_data_pymupdf(pdf_path, target_days=0)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    return _extract_dte_days_data(pdf_path, target_days=0)


def extract_1dte_data(pdf_path: str) -> pd.DataFrame:
    """Estrae solo i dati 1DTE dal PDF Open Interest Matrix.

    Molti PDF hanno struttura: Strike | None | Call_0DTE | Put_0DTE | Call_1DTE | Put_1DTE | ...
    """

    df = _extract_dte_pair_data_pymupdf(pdf_path, target_days=1)
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df

    return _extract_dte_days_data(pdf_path, target_days=1)


def extract_nearest_positive_dte_data(pdf_path: str) -> pd.DataFrame:
    """Fallback: estrae i dati della scadenza con DTE minimo > 0 disponibile nel PDF."""

    mapping = _find_dte_column_mapping(pdf_path)
    positive_days = sorted([d for d in mapping.keys() if isinstance(d, int) and d > 0])
    for d in positive_days:
        df = _extract_dte_days_data(pdf_path, target_days=d)
        if not df.empty:
            return df

    # Fallback: some PDFs don't yield tables via pdfplumber; try coordinate-based parsing.
    pymu_days = _find_available_dtes_pymupdf(pdf_path)
    positive_days = sorted([d for d in pymu_days if isinstance(d, int) and d > 0])
    for d in positive_days:
        df = _extract_dte_pair_data_pymupdf(pdf_path, target_days=d)
        if not df.empty:
            return df

    return pd.DataFrame()


def _extract_dte_days_data(pdf_path: str, target_days: int) -> pd.DataFrame:
    mapping = _find_dte_column_mapping(pdf_path)
    pair = mapping.get(int(target_days))
    if pair:
        call_col, put_col = pair
        df = _extract_dte_pair_data(pdf_path, call_col=call_col, put_col=put_col)
        if not df.empty:
            return df

    # Fallback: try coordinate-based parsing when pdfplumber table extraction fails.
    return _extract_dte_pair_data_pymupdf(pdf_path, target_days=int(target_days))


def _find_available_dtes_pymupdf(pdf_path: str) -> list[int]:
    # Prefer extracting the ordered day list from the PDF text stream; this is
    # often more reliable than trying to infer day labels from table coordinates.
    code_to_day = _find_contract_code_to_day_pypdf2(pdf_path)
    if code_to_day:
        return sorted({d for d in code_to_day.values() if isinstance(d, int)})

    ordered_days = _find_dte_days_order_pypdf2(pdf_path)
    if ordered_days:
        return ordered_days

    try:
        import fitz  # PyMuPDF
    except Exception:
        return []

    def _parse_day(text: str) -> Optional[int]:
        m = re.search(r'\b(\d+)\s*DTE\b', text.upper())
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
        return None

    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count < 1:
                return []
            page = doc[0]
            words = page.get_text('words')
    except Exception:
        return []

    days: set[int] = set()
    # Direct tokens like "1DTE" or "1 DTE" (number token near DTE token)
    simple = [(str(txt), float(x0), float(y0), float(x1)) for x0, y0, x1, y1, txt, *_ in words]
    for txt, *_ in simple:
        d = _parse_day(txt)
        if d is not None:
            days.add(d)

    # Patterns like "1" followed by a separate "DTE" token on the same row.
    dte_tokens = [(x0, y0) for x0, y0, x1, y1, txt, *_ in words if str(txt).upper() == 'DTE']
    if dte_tokens:
        numeric_tokens = [(x0, y0, x1, str(txt)) for x0, y0, x1, y1, txt, *_ in words]
        for dte_x0, dte_y0 in dte_tokens:
            # Find the nearest numeric token immediately to the left on the same row.
            best = None
            best_x1 = None
            for x0, y0, x1, txt in numeric_tokens:
                if abs(y0 - dte_y0) > 3.0:
                    continue
                if x1 > dte_x0 + 1:
                    continue
                m = re.match(r'^\d{1,3}$', txt.strip())
                if not m:
                    continue
                if best_x1 is None or x1 > best_x1:
                    best = txt
                    best_x1 = x1
            if best is not None:
                try:
                    days.add(int(best))
                except Exception:
                    pass

    return sorted(days)


def _find_dte_days_order_pypdf2(pdf_path: str) -> list[int]:
    """Extract the ordered list of DTE day numbers from the PDF text.

    Many QuikStrike PDFs include a header like "1 DTE", "2 DTE", ... in the
    intended left-to-right order.
    """

    try:
        from PyPDF2 import PdfReader
    except Exception:
        return []

    try:
        reader = PdfReader(pdf_path)
        if not reader.pages:
            return []
        text = reader.pages[0].extract_text() or ''
    except Exception:
        return []

    days: list[int] = []
    seen: set[int] = set()
    # Some PDFs concatenate the next token after DTE (e.g. "1 DTEE1BF6").
    # So we intentionally don't require a trailing word boundary after "DTE".
    for m in re.finditer(r'\b(\d{1,3})\s*DTE', text.upper()):
        try:
            d = int(m.group(1))
        except Exception:
            continue
        if d < 0 or d > 365:
            continue
        if d not in seen:
            days.append(d)
            seen.add(d)
    return days


def _find_contract_code_to_day_pypdf2(pdf_path: str) -> dict[str, int]:
    """Extract mapping of contract code -> DTE days from the PDF text header."""

    try:
        from PyPDF2 import PdfReader
    except Exception:
        return {}

    try:
        reader = PdfReader(pdf_path)
        if not reader.pages:
            return {}
        text = (reader.pages[0].extract_text() or '').upper()
    except Exception:
        return {}

    out: dict[str, int] = {}
    # Common QuikStrike header pattern: <CODE> <n> DTE (sometimes without spaces).
    for m in re.finditer(r'\b([A-Z][A-Z0-9]{2,12})\s*(\d{1,3})\s*DTE', text):
        code = m.group(1).strip().upper()
        if code.startswith('STRIKE'):
            code = code.replace('STRIKE', '', 1)
        try:
            d = int(m.group(2))
        except Exception:
            continue
        if d < 0 or d > 365:
            continue
        # Keep first occurrence.
        if code and code not in out:
            out[code] = d

    return out


def _extract_dte_pair_data_pymupdf(pdf_path: str, target_days: int) -> pd.DataFrame:
    """Fallback extractor using PyMuPDF word coordinates.

    This supports QuikStrike-style Open Interest Matrix PDFs where pdfplumber fails to
    reconstruct tables. It reconstructs Call/Put columns based on the C/P header row.
    """

    try:
        import fitz  # PyMuPDF
    except Exception:
        return pd.DataFrame()

    def _is_number_token(text: str) -> bool:
        s = text.strip().replace(',', '')
        if not s:
            return False
        # Keep plain integers/decimals only.
        return bool(re.match(r'^-?\d+(?:\.\d+)?$', s))

    def _parse_number(text: str) -> float:
        return _parse_pdf_number(text)

    try:
        with fitz.open(pdf_path) as doc:
            if doc.page_count < 1:
                return pd.DataFrame()
            page = doc[0]
            raw_words = page.get_text('words')
    except Exception:
        return pd.DataFrame()

    # Normalize word tuples.
    words: list[tuple[float, float, float, float, str]] = []
    for x0, y0, x1, y1, txt, *_ in raw_words:
        t = ('' if txt is None else str(txt)).strip()
        if not t:
            continue
        words.append((float(x0), float(y0), float(x1), float(y1), t))
    if not words:
        return pd.DataFrame()

    # Collect all C/P tokens.
    cp_words = [(x0, y0, x1, t) for x0, y0, x1, y1, t in words if t in {'C', 'P', 'CP'}]
    if not cp_words:
        return pd.DataFrame()

    # Heuristic: some QuikStrike PDFs are "transposed" (strikes across the top,
    # expiries down the side with C/P rows). In that case, C/P tokens are aligned
    # in a single narrow x-column and vary mostly by y.
    cp_xs = [x0 for x0, _, _, _ in cp_words]
    cp_ys = [y0 for _, y0, _, _ in cp_words]
    x_span = (max(cp_xs) - min(cp_xs)) if cp_xs else 0.0
    y_span = (max(cp_ys) - min(cp_ys)) if cp_ys else 0.0
    looks_transposed = x_span < 8.0 and y_span > 200.0

    if looks_transposed:
        # 1) Find the strike header row: many 4-5 digit numbers on the same y.
        strike_tokens = []
        for x0, y0, x1, y1, t in words:
            if not _is_number_token(t):
                continue
            v = _parse_number(t)
            if v < 1000 or v > 10000:
                continue
            # strikes are typically integer-ish
            strike_tokens.append((x0, y0, x1, v))

        if not strike_tokens:
            return pd.DataFrame()
        strike_tokens.sort(key=lambda it: (it[1], it[0]))
        rows_by_y: list[list[tuple[float, float, float, float]]] = []
        for tok in strike_tokens:
            if not rows_by_y or abs(tok[1] - rows_by_y[-1][0][1]) > 4.0:
                rows_by_y.append([tok])
            else:
                rows_by_y[-1].append(tok)
        strike_row = max(rows_by_y, key=lambda r: len(r))
        if len(strike_row) < 10:
            return pd.DataFrame()
        strike_row_sorted = sorted(strike_row, key=lambda it: it[0])
        strike_cols: list[tuple[float, float]] = [
            (float(v), (x0 + x1) / 2.0) for x0, y0, x1, v in strike_row_sorted
        ]
        # Deduplicate by strike value keeping left-most x.
        seen_strikes = set()
        strike_cols = [(s, x) for s, x in strike_cols if (s not in seen_strikes and not seen_strikes.add(s))]
        strike_cols.sort(key=lambda it: it[1])
        if len(strike_cols) < 10:
            return pd.DataFrame()

        xs = [x for _, x in strike_cols]
        diffs = [b - a for a, b in zip(xs, xs[1:]) if (b - a) > 0]
        tol_x = (sorted(diffs)[len(diffs) // 2] / 2.0) if diffs else 12.0

        # 2) Map contract codes to DTE days from PyPDF2 header.
        code_to_day = _find_contract_code_to_day_pypdf2(pdf_path)
        if not code_to_day:
            return pd.DataFrame()
        target_codes = {code for code, d in code_to_day.items() if int(d) == int(target_days)}
        if not target_codes:
            return pd.DataFrame()

        # 3) Find the y position of the contract code on the page.
        code_positions = [(t, y0) for x0, y0, x1, y1, t in words if t in target_codes]
        if not code_positions:
            return pd.DataFrame()
        # Use the first (top-most) matching code occurrence.
        code_y = sorted(code_positions, key=lambda it: it[1])[0][1]

        # 4) Find nearest C and P rows around that code y.
        cp_candidates = [(y0, t) for x0, y0, x1, t in cp_words if abs(y0 - code_y) <= 30.0]
        call_y = None
        put_y = None
        for y0, t in sorted(cp_candidates, key=lambda it: abs(it[0] - code_y)):
            if t == 'C' and call_y is None:
                call_y = y0
            if t == 'P' and put_y is None:
                put_y = y0
            if call_y is not None and put_y is not None:
                break
        if call_y is None or put_y is None:
            return pd.DataFrame()

        # Numeric values are often slightly offset from the C/P label baseline.
        # Snap to the densest numeric row near each label.
        min_strike_x = min(x for _, x in strike_cols)

        def snap_to_numeric_row(y_hint: float) -> float:
            candidates = []
            for x0, y0, x1, y1, t in words:
                if abs(y0 - y_hint) > 15.0:
                    continue
                if x0 < (min_strike_x - 5.0):
                    continue
                if not _is_number_token(t):
                    continue
                candidates.append((y0, x0))
            if not candidates:
                return y_hint
            candidates.sort()
            clusters: list[list[tuple[float, float]]] = []
            for y0, x0 in candidates:
                if not clusters or abs(y0 - clusters[-1][0][0]) > 2.5:
                    clusters.append([(y0, x0)])
                else:
                    clusters[-1].append((y0, x0))
            best = max(clusters, key=lambda c: len(c))
            return sum(y for y, _ in best) / len(best)

        call_y = snap_to_numeric_row(call_y)
        put_y = snap_to_numeric_row(put_y)

        # 5) Collect numeric tokens on those two rows.
        def row_numbers_at(y_target: float) -> list[tuple[float, float]]:
            out = []
            for x0, y0, x1, y1, t in words:
                if abs(y0 - y_target) > 4.0:
                    continue
                if not _is_number_token(t):
                    continue
                out.append(((x0 + x1) / 2.0, t))
            out.sort(key=lambda it: it[0])
            return out

        call_nums = row_numbers_at(call_y)
        put_nums = row_numbers_at(put_y)
        if not call_nums and not put_nums:
            return pd.DataFrame()

        def pick_value(nums: list[tuple[float, str]], x_target: float) -> float:
            best = None
            best_dist = None
            for xc, t in nums:
                dist = abs(xc - x_target)
                if dist > tol_x:
                    continue
                if best_dist is None or dist < best_dist:
                    best = t
                    best_dist = dist
            return _parse_number(best) if best is not None else 0.0

        strikes: list[float] = []
        calls: list[float] = []
        puts: list[float] = []
        gammas: list[float] = []
        for strike, x_target in strike_cols:
            c = pick_value(call_nums, x_target)
            p = pick_value(put_nums, x_target)
            strikes.append(float(strike))
            calls.append(float(c))
            puts.append(float(p))
            gammas.append(float((c - p) * 1000))

        return pd.DataFrame({
            'Strike': strikes,
            'Call_OI': calls,
            'Put_OI': puts,
            'Gamma_Exposure': gammas,
        })

    # --- Non-transposed (wide) matrix parser ---

    # Locate the C/P header row (row with the most C/P tokens).
    bins: dict[int, list[tuple[float, float, float, str]]] = {}
    for x0, y0, x1, t in cp_words:
        key = int(round(y0 / 2.0))
        bins.setdefault(key, []).append((x0, y0, x1, t))
    best_key = max(bins.keys(), key=lambda k: len(bins[k]))
    header_cp = bins[best_key]
    cp_y = sum(y0 for _, y0, _, _ in header_cp) / len(header_cp)

    # Build ordered list of (C,P) column x-centers from header row.
    cp_entries: list[tuple[str, float]] = []
    for x0, y0, x1, t in header_cp:
        if abs(y0 - cp_y) > 4.0:
            continue
        x_center = (x0 + x1) / 2.0
        if t == 'CP':
            cp_entries.append(('C', x_center - 1.0))
            cp_entries.append(('P', x_center + 1.0))
        else:
            cp_entries.append((t, x_center))
    cp_entries.sort(key=lambda it: it[1])
    if len(cp_entries) < 2:
        return pd.DataFrame()

    cp_pairs: list[tuple[float, float]] = []
    i = 0
    while i + 1 < len(cp_entries):
        t1, x1 = cp_entries[i]
        t2, x2 = cp_entries[i + 1]
        if t1 == 'C' and t2 == 'P':
            cp_pairs.append((x1, x2))
            i += 2
            continue
        i += 1
    if not cp_pairs:
        return pd.DataFrame()

    # Build day -> (call_x, put_x) mapping from the PDF text order (PyPDF2).
    days_order = _find_dte_days_order_pypdf2(pdf_path)
    day_to_pair: dict[int, tuple[float, float]] = {}
    if days_order and len(days_order) == len(cp_pairs):
        for d, (call_x, put_x) in zip(days_order, cp_pairs):
            day_to_pair[int(d)] = (float(call_x), float(put_x))
    elif days_order:
        # If the counts don't match, still map sequentially up to the shortest.
        for d, (call_x, put_x) in zip(days_order, cp_pairs):
            day_to_pair[int(d)] = (float(call_x), float(put_x))
    else:
        # Last resort: treat the first pair as 1DTE, second as 2DTE, ...
        for idx, (call_x, put_x) in enumerate(cp_pairs, start=1):
            day_to_pair[int(idx)] = (float(call_x), float(put_x))

    pair = day_to_pair.get(int(target_days))
    if not pair:
        return pd.DataFrame()
    call_x, put_x = pair

    # Parse data rows below the C/P header row.
    min_data_y = cp_y + 6.0
    strike_x_threshold = min(call_x, put_x) - 10.0

    numeric_words: list[tuple[float, float, float, float, str]] = []
    for x0, y0, x1, y1, t in words:
        if y0 < min_data_y:
            continue
        if not _is_number_token(t):
            continue
        numeric_words.append((x0, y0, x1, y1, t))
    if not numeric_words:
        return pd.DataFrame()
    numeric_words.sort(key=lambda it: (it[1], it[0]))

    # Group into rows by y.
    rows: list[list[tuple[float, float, float, float, str]]] = []
    for w in numeric_words:
        if not rows or abs(w[1] - rows[-1][0][1]) > 3.0:
            rows.append([w])
        else:
            rows[-1].append(w)

    strikes: list[float] = []
    calls: list[float] = []
    puts: list[float] = []
    gammas: list[float] = []

    for row in rows:
        row_sorted = sorted(row, key=lambda it: it[0])
        # Pick strike as left-most plausible 4-digit value on the left side.
        strike_candidates = [w for w in row_sorted if w[0] <= strike_x_threshold]
        if not strike_candidates:
            continue
        strike_word = strike_candidates[0]
        strike_val = _parse_number(strike_word[4])
        if strike_val <= 0:
            continue

        def pick_near(target_x: float) -> float:
            best = None
            best_dist = None
            for x0, y0, x1, y1, t in row_sorted:
                xc = (x0 + x1) / 2.0
                dist = abs(xc - target_x)
                if dist > 12.0:
                    continue
                if best_dist is None or dist < best_dist:
                    best = t
                    best_dist = dist
            return _parse_number(best) if best is not None else 0.0

        call = pick_near(call_x)
        put = pick_near(put_x)
        gamma = (call - put) * 1000

        strikes.append(float(strike_val))
        calls.append(float(call))
        puts.append(float(put))
        gammas.append(float(gamma))

    if not strikes:
        return pd.DataFrame()

    return pd.DataFrame({
        'Strike': strikes,
        'Call_OI': calls,
        'Put_OI': puts,
        'Gamma_Exposure': gammas,
    })


def _extract_dte_pair_data(pdf_path: str, call_col: int, put_col: int) -> pd.DataFrame:
    """Estrae una coppia Call/Put da una Open Interest Matrix usando indici colonna."""

    def _to_float(value: object) -> float:
        return _parse_pdf_number(value)

    def _is_strike(value: object) -> bool:
        try:
            raw = ("" if value is None else str(value)).strip()
            if not raw:
                return False
            parsed = _parse_pdf_number(raw)
            return parsed != 0.0 or any(ch.isdigit() for ch in raw)
        except Exception:
            return False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            for table in tables:
                if not table or len(table) < 3:
                    continue

                max_len = max(len(r) for r in table)
                norm = [r + [""] * (max_len - len(r)) for r in table]
                df = pd.DataFrame(norm)

                # Trova la riga con "STRIKE" (può non essere solo nella prima cella)
                strike_row = None
                for idx, row in df.iterrows():
                    joined = " ".join(str(x) for x in row.tolist())
                    if 'STRIKE' in joined.upper():
                        strike_row = idx
                        break

                if strike_row is None:
                    continue

                # Trova prima riga dati dopo header (prima colonna numerica)
                data_start = None
                for ridx in range(strike_row + 1, len(df)):
                    if _is_strike(df.iloc[ridx, 0]):
                        data_start = ridx
                        break

                if data_start is None:
                    continue

                strikes: list[float] = []
                calls: list[float] = []
                puts: list[float] = []
                gammas: list[float] = []

                for ridx in range(data_start, len(df)):
                    try:
                        row = df.iloc[ridx]
                        if not _is_strike(row.iloc[0]):
                            continue

                        strike = _to_float(row.iloc[0])
                        call = _to_float(row.iloc[call_col]) if call_col < len(row) else 0.0
                        put = _to_float(row.iloc[put_col]) if put_col < len(row) else 0.0
                        gamma = (call - put) * 1000

                        strikes.append(strike)
                        calls.append(call)
                        puts.append(put)
                        gammas.append(gamma)
                    except Exception:
                        continue

                if strikes:
                    return pd.DataFrame({
                        'Strike': strikes,
                        'Call_OI': calls,
                        'Put_OI': puts,
                        'Gamma_Exposure': gammas
                    })

    return pd.DataFrame()


def _find_dte_column_mapping(pdf_path: str) -> Dict[int, tuple[int, int]]:
    """Ritorna mappa {dte_days: (call_col, put_col)} rilevata dalla tabella.

    Supporta intestazioni tipo "EWZ5\n0 DTE" con celle vuote/None tra Call e Put.
    """

    def _parse_dte_days(cell: object) -> Optional[int]:
        if cell is None:
            return None
        text = str(cell).upper().replace('\n', ' ')
        m = re.search(r'\b(\d+)\s*DTE\b', text)
        if not m:
            m = re.search(r'\b(\d+)\s*DAYS?\b', text)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 3:
                    continue

                max_len = max(len(r) for r in table)
                norm = [r + [""] * (max_len - len(r)) for r in table]
                df = pd.DataFrame(norm)

                # find STRIKE row
                strike_row = None
                for idx, row in df.iterrows():
                    joined = " ".join(str(x) for x in row.tolist())
                    if 'STRIKE' in joined.upper():
                        strike_row = idx
                        break
                if strike_row is None:
                    continue

                # prefer next row for C/P labels
                cp_row_idx = strike_row + 1
                if cp_row_idx >= len(df):
                    continue

                # determine day label per column from STRIKE header row, propagating across blanks
                days_by_col: list[Optional[int]] = [None] * df.shape[1]
                current_days: Optional[int] = None
                for col in range(df.shape[1]):
                    parsed = _parse_dte_days(df.iloc[strike_row, col])
                    if parsed is not None:
                        current_days = parsed
                    days_by_col[col] = current_days

                # map day -> call/put columns based on C/P row
                mapping: Dict[int, Dict[str, int]] = {}
                cp_row = df.iloc[cp_row_idx]
                for col in range(df.shape[1]):
                    d = days_by_col[col]
                    if d is None:
                        continue
                    cp = str(cp_row.iloc[col] or '').strip().upper()
                    if cp not in {'C', 'P'}:
                        continue
                    mapping.setdefault(int(d), {})
                    # keep the first occurrence (leftmost)
                    if cp == 'C' and 'C' not in mapping[int(d)]:
                        mapping[int(d)]['C'] = col
                    if cp == 'P' and 'P' not in mapping[int(d)]:
                        mapping[int(d)]['P'] = col

                # finalize only complete pairs
                out: Dict[int, tuple[int, int]] = {}
                for d, cols in mapping.items():
                    if 'C' in cols and 'P' in cols:
                        out[int(d)] = (int(cols['C']), int(cols['P']))

                if out:
                    return out

    return {}


def analyze_0dte(df: pd.DataFrame, current_price: float = None):
    """Analizza i dati 0DTE e restituisce risultati strutturati"""
    
    if df.empty:
        return {'error': 'Nessun dato 0DTE trovato'}
    
    results = {
        'current_price': current_price,
        'gamma_flip': None,
        'gamma_flip_zone': None,
        'supports': [],
        'resistances': [],
        'stats': {}
    }

    # Sort by strike
    df_sorted = df.sort_values('Strike').reset_index(drop=True)

    strikes = df_sorted['Strike'].astype(float).tolist()

    flip_low = None
    flip_high = None
    flip_zone_low = None
    flip_zone_high = None

    # 1) Preferred: "around price" operational flip.
    # Pick the strike ABOVE current price (within +30pts) where |Call_OI - Put_OI| is minimal.
    if current_price is not None:
        cp = float(current_price)
        window_high = cp + 30.0
        window_df = df_sorted[(df_sorted['Strike'] > cp) & (df_sorted['Strike'] <= window_high)].copy()
        if not window_df.empty:
            window_df['abs_net'] = (window_df['Call_OI'] - window_df['Put_OI']).abs()
            best_idx = window_df['abs_net'].idxmin()
            best_pos = int(df_sorted.index[df_sorted['Strike'] == float(window_df.loc[best_idx, 'Strike'])][0])

            best_strike = float(df_sorted.loc[best_pos, 'Strike'])
            prev_strike = float(df_sorted.loc[max(0, best_pos - 1), 'Strike'])
            next_strike = float(df_sorted.loc[min(len(df_sorted) - 1, best_pos + 1), 'Strike'])

            flip_low = prev_strike
            flip_high = best_strike
            flip_zone_low = prev_strike
            flip_zone_high = next_strike

    # 2) Fallback: local balance sign-change method.
    if flip_zone_low is None or flip_zone_high is None:
        W_POINTS = 25.0
        balances = []
        for s in strikes:
            puts_below = float(df_sorted[(df_sorted['Strike'] >= s - W_POINTS) & (df_sorted['Strike'] <= s)]['Put_OI'].sum())
            calls_above = float(df_sorted[(df_sorted['Strike'] >= s) & (df_sorted['Strike'] <= s + W_POINTS)]['Call_OI'].sum())
            balances.append(calls_above - puts_below)

        sign_change_candidates = []
        for i in range(1, len(strikes)):
            a = float(balances[i - 1])
            b = float(balances[i])
            if a == 0 or b == 0 or (a < 0 < b) or (a > 0 > b):
                stability = min(abs(a), abs(b))
                mid = (float(strikes[i - 1]) + float(strikes[i])) / 2
                dist = abs(mid - float(current_price)) if current_price is not None else 0.0
                sign_change_candidates.append((stability, -dist, i))

        if sign_change_candidates:
            sign_change_candidates.sort(reverse=True)
            _, _, i = sign_change_candidates[0]
            flip_low = float(strikes[i - 1])
            flip_high = float(strikes[i])

            right = float(strikes[i])
            next_strike = float(strikes[i + 1]) if (i + 1) < len(strikes) else right
            flip_zone_low = right
            flip_zone_high = next_strike

    if flip_low is not None and flip_high is not None:
        if flip_zone_low is not None and flip_zone_high is not None:
            zone_low = round(min(flip_zone_low, flip_zone_high), 2)
            zone_high = round(max(flip_zone_low, flip_zone_high), 2)
        else:
            zone_low = round(min(flip_low, flip_high), 2)
            zone_high = round(max(flip_low, flip_high), 2)

        results['gamma_flip_zone'] = {
            'low': zone_low,
            'high': zone_high
        }

        # Operational flip = midpoint of the zone
        gamma_flip = (zone_low + zone_high) / 2
        results['gamma_flip'] = round(gamma_flip, 2)

        # Regime (same semantics, using flip midpoint)
        if current_price is not None and current_price > gamma_flip:
            results['regime'] = 'Positive Gamma (Low Volatility)'
            results['strategy'] = 'Mean reversion - vendere breakout, comprare pullback'
        elif current_price is not None and current_price < gamma_flip:
            results['regime'] = 'Negative Gamma (High Volatility)'
            results['strategy'] = 'Trend following - seguire breakout, evitare fade'
        else:
            results['regime'] = 'At Gamma Flip'
            results['strategy'] = 'Cautela - punto di transizione'

        # 0DTE-style levels
        zone_low = min(results['gamma_flip_zone']['low'], results['gamma_flip_zone']['high'])
        zone_high = max(results['gamma_flip_zone']['low'], results['gamma_flip_zone']['high'])

        below_flip = df_sorted[df_sorted['Strike'] < zone_low].copy()
        # include boundary in resistances (often the first call-wall is exactly on zone_high)
        above_flip = df_sorted[df_sorted['Strike'] >= zone_high].copy()

        def _prefer_25pt_levels(df_levels: pd.DataFrame, side: str) -> pd.DataFrame:
            # Prefer strikes that are multiples of 25 when available (common "walls")
            if df_levels.empty:
                return df_levels
            df_levels = df_levels.copy()
            df_levels['is_25'] = (df_levels['Strike'] % 25 == 0)
            key_col = 'Put_OI' if side == 'put' else 'Call_OI'
            top = df_levels.nlargest(12, key_col)
            preferred = top[top['is_25']]
            if len(preferred) >= 3:
                return preferred.nlargest(3, key_col)
            # fill remaining
            remainder = top[~top['is_25']]
            combined = pd.concat([preferred, remainder], ignore_index=True)
            return combined.nlargest(3, key_col)

        # PUT supports below flip (largest Put OI)
        if not below_flip.empty:
            top_puts = _prefer_25pt_levels(below_flip, side='put')
            results['supports'] = [
                {
                    'strike': float(row['Strike']),
                    'call_oi': int(row['Call_OI']),
                    'put_oi': int(row['Put_OI']),
                    'gamma': int(row['Gamma_Exposure'])
                }
                for _, row in top_puts.iterrows()
            ]
        else:
            results['supports_note'] = 'Nessun livello sotto la zona di flip'

        # CALL resistances above flip (largest Call OI)
        if not above_flip.empty:
            top_calls = _prefer_25pt_levels(above_flip, side='call')
            results['resistances'] = [
                {
                    'strike': float(row['Strike']),
                    'call_oi': int(row['Call_OI']),
                    'put_oi': int(row['Put_OI']),
                    'gamma': int(row['Gamma_Exposure'])
                }
                for _, row in top_calls.iterrows()
            ]
        else:
            results['resistances_note'] = 'Nessun livello sopra la zona di flip'
    else:
        results['gamma_flip_note'] = 'Impossibile determinare gamma flip: nessun incrocio Call/Put trovato'
    
    # Statistiche
    total_calls = df['Call_OI'].sum()
    total_puts = df['Put_OI'].sum()
    
    results['stats'] = {
        'total_strikes': len(df),
        'strike_range': f"{df['Strike'].min():.0f} - {df['Strike'].max():.0f}",
        'total_call_oi': int(total_calls),
        'total_put_oi': int(total_puts),
        'put_call_ratio': round(total_puts / total_calls, 2) if total_calls > 0 else None
    }
    
    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sp500-price', methods=['GET'])
def sp500_price():
    data = get_sp500_price_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare il prezzo S&P 500 in questo momento"}), 503

    return jsonify(data)


@app.route('/api/es-price', methods=['GET'])
def es_price():
    data = get_es_price_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare il prezzo ES in questo momento"}), 503

    return jsonify(data)


@app.route('/api/health', methods=['GET'], endpoint='api_health')
def api_health():
    mongo = _get_mongo_collection()
    return jsonify({
        "status": "ok",
        "pymupdf_available": bool(_PYMUPDF_AVAILABLE),
        "app_build": _APP_BUILD,
        "python": _RUNTIME_PYTHON,
        "in_venv": bool(_IN_VENV),
        "virtual_env": os.getenv("VIRTUAL_ENV"),
        "mongo_configured": mongo is not None,
    })


@app.route('/api/nvda-snapshot', methods=['GET'])
def nvda_snapshot():
    data = get_nvda_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare NVDA option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/spy-snapshot', methods=['GET'])
def spy_snapshot():
    data = get_spy_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare SPY option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/msft-snapshot', methods=['GET'])
def msft_snapshot():
    data = get_msft_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare MSFT option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/spx-snapshot', methods=['GET'])
def spx_snapshot():
    data = get_spx_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare SPX option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/xsp-snapshot', methods=['GET'])
def xsp_snapshot():
    data = get_xsp_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare XSP option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/aapl-snapshot', methods=['GET'])
def aapl_snapshot():
    data = get_aapl_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare AAPL option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/goog-snapshot', methods=['GET'])
def goog_snapshot():
    data = get_goog_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare GOOG option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/amzn-snapshot', methods=['GET'])
def amzn_snapshot():
    data = get_amzn_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare AMZN option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/pressure-history', methods=['GET'])
def pressure_history():
    """Return recent pressure points for chart persistence."""

    coll = _get_mongo_collection()
    if coll is None:
        return jsonify({"error": "MongoDB non configurato"}), 503

    try:
        hours = float(request.args.get('hours', '8') or '8')
    except Exception:
        hours = 8.0
    hours = max(0.25, min(hours, 72.0))

    now_ts = int(time.time())
    since_ts = now_ts - int(hours * 3600)

    try:
        # 8h @ 1 point/sec = 28,800 points. Keep some headroom.
        cursor = coll.find({"ts": {"$gte": since_ts}}).sort("ts", 1).limit(50000)
        points = []
        for doc in cursor:
            points.append({
                "ts": int(doc.get("ts")),
                "score": doc.get("score"),
                "breakdown": doc.get("breakdown"),
            })
        return jsonify({"points": points, "hours": hours, "since_ts": since_ts, "now_ts": now_ts})
    except Exception as e:
        return jsonify({"error": f"Errore MongoDB: {e}"}), 503


@app.route('/api/pressure-point', methods=['POST'])
def pressure_point():
    """Upsert a single pressure point (1-second granularity)."""

    coll = _get_mongo_collection()
    if coll is None:
        return jsonify({"error": "MongoDB non configurato"}), 503

    data = request.get_json(silent=True) or {}
    try:
        ts = int(data.get('ts'))
    except Exception:
        ts = None

    score = data.get('score')
    breakdown = data.get('breakdown')

    if ts is None or score is None:
        return jsonify({"error": "Payload non valido: richiesti ts e score"}), 400

    try:
        coll.update_one(
            {"ts": ts},
            {
                "$set": {
                    "ts": ts,
                    "score": score,
                    "breakdown": breakdown,
                    "updated_at": int(time.time()),
                },
                "$setOnInsert": {"created_at": _dt.datetime.utcnow()},
            },
            upsert=True,
        )
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"error": f"Errore MongoDB: {e}"}), 503


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Solo file PDF sono supportati'}), 400
    
    try:
        # Salva il file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Estrai prezzo corrente se fornito
        current_price = request.form.get('current_price')
        current_price = float(current_price) if current_price else None
        
        # Estrai dati: preferisci 0DTE, fallback a 1DTE; se 1DTE manca, prova la scadenza positiva più vicina.
        # Track attempts to make failures diagnosable in the UI.
        extraction_attempts = []

        def _attempt(label: str, fn):
            t0 = time.time()
            out = fn()
            dt = round(time.time() - t0, 2)
            rows = int(len(out)) if isinstance(out, pd.DataFrame) else 0
            extraction_attempts.append({"label": label, "rows": rows, "seconds": dt})
            return out if isinstance(out, pd.DataFrame) else pd.DataFrame()

        df = _attempt(
            "0DTE-pymupdf",
            lambda: _extract_dte_pair_data_pymupdf(filepath, target_days=0) if _PYMUPDF_AVAILABLE else pd.DataFrame(),
        )
        if df.empty:
            df = _attempt("0DTE-pdfplumber", lambda: _extract_dte_days_data(filepath, target_days=0))
        if df.empty:
            df = _attempt(
                "1DTE-pymupdf",
                lambda: _extract_dte_pair_data_pymupdf(filepath, target_days=1) if _PYMUPDF_AVAILABLE else pd.DataFrame(),
            )
        if df.empty:
            df = _attempt("1DTE-pdfplumber", lambda: _extract_dte_days_data(filepath, target_days=1))
        if df.empty:
            df = _attempt("nearest-positive-dte", lambda: extract_nearest_positive_dte_data(filepath))

        # Analizza
        results = analyze_0dte(df, current_price)

        # Attach extraction details to help explain "no data" situations.
        if isinstance(results, dict):
            results.setdefault('extraction_attempts', extraction_attempts)
            results.setdefault('pymupdf_available', _PYMUPDF_AVAILABLE)
            results.setdefault('python', _RUNTIME_PYTHON)
            results.setdefault('in_venv', bool(_IN_VENV))

        # Messaggio più chiaro se manca sia 0DTE che 1DTE
        if isinstance(results, dict) and results.get('error') == 'Nessun dato 0DTE trovato':
            base = 'Nessun dato 0DTE trovato; ho provato anche 1DTE (e la scadenza positiva più vicina) senza successo'
            if not _PYMUPDF_AVAILABLE:
                base += ' (nota: PyMuPDF/fitz non disponibile; avvia l\'app nel tuo .venv o installa le dipendenze)'
                base += f" [python={_RUNTIME_PYTHON}]"
            results['error'] = base
        
        # Rimuovi il file temporaneo
        os.remove(filepath)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Errore durante l\'analisi: {str(e)}'}), 500


if __name__ == '__main__':
    port_env = os.getenv('PORT')
    try:
        port = int(port_env) if port_env else 5005
    except ValueError:
        port = 5005
    app.run(debug=True, port=port)
