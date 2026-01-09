"""
Flask web application per analisi gamma exposure 0DTE
"""

# ============================================================================
# IMPORTS
# ============================================================================

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
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
from werkzeug.middleware.proxy_fix import ProxyFix
import re
import importlib.util
import sys
from functools import wraps
import uuid

try:
    from authlib.integrations.flask_client import OAuth
except Exception:  # pragma: no cover
    OAuth = None

# Optional: load local .env for development (no-op if not installed / not present).
# Load it from this file's directory so it works regardless of current working directory.
try:  # pragma: no cover
    from dotenv import load_dotenv

    _dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    load_dotenv(dotenv_path=_dotenv_path, override=False)
except Exception:
    pass

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover
    MongoClient = None

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

# ============================================================================
# CONFIGURATION & GLOBALS
# ============================================================================

_PYMUPDF_AVAILABLE = importlib.util.find_spec("fitz") is not None
_RUNTIME_PYTHON = sys.executable
_IN_VENV = getattr(sys, "base_prefix", sys.prefix) != sys.prefix
try:
    _APP_BUILD = int(os.path.getmtime(__file__))
except Exception:
    _APP_BUILD = None

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Behind reverse proxies (e.g., Vercel), trust forwarded headers so url_for(..., _external=True)
# produces the correct https://<host>/... callback URLs.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Session secret (required for OAuth login). In production, set FLASK_SECRET_KEY.
_secret_from_env = (os.getenv('FLASK_SECRET_KEY') or os.getenv('SECRET_KEY') or '').strip()
app.secret_key = _secret_from_env or 'dev-secret-key-change-me'

# Basic cookie hardening (safe defaults; secure cookie should be enabled behind HTTPS).
app.config.setdefault('SESSION_COOKIE_HTTPONLY', True)
app.config.setdefault('SESSION_COOKIE_SAMESITE', 'Lax')
if os.getenv('VERCEL'):
    app.config['SESSION_COOKIE_SECURE'] = True

oauth = OAuth(app) if OAuth is not None else None

# ============================================================================
# AUTHENTICATION & SESSION MANAGEMENT
# ============================================================================

def _ensure_google_oauth_registered() -> bool:
    """Register the Google OAuth client if possible.

    This is intentionally lazy so that config changes in .env + server restart
    are reflected reliably, even with Flask's reloader.
    """

    global oauth
    if oauth is None:
        return False
    if hasattr(oauth, 'google'):
        return True

    google_client_id = (os.getenv('GOOGLE_CLIENT_ID') or '').strip()
    google_client_secret = (os.getenv('GOOGLE_CLIENT_SECRET') or '').strip()
    secret_from_env = (os.getenv('FLASK_SECRET_KEY') or os.getenv('SECRET_KEY') or '').strip()

    if not (secret_from_env and google_client_id and google_client_secret):
        return False

    try:
        oauth.register(
            name='google',
            client_id=google_client_id,
            client_secret=google_client_secret,
            server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
            client_kwargs={'scope': 'openid email profile'},
        )
        return hasattr(oauth, 'google')
    except Exception:
        return False


def _google_oauth_missing_vars():
    missing = []
    if not (os.getenv('GOOGLE_CLIENT_ID') or '').strip():
        missing.append('GOOGLE_CLIENT_ID')
    if not (os.getenv('GOOGLE_CLIENT_SECRET') or '').strip():
        missing.append('GOOGLE_CLIENT_SECRET')
    if not ((os.getenv('FLASK_SECRET_KEY') or os.getenv('SECRET_KEY') or '').strip()):
        missing.append('FLASK_SECRET_KEY')
    return missing


def _is_authenticated() -> bool:
    return bool(session.get('user'))


def _is_admin() -> bool:
    """Return True if the current user can access admin pages.

    If ADMIN_EMAILS is set (comma-separated list), only those emails are allowed.
    If not set, any authenticated user is allowed (useful for single-user deployments).
    """

    if not _is_authenticated():
        return False

    admin_emails_raw = (os.getenv('ADMIN_EMAILS') or '').strip()
    # On Vercel, require an explicit allowlist.
    if os.getenv('VERCEL') and not admin_emails_raw:
        return False
    if not admin_emails_raw:
        return True

    allowed = {e.strip().lower() for e in admin_emails_raw.split(',') if e.strip()}
    user = session.get('user') or {}
    email = (user.get('email') if isinstance(user, dict) else None) or ''
    return email.strip().lower() in allowed


def _wants_json() -> bool:
    accept = (request.headers.get('Accept') or '').lower()
    return request.path.startswith('/api/') or request.path == '/analyze' or 'application/json' in accept


@app.before_request
def _require_login():
    # Allow preflight
    if request.method == 'OPTIONS':
        return None

    path = request.path or '/'
    public_prefixes = ('/login', '/logout', '/auth')
    if path == '/api/health' or path.startswith(public_prefixes) or path.startswith('/static'):
        return None

    if _is_authenticated():
        return None

    if _wants_json():
        return jsonify({'error': 'Unauthorized'}), 401

    # Store next and redirect to login
    try:
        session['next_url'] = request.full_path if request.full_path else path
    except Exception:
        session['next_url'] = path
    return redirect(url_for('login'))


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if _is_authenticated():
            return fn(*args, **kwargs)
        if _wants_json():
            return jsonify({'error': 'Unauthorized'}), 401
        session['next_url'] = request.full_path or request.path or '/'
        return redirect(url_for('login'))

    return wrapper


# ============================================================================
# MONGODB HELPERS
# ============================================================================

_MONGO_CLIENT: Optional["MongoClient"] = None
_MONGO_COLLECTION = None
_MONGO_LOGIN_COLLECTION = None
_MONGO_LAST_ANALYSIS_COLLECTION = None
_MONGO_GAMMA_STATS_COLLECTION = None


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


def _get_mongo_login_collection():
    """Return Mongo collection for login sessions or None if not configured/available."""

    global _MONGO_CLIENT, _MONGO_LOGIN_COLLECTION
    if _MONGO_LOGIN_COLLECTION is not None:
        return _MONGO_LOGIN_COLLECTION

    if MongoClient is None:
        return None

    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None

    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_LOGIN_COLLECTION") or "login_sessions").strip()

    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)

        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]

        # TTL to avoid unbounded growth (default 90 days). Requires a datetime field.
        ttl_days = os.getenv("LOGIN_SESSIONS_TTL_DAYS")
        try:
            ttl = int(ttl_days) if ttl_days else 90
            if ttl > 0:
                coll.create_index("created_at", expireAfterSeconds=60 * 60 * 24 * ttl)
        except Exception:
            pass

        # Helpful query indexes
        try:
            coll.create_index([("user.email", 1), ("created_at", -1)])
        except Exception:
            pass
        try:
            coll.create_index([("user.sub", 1), ("created_at", -1)])
        except Exception:
            pass

        _MONGO_LOGIN_COLLECTION = coll
        return _MONGO_LOGIN_COLLECTION
    except Exception:
        return None


def _log_login_event(event_type: str, user: Optional[dict] = None, extra: Optional[dict] = None) -> None:
    """Best-effort logging of auth events to MongoDB (no-op if not configured)."""

    coll = _get_mongo_login_collection()
    if coll is None:
        return

    try:
        login_session_id = session.get('login_session_id')
        if not login_session_id:
            login_session_id = str(uuid.uuid4())
            session['login_session_id'] = login_session_id

        doc = {
            "event": event_type,
            "login_session_id": login_session_id,
            "created_at": _dt.datetime.utcnow(),
            "ts": int(time.time()),
            "user": (user if isinstance(user, dict) else session.get('user')),
            "ip": request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or request.remote_addr,
            "user_agent": request.headers.get('User-Agent'),
        }
        if extra and isinstance(extra, dict):
            doc["extra"] = extra
        coll.insert_one(doc)
    except Exception:
        # Never break login/logout due to logging.
        return


def _current_user_key() -> Optional[str]:
    user = session.get('user')
    if not isinstance(user, dict):
        return None
    sub = (user.get('sub') or '').strip()
    if sub:
        return f"google:{sub}"
    email = (user.get('email') or '').strip().lower()
    if email:
        return f"email:{email}"
    return None


def _get_mongo_last_analysis_collection():
    """Return Mongo collection for per-user last analysis or None if not configured."""

    global _MONGO_CLIENT, _MONGO_LAST_ANALYSIS_COLLECTION
    if _MONGO_LAST_ANALYSIS_COLLECTION is not None:
        return _MONGO_LAST_ANALYSIS_COLLECTION

    if MongoClient is None:
        return None

    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None

    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_LAST_ANALYSIS_COLLECTION") or "last_analysis").strip()

    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)

        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]

        try:
            coll.create_index("user_key", unique=True)
        except Exception:
            pass

        _MONGO_LAST_ANALYSIS_COLLECTION = coll
        return _MONGO_LAST_ANALYSIS_COLLECTION
    except Exception:
        return None


def _save_last_analysis(filename: str, analysis: dict) -> None:
    coll = _get_mongo_last_analysis_collection()
    if coll is None:
        return

    user_key = _current_user_key()
    if not user_key:
        return

    try:
        # Ensure Mongo-safe JSON primitives.
        payload = json.loads(json.dumps(analysis, default=str))
    except Exception:
        payload = analysis

    doc = {
        "user_key": user_key,
        "user": session.get('user'),
        "filename": filename,
        "updated_at": _dt.datetime.utcnow(),
        "analysis": payload,
    }

    try:
        coll.replace_one({"user_key": user_key}, doc, upsert=True)
    except Exception:
        return


def _load_last_analysis() -> Optional[dict]:
    coll = _get_mongo_last_analysis_collection()
    if coll is None:
        return None
    user_key = _current_user_key()
    if not user_key:
        return None
    try:
        doc = coll.find_one({"user_key": user_key})
        if not doc:
            return None
        return doc
    except Exception:
        return None


def _get_mongo_gamma_stats_collection():
    """Return Mongo collection for gamma statistics tracking or None if not configured."""
    
    global _MONGO_CLIENT, _MONGO_GAMMA_STATS_COLLECTION
    if _MONGO_GAMMA_STATS_COLLECTION is not None:
        return _MONGO_GAMMA_STATS_COLLECTION
    
    if MongoClient is None:
        return None
    
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    
    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_GAMMA_STATS_COLLECTION") or "gamma_statistics").strip()
    
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        
        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]
        
        # Indici per query efficienti
        try:
            coll.create_index([("strike", 1), ("timestamp", -1)])
            coll.create_index("timestamp")
        except Exception:
            pass
        
        _MONGO_GAMMA_STATS_COLLECTION = coll
        return _MONGO_GAMMA_STATS_COLLECTION
    except Exception:
        return None


def _save_gamma_statistics(supports: list, resistances: list, price: float = None) -> None:
    """Salva statistiche gamma nel database per tracking storico."""
    coll = _get_mongo_gamma_stats_collection()
    if coll is None:
        return
    
    timestamp = _dt.datetime.utcnow()
    user_key = _current_user_key()
    
    # Salva ogni livello con le sue statistiche
    documents = []
    
    for level in supports:
        if isinstance(level, dict):
            doc = {
                "strike": float(level.get("strike", 0)),
                "type": "support",
                "gamma": float(level.get("gamma", 0)),
                "call_oi": float(level.get("call_oi", 0)),
                "put_oi": float(level.get("put_oi", 0)),
                "timestamp": timestamp,
                "user_key": user_key,
                "current_price": float(price) if price else None,
            }
            documents.append(doc)
    
    for level in resistances:
        if isinstance(level, dict):
            doc = {
                "strike": float(level.get("resistance", 0) or level.get("strike", 0)),
                "type": "resistance",
                "gamma": float(level.get("gamma", 0)),
                "call_oi": float(level.get("call_oi", 0)),
                "put_oi": float(level.get("put_oi", 0)),
                "timestamp": timestamp,
                "user_key": user_key,
                "current_price": float(price) if price else None,
            }
            documents.append(doc)
    
    if documents:
        try:
            coll.insert_many(documents)
        except Exception:
            pass


def _get_gamma_statistics(strike: float, days_back: int = 30) -> dict:
    """Recupera statistiche storiche per uno strike specifico."""
    coll = _get_mongo_gamma_stats_collection()
    if coll is None:
        return {}
    
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days_back)
    
    try:
        docs = list(coll.find({
            "strike": {"$gte": strike - 5, "$lte": strike + 5},  # Range di 10 punti
            "timestamp": {"$gte": cutoff}
        }).sort("timestamp", -1).limit(100))
        
        if not docs:
            return {}
        
        gammas = [abs(d.get("gamma", 0)) for d in docs]
        
        return {
            "count": len(gammas),
            "avg_gamma": sum(gammas) / len(gammas) if gammas else 0,
            "max_gamma": max(gammas) if gammas else 0,
            "min_gamma": min(gammas) if gammas else 0,
            "recent_gamma": gammas[0] if gammas else 0,
        }
    except Exception:
        return {}


def _get_top_gamma_levels(limit: int = 10, days_back: int = 7) -> list:
    """Recupera i livelli con i gamma più alti degli ultimi giorni."""
    coll = _get_mongo_gamma_stats_collection()
    if coll is None:
        return []
    
    cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days_back)
    
    try:
        # Aggregazione per ottenere i gamma medi per strike
        pipeline = [
            {"$match": {"timestamp": {"$gte": cutoff}}},
            {"$group": {
                "_id": "$strike",
                "avg_gamma": {"$avg": {"$abs": "$gamma"}},
                "count": {"$sum": 1},
                "type": {"$first": "$type"}
            }},
            {"$sort": {"avg_gamma": -1}},
            {"$limit": limit}
        ]
        
        results = list(coll.aggregate(pipeline))
        
        return [{
            "strike": r["_id"],
            "avg_gamma": r["avg_gamma"],
            "count": r["count"],
            "type": r.get("type", "unknown")
        } for r in results]
    except Exception:
        return []

    try:
        login_session_id = session.get('login_session_id')
        if not login_session_id:
            login_session_id = str(uuid.uuid4())
            session['login_session_id'] = login_session_id

        doc = {
            "event": event_type,
            "login_session_id": login_session_id,
            "created_at": _dt.datetime.utcnow(),
            "ts": int(time.time()),
            "user": (user if isinstance(user, dict) else session.get('user')),
            "ip": request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or request.remote_addr,
            "user_agent": request.headers.get('User-Agent'),
        }
        if extra and isinstance(extra, dict):
            doc["extra"] = extra
        coll.insert_one(doc)
    except Exception:
        # Never break login/logout due to logging.
        return


# ============================================================================
# FILE SYSTEM HELPERS
# ============================================================================

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

# ============================================================================
# CACHE GLOBALS (Market Data)
# ============================================================================

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

# ============================================================================
# DATA PARSING & EXTRACTION UTILITIES
# ============================================================================


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


def _parse_nasdaq_price(value: object) -> float:
    """Parse price strings coming from Nasdaq `lastTrade`.

    Nasdaq prices should be treated as US-style numbers:
    - '.' is decimal separator
    - ',' (if present) is thousands separator

    This avoids mis-parsing values like '187.285' as '187285'.
    """

    raw = ("" if value is None else str(value)).strip()
    if not raw or raw.lower() in {"none", "nan", ""}:
        return 0.0

    raw = raw.replace("\u00a0", "").replace(" ", "")
    raw = raw.replace("$", "")
    raw = re.sub(r"[^0-9,\.\-]", "", raw)
    if not raw or raw in {"-", ".", ","}:
        return 0.0

    # Nasdaq should be US formatted: commas are thousands, dot is decimal.
    raw = raw.replace(",", "")
    try:
        return float(raw)
    except Exception:
        # Fallback to the generic parser as a last resort.
        return _parse_pdf_number(value)


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


def _fetch_yahoo_options(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch options data from Yahoo Finance using yfinance library."""
    if not yf:
        print("[DEBUG] yfinance library not available")
        return None
    
    print(f"[DEBUG] Fetching Yahoo Finance options for {symbol} using yfinance")
    try:
        ticker = yf.Ticker(symbol)
        
        # Get current price
        info = ticker.info
        current_price = info.get("regularMarketPrice") or info.get("currentPrice")
        
        # Get available expiration dates
        expirations = ticker.options
        if not expirations:
            print(f"[DEBUG] No expirations found for {symbol}")
            return None
        
        # Get nearest expiration (0DTE or closest)
        nearest_exp = expirations[0]
        print(f"[DEBUG] Using expiration: {nearest_exp}")
        
        # Get options chain for nearest expiration
        opt_chain = ticker.option_chain(nearest_exp)
        calls_df = opt_chain.calls
        puts_df = opt_chain.puts
        
        # Filter strikes: keep only 15 above and 15 below current price
        if current_price:
            # Get all unique strikes
            all_strikes = sorted(set(calls_df['strike'].tolist() + puts_df['strike'].tolist()))
            
            # Find strikes around current price
            strikes_below = [s for s in all_strikes if s < current_price][-15:]  # Last 15 below
            strikes_above = [s for s in all_strikes if s >= current_price][:15]  # First 15 above
            relevant_strikes = set(strikes_below + strikes_above)
            
            # Filter calls and puts
            calls_df = calls_df[calls_df['strike'].isin(relevant_strikes)]
            puts_df = puts_df[puts_df['strike'].isin(relevant_strikes)]
            
            print(f"[DEBUG] Filtered to {len(relevant_strikes)} strikes around price {current_price}")
        
        print(f"[DEBUG] Yahoo Finance fetch SUCCESS - {len(calls_df)} calls, {len(puts_df)} puts")
        
        # Convert to the expected format
        return {
            "symbol": symbol,
            "price": current_price,
            "expiration": nearest_exp,
            "calls": calls_df.to_dict('records'),
            "puts": puts_df.to_dict('records'),
        }
    except Exception as e:
        print(f"[DEBUG] Yahoo Finance fetch FAILED: {e}")
        import traceback
        traceback.print_exc()
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
            last_sale_price = _parse_nasdaq_price(m.group(1))
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

    results = analyze_0dte(
        df,
        current_price=float(last_sale_price) if last_sale_price else None,
        prefer_strike_multiple=None,
    )
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

# ============================================================================
# MARKET DATA FETCHERS (NASDAQ Options & Stocks)
# ============================================================================


def get_nvda_snapshot_cached(max_age_seconds: int = 60, levels_mode: str = "price") -> Optional[Dict[str, Any]]:
    """Fetch NVDA last price + option-chain derived gamma flip for the nearest expiry."""

    now_ts = time.time()
    requested = (levels_mode or "price").strip().lower()
    mode_key = "flip" if requested in {"flip", "gamma", "gamma_flip", "flip_zone"} else "price"

    fetched_at = float(_NVDA_SNAPSHOT_CACHE.get("fetched_at") or 0.0)
    if (now_ts - fetched_at) <= max_age_seconds:
        by_mode = _NVDA_SNAPSHOT_CACHE.get("value_by_mode")
        if isinstance(by_mode, dict) and by_mode.get(mode_key):
            return by_mode.get(mode_key)

        cached = _NVDA_SNAPSHOT_CACHE.get("value")
        if isinstance(cached, dict) and (cached.get("levels_mode") == mode_key or cached.get("levels_mode_requested") == mode_key):
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
            last_sale_price = _parse_nasdaq_price(m.group(1))
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

    base_snapshot: Dict[str, Any] = {
        "symbol": "NVDA",
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    # Precompute both variants so the frontend toggle doesn't trigger extra network calls.
    by_mode: Dict[str, Any] = {}
    for m in ("price", "flip"):
        results = analyze_0dte(
            df,
            current_price=float(last_sale_price) if last_sale_price else None,
            levels_mode=m,
            prefer_strike_multiple=None,
        )
        snapshot = dict(base_snapshot)
        if isinstance(results, dict):
            snapshot.update(results)
        by_mode[m] = snapshot

    _NVDA_SNAPSHOT_CACHE["value_by_mode"] = by_mode
    _NVDA_SNAPSHOT_CACHE["value"] = by_mode.get(mode_key) or by_mode.get("price")
    _NVDA_SNAPSHOT_CACHE["fetched_at"] = now_ts
    return _NVDA_SNAPSHOT_CACHE["value"]


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
            last_sale_price = _parse_nasdaq_price(m.group(1))
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


def get_msft_snapshot_cached(max_age_seconds: int = 60, levels_mode: str = "price") -> Optional[Dict[str, Any]]:
    """Fetch MSFT last price + option-chain derived gamma flip for the nearest expiry."""

    now_ts = time.time()
    requested = (levels_mode or "price").strip().lower()
    mode_key = "flip" if requested in {"flip", "gamma", "gamma_flip", "flip_zone"} else "price"

    fetched_at = float(_MSFT_SNAPSHOT_CACHE.get("fetched_at") or 0.0)
    if (now_ts - fetched_at) <= max_age_seconds:
        by_mode = _MSFT_SNAPSHOT_CACHE.get("value_by_mode")
        if isinstance(by_mode, dict) and by_mode.get(mode_key):
            return by_mode.get(mode_key)

        cached = _MSFT_SNAPSHOT_CACHE.get("value")
        if isinstance(cached, dict) and (cached.get("levels_mode") == mode_key or cached.get("levels_mode_requested") == mode_key):
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
            last_sale_price = _parse_nasdaq_price(m.group(1))
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

    base_snapshot: Dict[str, Any] = {
        "symbol": "MSFT",
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    by_mode: Dict[str, Any] = {}
    for m in ("price", "flip"):
        results = analyze_0dte(
            df,
            current_price=float(last_sale_price) if last_sale_price else None,
            levels_mode=m,
            prefer_strike_multiple=None,
        )
        snapshot = dict(base_snapshot)
        if isinstance(results, dict):
            snapshot.update(results)
        by_mode[m] = snapshot

    _MSFT_SNAPSHOT_CACHE["value_by_mode"] = by_mode
    _MSFT_SNAPSHOT_CACHE["value"] = by_mode.get(mode_key) or by_mode.get("price")
    _MSFT_SNAPSHOT_CACHE["fetched_at"] = now_ts
    return _MSFT_SNAPSHOT_CACHE["value"]


def get_spx_snapshot_cached(metric: str = 'volume', max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch SPX last price + option-chain derived gamma flip for the nearest expiry.

    Yahoo Finance data is fetched only at 8:00 AM and 2:30 PM ET to avoid rate limits.
    Between these times, cached data is served.
    
    Args:
        metric: 'volume' or 'openInterest' - which metric to use for level calculation
        max_age_seconds: maximum age of cached data in seconds
    """

    now_ts = time.time()
    now_dt = _dt.datetime.now()
    
    # Check if we should fetch new data (only at 8:00 AM or 2:30 PM)
    should_fetch = False
    current_hour = now_dt.hour
    current_minute = now_dt.minute
    
    # 8:00 AM window (8:00-8:05)
    if current_hour == 8 and current_minute < 5:
        should_fetch = True
    # 2:30 PM window (14:30-14:35)
    elif current_hour == 14 and 30 <= current_minute < 35:
        should_fetch = True
    
    # Use metric-specific cache key
    cache_key = f"value_{metric}"
    fetched_at_key = f"fetched_at_{metric}"
    
    cached = _SPX_SNAPSHOT_CACHE.get(cache_key)
    fetched_at = float(_SPX_SNAPSHOT_CACHE.get(fetched_at_key) or 0.0)
    
    # If we're in a fetch window and haven't fetched recently (within 5 minutes)
    if should_fetch and (now_ts - fetched_at) > 300:
        print(f"[DEBUG] SPX scheduled fetch time: {current_hour}:{current_minute:02d} with metric={metric}")
        pass  # Continue to fetch
    # Otherwise return cached data if available
    elif cached:
        print(f"[DEBUG] Using cached SPX data with metric={metric} (fetched {int((now_ts - fetched_at)/60)} minutes ago)")
        return cached

    # Try Yahoo Finance first
    yahoo_data = _fetch_yahoo_options("^SPX")
    print(f"[DEBUG] Yahoo Finance data received: {yahoo_data is not None}")
    if yahoo_data:
        print(f"[DEBUG] Yahoo data keys: {yahoo_data.keys()}")
        try:
            calls = yahoo_data.get("calls", [])
            puts = yahoo_data.get("puts", [])
            last_price = yahoo_data.get("price")
            expiration_str = yahoo_data.get("expiration")
            
            if calls and puts and last_price:
                # Parse expiration date (YYYY-MM-DD format from yfinance)
                expiration_date = _dt.datetime.strptime(expiration_str, "%Y-%m-%d").date()
                
                strikes = []
                call_ois = []
                put_ois = []
                gammas = []
                
                # Combina calls e puts per strike - usa metric (volume o openInterest)
                strike_data = {}
                for call in calls:
                    strike = float(call.get("strike", 0))
                    if strike > 0:
                        strike_data[strike] = {
                            "call_oi": float(call.get(metric, 0) or 0),
                            "put_oi": 0
                        }
                
                for put in puts:
                    strike = float(put.get("strike", 0))
                    if strike > 0:
                        if strike in strike_data:
                            strike_data[strike]["put_oi"] = float(put.get(metric, 0) or 0)
                        else:
                            strike_data[strike] = {
                                "call_oi": 0,
                                "put_oi": float(put.get(metric, 0) or 0)
                            }
                
                for strike in sorted(strike_data.keys()):
                    data = strike_data[strike]
                    strikes.append(strike)
                    call_ois.append(data["call_oi"])
                    put_ois.append(data["put_oi"])
                    gammas.append((data["call_oi"] - data["put_oi"]) * 100)
                
                if strikes:
                    df = pd.DataFrame({
                        "Strike": strikes,
                        "Call_OI": call_ois,
                        "Put_OI": put_ois,
                        "Gamma_Exposure": gammas,
                    }).sort_values("Strike").reset_index(drop=True)
                    
                    results = analyze_0dte(df, current_price=last_price)
                    
                    snapshot = {
                        "symbol": "SPX",
                        "source": "yahoo",
                        "expiration": expiration_date.strftime("%B %d, %Y"),
                        "expiration_date": expiration_date.isoformat(),
                        "price": last_price,
                        "time": None,
                        "metric": metric,  # Add metric info to snapshot
                    }
                    
                    if isinstance(results, dict):
                        snapshot.update(results)
                    
                    _SPX_SNAPSHOT_CACHE[cache_key] = snapshot
                    _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
                    print(f"[DEBUG] Yahoo Finance SUCCESS - returning SPX snapshot with price {last_price} and metric={metric}")
                    return snapshot
        except Exception as e:
            print(f"[DEBUG] Yahoo Finance parsing failed: {e}")
            import traceback
            traceback.print_exc()
            pass  # Fall through to Nasdaq

    # Try Nasdaq as fallback
    payload = None
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
        # Final fallback to SPY
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
            last_sale_price = _parse_nasdaq_price(m.group(1))
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
            last_sale_price = _parse_nasdaq_price(m.group(1))
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

# ============================================================================
# PDF EXTRACTION FUNCTIONS (0DTE, 1DTE, Multi-DTE)
# ============================================================================

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

# ============================================================================
# GAMMA ANALYSIS CORE FUNCTIONS
# ============================================================================


def analyze_0dte(
    df: pd.DataFrame,
    current_price: float = None,
    levels_mode: str = "price",
    prefer_strike_multiple: Optional[float] = 25.0,
):
    """Analizza i dati 0DTE e restituisce risultati strutturati.

    levels_mode:
        - "price" (default): supporti/resistenze rispetto al prezzo corrente
        - "flip": supporti/resistenze rispetto alla gamma flip zone

    prefer_strike_multiple:
        Se impostato (default 25), prova a preferire strike multipli di quel valore
        quando ci sono abbastanza candidati (utile per ES). Se None, non applica
        alcuna preferenza (utile per stocks con strike a 0.5/1.0).
    """

    if df.empty:
        return {'error': 'Nessun dato 0DTE trovato'}

    requested_mode = (levels_mode or "").strip().lower()
    resolved_mode = "price" if requested_mode in {"price", "current", "current_price"} else "flip"
    if resolved_mode == "price" and current_price is None:
        resolved_mode = "flip"

    results = {
        'current_price': current_price,
        'gamma_flip': None,
        'gamma_flip_zone': None,
        'supports': [],
        'resistances': [],
        'stats': {},
        'levels_mode_requested': requested_mode or 'price',
        'levels_mode': resolved_mode,
    }

    if (requested_mode in {"price", "current", "current_price"}) and current_price is None:
        results['levels_mode_note'] = 'Modalità prezzo richiesta ma prezzo corrente mancante: uso flip zone'

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

        # Choose the threshold for supports/resistances based on selected mode.
        if resolved_mode == 'price' and current_price is not None:
            threshold = float(current_price)
            below_levels = df_sorted[df_sorted['Strike'] < threshold].copy()
            above_levels = df_sorted[df_sorted['Strike'] >= threshold].copy()
        else:
            below_levels = df_sorted[df_sorted['Strike'] < zone_low].copy()
            # include boundary in resistances (often the first call-wall is exactly on zone_high)
            above_levels = df_sorted[df_sorted['Strike'] >= zone_high].copy()

        def _pick_top_levels(df_levels: pd.DataFrame, side: str) -> pd.DataFrame:
            if df_levels.empty:
                return df_levels

            key_col = 'Put_OI' if side == 'put' else 'Call_OI'

            # Stocks: do not bias to strike multiples.
            if prefer_strike_multiple is None:
                return df_levels.nlargest(3, key_col)

            # ES: prefer strikes that are multiples of prefer_strike_multiple when available.
            m = float(prefer_strike_multiple)
            df_levels = df_levels.copy()
            strike = df_levels['Strike'].astype(float)
            # Robust multiple check for floats: consider strike a multiple if it's within epsilon.
            nearest = (strike / m).round() * m
            df_levels['is_multiple'] = (nearest - strike).abs() < 1e-6

            top = df_levels.nlargest(12, key_col)
            preferred = top[top['is_multiple']]
            if len(preferred) >= 3:
                return preferred.nlargest(3, key_col)
            remainder = top[~top['is_multiple']]
            combined = pd.concat([preferred, remainder], ignore_index=True)
            return combined.nlargest(3, key_col)

        # PUT supports below flip (largest Put OI)
        if not below_levels.empty:
            top_puts = _pick_top_levels(below_levels, side='put')
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
            results['supports_note'] = 'Nessun livello sotto il prezzo corrente' if resolved_mode == 'price' else 'Nessun livello sotto la zona di flip'

        # CALL resistances above flip (largest Call OI)
        if not above_levels.empty:
            top_calls = _pick_top_levels(above_levels, side='call')
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
            results['resistances_note'] = 'Nessun livello sopra il prezzo corrente' if resolved_mode == 'price' else 'Nessun livello sopra la zona di flip'
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

# ============================================================================
# WEB ROUTES - Authentication & Admin
# ============================================================================


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login')
def login():
    if _is_authenticated():
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/login/google')
def login_google():
    if oauth is None:
        return 'OAuth non configurato. Dipendenza mancante: Authlib.', 500

    _ensure_google_oauth_registered()
    if not hasattr(oauth, 'google'):
        missing = _google_oauth_missing_vars()
        if missing:
            return (
                'OAuth non configurato. Variabili mancanti: ' + ', '.join(missing) + '.',
                500,
            )
        return (
            'OAuth non configurato. Verifica GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET e FLASK_SECRET_KEY.',
            500,
        )

    redirect_uri = url_for('auth_callback', _external=True)
    # prompt=select_account forces account chooser if multiple accounts.
    return oauth.google.authorize_redirect(redirect_uri, prompt='select_account')


@app.route('/auth/callback')
def auth_callback():
    if oauth is None:
        return 'OAuth non configurato. Dipendenza mancante: Authlib.', 500

    _ensure_google_oauth_registered()
    if not hasattr(oauth, 'google'):
        missing = _google_oauth_missing_vars()
        if missing:
            return 'OAuth non configurato. Variabili mancanti: ' + ', '.join(missing) + '.', 500
        return 'OAuth non configurato.', 500

    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get('userinfo')
        if not userinfo:
            # Some flows provide only an id_token.
            userinfo = oauth.google.parse_id_token(token)

        session['user'] = {
            'sub': (userinfo.get('sub') if isinstance(userinfo, dict) else None),
            'email': (userinfo.get('email') if isinstance(userinfo, dict) else None),
            'name': (userinfo.get('name') if isinstance(userinfo, dict) else None),
            'picture': (userinfo.get('picture') if isinstance(userinfo, dict) else None),
        }

        _log_login_event('login', user=session.get('user'), extra={"provider": "google"})

        next_url = session.pop('next_url', None)
        if next_url and isinstance(next_url, str) and next_url.startswith('/'):
            return redirect(next_url)
        return redirect(url_for('index'))
    except Exception as e:
        return f'Errore autenticazione Google: {e}', 500


@app.route('/logout')
def logout():
    try:
        _log_login_event('logout', user=session.get('user'))
    except Exception:
        pass
    try:
        session.clear()
    except Exception:
        pass
    return redirect(url_for('login'))


@app.route('/admin')
@login_required
def admin_index():
    if not _is_admin():
        return jsonify({'error': 'Forbidden'}), 403
    return redirect(url_for('admin_login_sessions'))


@app.route('/admin/login-sessions')
@login_required
def admin_login_sessions():
    if not _is_admin():
        return jsonify({'error': 'Forbidden'}), 403

    coll = _get_mongo_login_collection()
    if coll is None:
        return (
            render_template(
                'admin.html',
                sessions=[],
                mongo_enabled=False,
                admin_emails=(os.getenv('ADMIN_EMAILS') or '').strip(),
            ),
            200,
        )

    # Default to a higher limit because a single user can generate many events
    # and push other users out of the most recent window.
    limit_raw = (request.args.get('limit') or '').strip()
    try:
        limit = int(limit_raw) if limit_raw else 500
    except Exception:
        limit = 500
    limit = max(1, min(limit, 2000))
    try:
        docs = list(coll.find({}, sort=[('created_at', -1)], limit=limit))
    except TypeError:
        # Some pymongo versions don't accept sort/limit kwargs like this
        docs = list(coll.find({}).sort('created_at', -1).limit(limit))

    sessions_out = []
    for d in docs:
        user = d.get('user') if isinstance(d, dict) else None
        if not isinstance(user, dict):
            user = {}

        created_at = d.get('created_at')
        try:
            created_at_str = created_at.isoformat() if created_at else None
        except Exception:
            created_at_str = None

        sessions_out.append({
            'event': d.get('event'),
            'created_at': created_at_str,
            'login_session_id': d.get('login_session_id'),
            'email': user.get('email'),
            'name': user.get('name'),
            'sub': user.get('sub'),
            'ip': d.get('ip'),
            'user_agent': d.get('user_agent'),
            'provider': (d.get('extra') or {}).get('provider') if isinstance(d.get('extra'), dict) else None,
        })

    return render_template(
        'admin.html',
        sessions=sessions_out,
        mongo_enabled=True,
        admin_emails=(os.getenv('ADMIN_EMAILS') or '').strip(),
        shown_limit=limit,
    )

# ============================================================================
# WEB ROUTES - API Endpoints (Market Data & MongoDB)
# ============================================================================


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
    google_oauth_configured = _ensure_google_oauth_registered()
    return jsonify({
        "status": "ok",
        "pymupdf_available": bool(_PYMUPDF_AVAILABLE),
        "app_build": _APP_BUILD,
        "python": _RUNTIME_PYTHON,
        "in_venv": bool(_IN_VENV),
        "virtual_env": os.getenv("VIRTUAL_ENV"),
        "mongo_configured": mongo is not None,
        "google_oauth_configured": google_oauth_configured,
        "google_oauth_missing": _google_oauth_missing_vars(),
        "authlib_available": OAuth is not None,
    })


@app.route('/api/last-analysis', methods=['GET'])
@login_required
def api_last_analysis():
    doc = _load_last_analysis()
    if not doc:
        return jsonify({"has_last_analysis": False})

    updated_at = doc.get('updated_at')
    try:
        updated_at_str = updated_at.isoformat() if updated_at else None
    except Exception:
        updated_at_str = None

    analysis = doc.get('analysis')
    if not isinstance(analysis, dict):
        analysis = None

    return jsonify({
        "has_last_analysis": True,
        "filename": doc.get('filename'),
        "updated_at": updated_at_str,
        "analysis": analysis,
    })


@app.route('/api/nvda-snapshot', methods=['GET'])
def nvda_snapshot():
    levels_mode = (request.args.get('levels_mode') or 'price').strip().lower()
    data = get_nvda_snapshot_cached(levels_mode=levels_mode)
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
    levels_mode = (request.args.get('levels_mode') or 'price').strip().lower()
    data = get_msft_snapshot_cached(levels_mode=levels_mode)
    if not data:
        return jsonify({"error": "Impossibile recuperare MSFT option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/spx-snapshot', methods=['GET'])
def spx_snapshot():
    # Allow force refresh for testing (add ?force=1 to URL)
    force = request.args.get('force') == '1'
    if force:
        print("[DEBUG] Force refresh SPX data requested")
        _SPX_SNAPSHOT_CACHE["fetched_at"] = 0.0  # Reset cache timestamp
    
    # Get metric parameter (volume or openInterest)
    metric = request.args.get('metric', 'volume')
    if metric not in ['volume', 'openInterest']:
        metric = 'volume'
    
    data = get_spx_snapshot_cached(metric=metric)
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


@app.route('/api/top-gamma-levels', methods=['GET'])
def top_gamma_levels():
    """Restituisce i livelli con i gamma più alti degli ultimi giorni."""
    try:
        days = int(request.args.get('days', '7'))
        limit = int(request.args.get('limit', '10'))
        days = max(1, min(days, 30))  # Limita tra 1 e 30 giorni
        limit = max(1, min(limit, 50))  # Limita tra 1 e 50 risultati
    except Exception:
        days = 7
        limit = 10
    
    levels = _get_top_gamma_levels(limit=limit, days_back=days)
    return jsonify({
        "levels": levels,
        "days": days,
        "limit": limit
    })


@app.route('/api/gamma-stats/<float:strike>', methods=['GET'])
def gamma_stats(strike):
    """Restituisce le statistiche storiche per uno strike specifico."""
    try:
        days = int(request.args.get('days', '30'))
        days = max(1, min(days, 90))  # Limita tra 1 e 90 giorni
    except Exception:
        days = 30
    
    stats = _get_gamma_statistics(strike, days_back=days)
    return jsonify({
        "strike": strike,
        "days": days,
        "stats": stats
    })

# ============================================================================
# WEB ROUTES - Main Application (PDF Analysis)
# ============================================================================


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
    
    file = request.files['file']
    original_filename = file.filename or 'upload.pdf'
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

        levels_mode = (request.form.get('levels_mode') or 'price').strip().lower()
        
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

        # Analizza. Per ES vogliamo poter mostrare sia i livelli basati su current price (CP)
        # che quelli basati su gamma flip (GF) senza dover rilanciare l'analisi.
        results = analyze_0dte(df, current_price, levels_mode=levels_mode)
        results_cp = analyze_0dte(df, current_price, levels_mode='price')
        results_gf = analyze_0dte(df, current_price, levels_mode='flip')

        if isinstance(results, dict):
            if isinstance(results_cp, dict) and not results_cp.get('error'):
                results['supports_cp'] = results_cp.get('supports') or []
                results['resistances_cp'] = results_cp.get('resistances') or []
                if results_cp.get('supports_note'):
                    results['supports_note_cp'] = results_cp.get('supports_note')
                if results_cp.get('resistances_note'):
                    results['resistances_note_cp'] = results_cp.get('resistances_note')
            else:
                results.setdefault('supports_cp', [])
                results.setdefault('resistances_cp', [])

            if isinstance(results_gf, dict) and not results_gf.get('error'):
                results['supports_gf'] = results_gf.get('supports') or []
                results['resistances_gf'] = results_gf.get('resistances') or []
                if results_gf.get('supports_note'):
                    results['supports_note_gf'] = results_gf.get('supports_note')
                if results_gf.get('resistances_note'):
                    results['resistances_note_gf'] = results_gf.get('resistances_note')
            else:
                results.setdefault('supports_gf', [])
                results.setdefault('resistances_gf', [])

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

        # Persist the last successful analysis per user (best-effort; no-op if Mongo not configured).
        try:
            if isinstance(results, dict) and not results.get('error'):
                _save_last_analysis(original_filename, results)
                # Salva anche le statistiche gamma per tracking storico
                supports = results.get('supports', [])
                resistances = results.get('resistances', [])
                _save_gamma_statistics(supports, resistances, current_price)
        except Exception:
            pass
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Errore durante l\'analisi: {str(e)}'}), 500

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================


if __name__ == '__main__':
    port_env = os.getenv('PORT')
    try:
        port = int(port_env) if port_env else 5005
    except ValueError:
        port = 5005
    app.run(debug=True, port=port)
