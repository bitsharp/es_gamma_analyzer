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
from typing import Any, Dict, List, Optional
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
import threading

try:  # Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

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


def _sanitize_next_url(next_url: Optional[str]) -> Optional[str]:
    """Return a safe in-app redirect path or None.

    Avoid redirecting users to static assets (e.g. /favicon.ico) that may have
    triggered the login flow.
    """

    if not next_url or not isinstance(next_url, str):
        return None

    # Only allow relative in-app paths.
    if not next_url.startswith('/'):
        return None

    # Strip querystring/fragments for safety and normalization.
    path_only = next_url.split('#', 1)[0].split('?', 1)[0]
    if not path_only:
        return None

    # Never redirect back to login/auth endpoints.
    if path_only.startswith(('/login', '/logout', '/auth')):
        return None

    # Never redirect to API or static asset endpoints.
    if path_only.startswith(('/api/', '/static/')):
        return None

    # Common asset extensions that should not become a post-login landing.
    lowered = path_only.lower()
    if lowered in ('/favicon.ico', '/robots.txt'):
        return None
    for ext in (
        '.ico', '.png', '.jpg', '.jpeg', '.gif', '.svg',
        '.css', '.js', '.map',
        '.woff', '.woff2', '.ttf', '.eot',
        '.txt',
    ):
        if lowered.endswith(ext):
            return None

    return path_only


@app.before_request
def _require_login():
    # Allow preflight
    if request.method == 'OPTIONS':
        return None

    path = request.path or '/'
    public_prefixes = ('/login', '/logout', '/auth')
    public_paths = {'/favicon.ico', '/robots.txt'}
    # Debug endpoints are public only in local/dev runs (never on Vercel).
    if (not os.getenv('VERCEL')) and path.startswith('/api/debug/'):
        return None

    if path == '/api/health' or path in public_paths or path.startswith(public_prefixes) or path.startswith('/static'):
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

# ==========================================================================
# MONGODB PERSISTENCE (ES←SPX OI→ES converted levels)
# ==========================================================================

_MONGO_CONVERSIONS_COLLECTION = None


def _get_mongo_conversions_collection():
    """Return Mongo collection for ES←SPX conversions, or None if not configured."""

    global _MONGO_CLIENT, _MONGO_CONVERSIONS_COLLECTION
    if _MONGO_CONVERSIONS_COLLECTION is not None:
        return _MONGO_CONVERSIONS_COLLECTION

    if MongoClient is None:
        return None

    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None

    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_CONVERSIONS_COLLECTION") or "es_spx_conversions").strip()

    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)

        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]

        # Unique per date+kind.
        try:
            coll.create_index([("date_key", 1), ("capture_kind", 1)], unique=True)
        except Exception:
            pass

        # Helpful query indexes
        try:
            coll.create_index([("date_key", -1), ("capture_rank", 1)])
        except Exception:
            pass

        # Optional TTL
        ttl_days_raw = (os.getenv("CONVERSIONS_TTL_DAYS") or "").strip()
        if ttl_days_raw:
            try:
                ttl_days = int(ttl_days_raw)
                if ttl_days > 0:
                    coll.create_index("created_at", expireAfterSeconds=60 * 60 * 24 * ttl_days)
            except Exception:
                pass

        _MONGO_CONVERSIONS_COLLECTION = coll
        return _MONGO_CONVERSIONS_COLLECTION
    except Exception:
        return None


def _capture_rank(kind: str) -> int:
    if kind == 'close':
        return 0
    if kind == '1430':
        return 1
    if kind == 'morning':
        return 2
    return 9


def _conv_mongo_upsert(doc: Dict[str, Any]) -> bool:
    coll = _get_mongo_conversions_collection()
    if coll is None:
        return False

    date_key = (doc.get('date_key') or '').strip()
    capture_kind = (doc.get('capture_kind') or '').strip()
    if not (date_key and capture_kind):
        return False

    now_dt = _dt.datetime.utcnow()

    def _as_float_or_none(v):
        try:
            return float(v)
        except Exception:
            return None

    def _as_float_list(v):
        out = []
        if not isinstance(v, list):
            return out
        for n in v:
            try:
                out.append(float(n))
            except Exception:
                continue
        return out

    payload = {
        'date_key': date_key,
        'capture_kind': capture_kind,
        'capture_rank': _capture_rank(capture_kind),
        'is_seed': bool(doc.get('is_seed')),
        'captured_at': doc.get('captured_at'),
        'based_on_date_key': doc.get('based_on_date_key'),
        'es_price': _as_float_or_none(doc.get('es_price')),
        'spx_price': _as_float_or_none(doc.get('spx_price')),
        'spread': _as_float_or_none(doc.get('spread')),
        'supports': _as_float_list(doc.get('supports') or []),
        'resistances': _as_float_list(doc.get('resistances') or []),
        'spx_supports_raw': _as_float_list(doc.get('spx_supports_raw') or []),
        'spx_resistances_raw': _as_float_list(doc.get('spx_resistances_raw') or []),
        'updated_at': now_dt,
    }

    try:
        coll.update_one(
            {'date_key': date_key, 'capture_kind': capture_kind},
            {
                '$set': payload,
                '$setOnInsert': {'created_at': now_dt},
            },
            upsert=True,
        )
        return True
    except Exception:
        return False


def _conv_mongo_get(date_key: str, capture_kind: str) -> Optional[Dict[str, Any]]:
    coll = _get_mongo_conversions_collection()
    if coll is None:
        return None

    if not (date_key and capture_kind):
        return None

    try:
        doc = coll.find_one({'date_key': date_key, 'capture_kind': capture_kind})
    except Exception:
        doc = None
    if not doc or not isinstance(doc, dict):
        return None

    def _iso(v):
        try:
            return v.isoformat() if v else None
        except Exception:
            return None

    return {
        'date_key': doc.get('date_key'),
        'capture_kind': doc.get('capture_kind'),
        'is_seed': bool(doc.get('is_seed')),
        'captured_at': doc.get('captured_at'),
        'based_on_date_key': doc.get('based_on_date_key'),
        'es_price': doc.get('es_price'),
        'spx_price': doc.get('spx_price'),
        'spread': doc.get('spread'),
        'supports': doc.get('supports') if isinstance(doc.get('supports'), list) else [],
        'resistances': doc.get('resistances') if isinstance(doc.get('resistances'), list) else [],
        'spx_supports_raw': doc.get('spx_supports_raw') if isinstance(doc.get('spx_supports_raw'), list) else [],
        'spx_resistances_raw': doc.get('spx_resistances_raw') if isinstance(doc.get('spx_resistances_raw'), list) else [],
        'created_at': _iso(doc.get('created_at')),
        'updated_at': _iso(doc.get('updated_at')),
    }


def _conv_mongo_find_latest_before(date_key: str) -> Optional[Dict[str, Any]]:
    coll = _get_mongo_conversions_collection()
    if coll is None:
        return None

    if not date_key:
        return None

    try:
        doc = coll.find(
            {
                'date_key': {'$lt': date_key},
                'capture_kind': {'$in': ['close', '1430']},
            }
        ).sort([('date_key', -1), ('capture_rank', 1)]).limit(1)
        doc = next(doc, None)
    except Exception:
        doc = None

    if not doc or not isinstance(doc, dict):
        return None

    return _conv_mongo_get(str(doc.get('date_key') or ''), str(doc.get('capture_kind') or ''))


def _conv_mongo_get_best_for_date(date_key: str) -> Optional[Dict[str, Any]]:
    """Return best available baseline doc for a specific date.

    Prefer cash close over 14:30 when both exist.
    """

    if not date_key:
        return None
    return _conv_mongo_get(date_key, 'close') or _conv_mongo_get(date_key, '1430')


def _find_previous_baseline_date_key(today_key: str, max_days_back: int = 7) -> Optional[str]:
    """Find the most recent prior date_key (preferring yesterday) that has a baseline.

    We intentionally walk backwards day-by-day so that "yesterday" is preferred when available.
    This handles weekends by naturally landing on the previous trading day.
    """

    if not today_key:
        return None
    try:
        d0 = _dt.date.fromisoformat(today_key)
    except Exception:
        return None

    for i in range(1, max(1, int(max_days_back)) + 1):
        dk = (d0 - _dt.timedelta(days=i)).isoformat()
        if _conv_mongo_get_best_for_date(dk):
            return dk
    return None


def _compute_es_spx_conversion_from_baseline(today_key: str) -> Optional[Dict[str, Any]]:
    """Compute a morning/provisional conversion using stored baseline raw strikes."""

    # Prefer yesterday's SPX options (or previous available day) when present.
    prev_key = _find_previous_baseline_date_key(today_key)
    baseline = _conv_mongo_get_best_for_date(prev_key) if prev_key else None
    if not baseline:
        baseline = _conv_mongo_find_latest_before(today_key)
    if not baseline:
        return None

    raw_s = baseline.get('spx_supports_raw') or []
    raw_r = baseline.get('spx_resistances_raw') or []
    if not (isinstance(raw_s, list) and isinstance(raw_r, list) and raw_s and raw_r):
        return None

    spx_supports_meta = baseline.get('spx_supports_meta') if isinstance(baseline.get('spx_supports_meta'), list) else None
    spx_resistances_meta = baseline.get('spx_resistances_meta') if isinstance(baseline.get('spx_resistances_meta'), list) else None

    # Prices: prefer live values, but fall back to baseline prices (useful pre-market).
    es = get_es_price_cached(max_age_seconds=60) or {}
    spx_idx = get_spx_index_price_cached(max_age_seconds=60) or {}
    spx = get_spx_snapshot_cached(metric='hybrid', max_age_seconds=60) or {}

    # If baseline meta is missing (older docs), enrich it from the current SPX hybrid snapshot
    # window strike data so the UI can show OI/Vol in tooltips.
    try:
        if (not isinstance(spx_supports_meta, list) or not spx_supports_meta) or (not isinstance(spx_resistances_meta, list) or not spx_resistances_meta):
            wl = spx.get('window_levels') if isinstance(spx, dict) else None
            if (not isinstance(wl, list) or not wl):
                # Cached snapshots from older versions may not include window_levels.
                # Force-refresh the hybrid cache and retry once.
                try:
                    _SPX_SNAPSHOT_CACHE['value_hybrid'] = None
                    _SPX_SNAPSHOT_CACHE['fetched_at_hybrid'] = 0.0
                except Exception:
                    pass
                spx2 = get_spx_snapshot_cached(metric='hybrid', max_age_seconds=60) or {}
                wl = spx2.get('window_levels') if isinstance(spx2, dict) else None
            if isinstance(wl, list) and wl:
                by_strike = {}
                for it in wl:
                    if not isinstance(it, dict):
                        continue
                    try:
                        k = float(it.get('strike'))
                    except Exception:
                        continue
                    by_strike[k] = it

                def _meta_for(arr):
                    out = []
                    for n in arr:
                        try:
                            k = float(n)
                        except Exception:
                            continue
                        m = by_strike.get(k)
                        if not isinstance(m, dict):
                            continue
                        out.append({
                            'strike': float(k),
                            'picked_by': '',
                            'call_oi': float(m.get('call_oi', 0) or 0),
                            'put_oi': float(m.get('put_oi', 0) or 0),
                            'call_vol': float(m.get('call_vol', 0) or 0),
                            'put_vol': float(m.get('put_vol', 0) or 0),
                            'total_oi': float(m.get('total_oi', 0) or 0),
                            'total_vol': float(m.get('total_vol', 0) or 0),
                        })
                    return out

                if not isinstance(spx_supports_meta, list) or not spx_supports_meta:
                    spx_supports_meta = _meta_for(raw_s)
                if not isinstance(spx_resistances_meta, list) or not spx_resistances_meta:
                    spx_resistances_meta = _meta_for(raw_r)
    except Exception:
        pass

    es_price = None
    spx_price = None
    try:
        if es.get('price') is not None:
            es_price = float(es.get('price'))
    except Exception:
        es_price = None
    try:
        if spx_idx.get('price') is not None:
            spx_price = float(spx_idx.get('price'))
        elif isinstance(spx, dict) and spx.get('price') is not None and not spx.get('error'):
            spx_price = float(spx.get('price'))
    except Exception:
        spx_price = None

    try:
        if es_price is None and baseline.get('es_price') is not None:
            es_price = float(baseline.get('es_price'))
    except Exception:
        pass
    try:
        if spx_price is None and baseline.get('spx_price') is not None:
            spx_price = float(baseline.get('spx_price'))
    except Exception:
        pass

    if not (isinstance(es_price, float) and isinstance(spx_price, float)):
        return None

    spread = es_price - spx_price

    def _convert(arr):
        out = []
        for n in arr:
            try:
                v = float(n)
                out.append(v + spread)
            except Exception:
                continue
        return out

    now_local = _dt.datetime.now()
    return {
        'date_key': today_key,
        'capture_kind': 'morning',
        'captured_at': now_local.strftime('%H:%M'),
        'based_on_date_key': str(baseline.get('date_key') or ''),
        'spread': spread,
        'es_price': es_price,
        'spx_price': spx_price,
        'supports': _convert(raw_s),
        'resistances': _convert(raw_r),
        'spx_supports_raw': raw_s,
        'spx_resistances_raw': raw_r,
        'spx_supports_meta': spx_supports_meta,
        'spx_resistances_meta': spx_resistances_meta,
    }


def _compute_es_spx_conversion_from_current_snapshot(date_key: str) -> Optional[Dict[str, Any]]:
    """Compute a best-effort conversion from the *current* SPX OI snapshot.

    This is a fallback for first-run scenarios where Mongo has no stored baseline yet.
    """

    if not date_key:
        return None

    spx = get_spx_snapshot_cached(metric='hybrid', max_age_seconds=60) or {}
    if not spx or not isinstance(spx, dict) or spx.get('error'):
        return None

    es = get_es_price_cached(max_age_seconds=60) or {}
    spx_idx = get_spx_index_price_cached(max_age_seconds=60) or {}
    es_price = es.get('price')
    spx_price = spx_idx.get('price') if spx_idx.get('price') is not None else spx.get('price')
    try:
        es_price_f = float(es_price)
        spx_price_f = float(spx_price)
    except Exception:
        return None

    spread = es_price_f - spx_price_f

    supports = spx.get('supports') if isinstance(spx.get('supports'), list) else []
    resistances = spx.get('resistances') if isinstance(spx.get('resistances'), list) else []

    raw_s = []
    raw_r = []
    meta_s = []
    meta_r = []
    for lvl in supports:
        if not isinstance(lvl, dict):
            continue
        try:
            raw_s.append(float(lvl.get('strike')))
        except Exception:
            continue
        meta_s.append({
            'strike': float(lvl.get('strike')),
            'picked_by': (lvl.get('picked_by') or ''),
            'call_oi': float(lvl.get('call_oi', 0) or 0),
            'put_oi': float(lvl.get('put_oi', 0) or 0),
            'call_vol': float(lvl.get('call_vol', 0) or 0),
            'put_vol': float(lvl.get('put_vol', 0) or 0),
            'total_oi': float(lvl.get('total_oi', 0) or 0),
            'total_vol': float(lvl.get('total_vol', 0) or 0),
        })
    for lvl in resistances:
        if not isinstance(lvl, dict):
            continue
        try:
            raw_r.append(float(lvl.get('strike')))
        except Exception:
            continue
        meta_r.append({
            'strike': float(lvl.get('strike')),
            'picked_by': (lvl.get('picked_by') or ''),
            'call_oi': float(lvl.get('call_oi', 0) or 0),
            'put_oi': float(lvl.get('put_oi', 0) or 0),
            'call_vol': float(lvl.get('call_vol', 0) or 0),
            'put_vol': float(lvl.get('put_vol', 0) or 0),
            'total_oi': float(lvl.get('total_oi', 0) or 0),
            'total_vol': float(lvl.get('total_vol', 0) or 0),
        })
    if not raw_s and not raw_r:
        return None

    now_local = _dt.datetime.now()
    return {
        'date_key': date_key,
        'capture_kind': 'morning',
        'captured_at': now_local.strftime('%H:%M'),
        'based_on_date_key': date_key,
        'spread': spread,
        'es_price': es_price_f,
        'spx_price': spx_price_f,
        'supports': [v + spread for v in raw_s],
        'resistances': [v + spread for v in raw_r],
        'spx_supports_raw': raw_s,
        'spx_resistances_raw': raw_r,
        'spx_supports_meta': meta_s if meta_s else None,
        'spx_resistances_meta': meta_r if meta_r else None,
    }


def _maybe_capture_es_spx_conversion(snapshot: Optional[Dict[str, Any]], now_dt: Optional[_dt.datetime] = None) -> None:
    """Best-effort: store 14:30 and cash-close conversions into MongoDB."""
    if not snapshot or not isinstance(snapshot, dict) or snapshot.get('error'):
        return

    now_dt = now_dt or _dt.datetime.now()
    h, m = now_dt.hour, now_dt.minute

    capture_kind = None
    # Local-time windows (match UI expectations).
    if h == 14 and 30 <= m < 35:
        capture_kind = '1430'
    elif h == 16 and m < 5:
        capture_kind = 'close'
    else:
        return

    today_key = now_dt.date().isoformat()

    es = get_es_price_cached(max_age_seconds=60) or {}
    spx_idx = get_spx_index_price_cached(max_age_seconds=60) or {}
    es_price = es.get('price')
    spx_price = spx_idx.get('price') if spx_idx.get('price') is not None else snapshot.get('price')
    try:
        es_price_f = float(es_price)
        spx_price_f = float(spx_price)
    except Exception:
        return

    spread = es_price_f - spx_price_f

    supports = snapshot.get('supports') if isinstance(snapshot.get('supports'), list) else []
    resistances = snapshot.get('resistances') if isinstance(snapshot.get('resistances'), list) else []

    raw_s = []
    raw_r = []
    meta_s = []
    meta_r = []
    for lvl in supports:
        if not isinstance(lvl, dict):
            continue
        try:
            raw_s.append(float(lvl.get('strike')))
        except Exception:
            continue
        meta_s.append({
            'strike': float(lvl.get('strike')),
            'picked_by': (lvl.get('picked_by') or ''),
            'call_oi': float(lvl.get('call_oi', 0) or 0),
            'put_oi': float(lvl.get('put_oi', 0) or 0),
            'call_vol': float(lvl.get('call_vol', 0) or 0),
            'put_vol': float(lvl.get('put_vol', 0) or 0),
            'total_oi': float(lvl.get('total_oi', 0) or 0),
            'total_vol': float(lvl.get('total_vol', 0) or 0),
        })
    for lvl in resistances:
        if not isinstance(lvl, dict):
            continue
        try:
            raw_r.append(float(lvl.get('strike')))
        except Exception:
            continue
        meta_r.append({
            'strike': float(lvl.get('strike')),
            'picked_by': (lvl.get('picked_by') or ''),
            'call_oi': float(lvl.get('call_oi', 0) or 0),
            'put_oi': float(lvl.get('put_oi', 0) or 0),
            'call_vol': float(lvl.get('call_vol', 0) or 0),
            'put_vol': float(lvl.get('put_vol', 0) or 0),
            'total_oi': float(lvl.get('total_oi', 0) or 0),
            'total_vol': float(lvl.get('total_vol', 0) or 0),
        })
    if not raw_s and not raw_r:
        return

    converted_s = [v + spread for v in raw_s]
    converted_r = [v + spread for v in raw_r]

    doc = {
        'date_key': today_key,
        'capture_kind': capture_kind,
        'is_seed': False,
        'captured_at': now_dt.strftime('%H:%M'),
        'spread': spread,
        'es_price': es_price_f,
        'spx_price': spx_price_f,
        'supports': converted_s,
        'resistances': converted_r,
        'spx_supports_raw': raw_s,
        'spx_resistances_raw': raw_r,
        'spx_supports_meta': meta_s if meta_s else None,
        'spx_resistances_meta': meta_r if meta_r else None,
    }
    _conv_mongo_upsert(doc)


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
    "last_success_at": 0.0,
}


def _seed_es_price_manual(price: float, note: str = "manual") -> None:
    """Seed the ES price cache from a user-provided value.

    This is used as a fallback when external price providers are rate-limited.
    """

    try:
        p = float(price)
    except Exception:
        return

    now = time.time()
    _ES_PRICE_CACHE["value"] = {
        "symbol": "ES",
        "price": p,
        "date": "",
        "time": "",
        "source": "manual",
        "instrument": "ES Futures",
        "note": f"Manual ES price ({note})",
        "stale": True,
    }
    _ES_PRICE_CACHE["fetched_at"] = now
    _ES_PRICE_CACHE["last_success_at"] = now


_SPX_INDEX_PRICE_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


def _seed_spx_price_manual(price: float, note: str = "manual") -> Optional[Dict[str, Any]]:
    try:
        p = float(price)
    except Exception:
        return None

    now = time.time()
    data = {
        "symbol": "^GSPC",
        "price": p,
        "date": "",
        "time": "",
        "source": "manual",
        "instrument": "SPX Index",
        "note": f"Manual SPX price ({note})",
    }
    _SPX_INDEX_PRICE_CACHE["value"] = data
    _SPX_INDEX_PRICE_CACHE["fetched_at"] = now
    return data


_ES_SPX_SPREAD_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_ES_SPX_OVERNIGHT_BASIS_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_COT_SP500_CACHE = {
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


_SPX_0DTE_VOLUME_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


def _run_with_timeout(fn, timeout_seconds: float):
    """Run a function in a thread with a hard timeout.

    Returns the function's return value, or raises TimeoutError.
    """

    result_container: Dict[str, Any] = {}
    error_container: Dict[str, Any] = {}

    def _target():
        try:
            result_container["value"] = fn()
        except Exception as e:
            error_container["error"] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout_seconds)

    if t.is_alive():
        raise TimeoutError(f"Operation timed out after {timeout_seconds:.1f}s")
    if "error" in error_container:
        raise error_container["error"]
    return result_container.get("value")


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

    path = f"/q/l/?s={urllib.parse.quote(symbol)}&f=sd2t2ohlcv&h&e=csv"
    urls = [f"https://stooq.com{path}", f"http://stooq.com{path}"]

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    raw = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                raw = response.read().decode("utf-8", errors="replace")
            if raw:
                break
        except Exception:
            continue

    if not raw:
        return None

    # Stooq may return a plain-text error (e.g. "Exceeded the daily hits limit").
    # Treat it as unavailable so callers can fall back to other sources.
    if "Exceeded the daily hits limit" in raw:
        return None

    try:

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


def _fetch_stooq_previous_daily_close(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch the previous completed *daily* close for a symbol from Stooq.

    We prefer the *previous* daily row to avoid using Stooq's intraday-updating quote.
    This makes it suitable as an "overnight close basis".

    Returns a dict with keys: symbol, price, date, source.
    """

    path = f"/q/d/l/?s={urllib.parse.quote(symbol)}&i=d"
    urls = [f"https://stooq.com{path}", f"http://stooq.com{path}"]

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "text/csv,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    raw = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as response:
                raw = response.read().decode("utf-8", errors="replace")
            if raw:
                break
        except Exception:
            continue

    if not raw:
        return None

    if "Exceeded the daily hits limit" in raw:
        return None

    try:
        reader = csv.DictReader(io.StringIO(raw))
        rows = [r for r in reader if isinstance(r, dict)]
        if not rows:
            return None

        # Stooq may include today's partial/intraday-updating row.
        # Use the previous completed row when possible.
        row = rows[-2] if len(rows) >= 2 else rows[-1]
        close_val = (row.get("Close") or "").strip()
        if not close_val or close_val.upper() in {"N/D", "NA", "NULL"}:
            return None

        date_s = (row.get("Date") or "").strip()
        return {
            "symbol": (row.get("Symbol") or symbol).strip(),
            "price": float(close_val),
            "date": date_s,
            "source": "stooq_daily_prev_close",
        }
    except Exception:
        return None


def _fetch_yahoo_quote_price(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch last price for a symbol from Yahoo's public quote endpoint.

    Uses urllib (no requests/yfinance) to avoid SSL/urllib3 issues.

    Returns a dict with keys: symbol, price, date, time, source.
    """

    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={urllib.parse.quote(symbol)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)

        result = (((payload.get("quoteResponse") or {}).get("result") or [])[:1] or [None])[0]
        if not isinstance(result, dict):
            return None

        price = result.get("regularMarketPrice")
        if price is None:
            return None

        ts = result.get("regularMarketTime")
        date_s = ""
        time_s = ""
        try:
            if ts:
                dt = _dt.datetime.fromtimestamp(int(ts))
                date_s = dt.strftime("%Y-%m-%d")
                time_s = dt.strftime("%H:%M:%S")
        except Exception:
            pass

        return {
            "symbol": (result.get("symbol") or symbol).strip(),
            "price": float(price),
            "date": date_s,
            "time": time_s,
            "source": "yahoo_quote",
        }
    except Exception:
        return None


def _fetch_yahoo_quote_snapshot(symbols: List[str]) -> Optional[Dict[str, Any]]:
    """Fetch selected fields for multiple symbols from Yahoo's public quote endpoint.

    Returns: { "source": "yahoo_quote", "quotes": {<symbol>: {...}} }
    Each quote includes: regularMarketPrice, regularMarketPreviousClose, regularMarketTime, marketState.
    """

    syms = [s for s in (symbols or []) if isinstance(s, str) and s.strip()]
    if not syms:
        return None

    url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={urllib.parse.quote(','.join(syms))}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            raw = response.read().decode("utf-8", errors="replace")
        payload = json.loads(raw)

        results = ((payload.get("quoteResponse") or {}).get("result") or [])
        if not isinstance(results, list):
            return None

        out: Dict[str, Any] = {"source": "yahoo_quote", "quotes": {}}
        for r in results:
            if not isinstance(r, dict):
                continue
            sym = (r.get("symbol") or "").strip()
            if not sym:
                continue

            q = {
                "symbol": sym,
                "regularMarketPrice": r.get("regularMarketPrice"),
                "regularMarketPreviousClose": r.get("regularMarketPreviousClose"),
                "regularMarketTime": r.get("regularMarketTime"),
                "marketState": r.get("marketState"),
            }
            out["quotes"][sym] = q

        return out
    except Exception:
        return None


def get_es_spx_overnight_basis_cached(max_age_seconds: int = 10 * 60) -> Optional[Dict[str, Any]]:
    """Return stable ES/SPX basis prices for after-hours.

    Uses Yahoo quote fields and prefers `regularMarketPreviousClose` for both legs,
    falling back to `regularMarketPrice` if missing.

    This is intended for freezing SPX OI→ES converted levels overnight.
    """

    now = time.time()
    cached = _ES_SPX_OVERNIGHT_BASIS_CACHE.get("value")
    fetched_at = float(_ES_SPX_OVERNIGHT_BASIS_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    # Preferred: Yahoo quote endpoint (but can fail on some macOS LibreSSL builds).
    snap = _fetch_yahoo_quote_snapshot(["ES=F", "^GSPC"])
    if isinstance(snap, dict) and isinstance(snap.get("quotes"), dict):
        quotes = snap.get("quotes")
        esq = quotes.get("ES=F") if isinstance(quotes.get("ES=F"), dict) else {}
        spxq = quotes.get("^GSPC") if isinstance(quotes.get("^GSPC"), dict) else {}

        def _num(v: Any) -> Optional[float]:
            try:
                if v is None:
                    return None
                return float(v)
            except Exception:
                return None

        es_close = _num(esq.get("regularMarketPreviousClose"))
        if es_close is None:
            es_close = _num(esq.get("regularMarketPrice"))

        spx_close = _num(spxq.get("regularMarketPreviousClose"))
        if spx_close is None:
            spx_close = _num(spxq.get("regularMarketPrice"))

        if es_close is not None and spx_close is not None:
            payload = {
                "es_close": es_close,
                "spx_close": spx_close,
                "spread_close": (es_close - spx_close),
                "asof": _dt.datetime.now().isoformat(timespec="seconds"),
                "source": "yahoo_quote",
                "raw": {
                    "es": esq,
                    "spx": spxq,
                },
            }
            _ES_SPX_OVERNIGHT_BASIS_CACHE["value"] = payload
            _ES_SPX_OVERNIGHT_BASIS_CACHE["fetched_at"] = now
            return payload

    # Fallback: Stooq daily previous close for both legs (stable intraday).
    spx_d = _fetch_stooq_previous_daily_close("^spx")
    es_d = _fetch_stooq_previous_daily_close("es.f")
    if not spx_d or not es_d:
        return None

    try:
        spx_close = float(spx_d.get("price"))
        es_close = float(es_d.get("price"))
    except Exception:
        return None

    payload = {
        "es_close": es_close,
        "spx_close": spx_close,
        "spread_close": (es_close - spx_close),
        "asof": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": "stooq_daily_prev_close",
        "raw": {
            "es": es_d,
            "spx": spx_d,
        },
    }

    _ES_SPX_OVERNIGHT_BASIS_CACHE["value"] = payload
    _ES_SPX_OVERNIGHT_BASIS_CACHE["fetched_at"] = now
    return payload


def get_cot_sp500_cached(max_age_seconds: int = 60 * 60) -> Optional[Dict[str, Any]]:
    """Fetch COT (Commitment of Traders) data for S&P 500 with caching.

    Source: CFTC Legacy Futures Only Report, exposed by an external service
    at http://178.104.133.41:8080/cot/sp500. The report is published weekly,
    so a 1h cache is generous.
    """

    now = time.time()
    cached = _COT_SP500_CACHE.get("value")
    fetched_at = float(_COT_SP500_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    url = "http://178.104.133.41:8080/cot/sp500"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ESGammaAnalyzer/1.0)",
        "Accept": "application/json",
    }
    # Keep well under Vercel's serverless function timeout (10s on hobby tier)
    # so the urllib exception surfaces as JSON instead of Vercel returning its
    # HTML 504 page, which would break res.json() on the client.
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as response:
            ctype = (response.headers.get("Content-Type") or "").lower()
            raw = response.read().decode("utf-8", errors="replace")
    except Exception as e:
        # On failure, return last cached value if any (stale-while-error).
        if cached:
            return cached
        return {"error": f"COT fetch error: {e}"}

    if "json" not in ctype and not raw.lstrip().startswith(("{", "[")):
        if cached:
            return cached
        return {"error": "COT upstream returned non-JSON response"}

    try:
        data = json.loads(raw)
    except Exception as e:
        if cached:
            return cached
        return {"error": f"COT parse error: {e}"}

    if not isinstance(data, dict):
        return {"error": "Invalid COT response"}

    _COT_SP500_CACHE["value"] = data
    _COT_SP500_CACHE["fetched_at"] = now
    return data


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
    """Fetch options chain data from Yahoo Finance.

    Preferred path: `yfinance` (convenient).
    Fallback path: direct JSON endpoint used by the Yahoo options page.

    We intentionally do NOT scrape the HTML page (it's JS-heavy and fragile).
    """

    def _fetch_yahoo_options_http(sym: str) -> Optional[Dict[str, Any]]:
        # The options page (e.g. https://finance.yahoo.com/quote/%5ESPX/options/?straddle=true)
        # is backed by this JSON endpoint.
        try:
            encoded = urllib.parse.quote(sym)
            base_url = f"https://query2.finance.yahoo.com/v7/finance/options/{encoded}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
            }

            def _get_json(url: str) -> Optional[Dict[str, Any]]:
                try:
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")
                    return json.loads(raw)
                except Exception:
                    return None

            first = _get_json(base_url)
            if not first:
                return None

            oc = (((first.get("optionChain") or {}).get("result") or [])[:1] or [None])[0]
            if not isinstance(oc, dict):
                return None

            quote = oc.get("quote") or {}
            price = quote.get("regularMarketPrice") or quote.get("postMarketPrice") or quote.get("preMarketPrice")
            try:
                price_f = float(price) if price is not None else None
            except Exception:
                price_f = None

            expirations = oc.get("expirationDates") or []
            if not expirations:
                return None

            # Yahoo returns Unix timestamps (seconds). Pick the nearest.
            try:
                exp_ts = int(expirations[0])
            except Exception:
                return None

            chain = _get_json(f"{base_url}?date={exp_ts}")
            if not chain:
                return None

            oc2 = (((chain.get("optionChain") or {}).get("result") or [])[:1] or [None])[0]
            if not isinstance(oc2, dict):
                return None

            opt_list = oc2.get("options") or []
            if not opt_list or not isinstance(opt_list[0], dict):
                return None

            calls = opt_list[0].get("calls") or []
            puts = opt_list[0].get("puts") or []

            # Convert expiry to YYYY-MM-DD (to match the yfinance path expectation).
            try:
                exp_date = _dt.datetime.utcfromtimestamp(exp_ts).date().isoformat()
            except Exception:
                exp_date = None

            return {
                "symbol": sym,
                "price": price_f,
                "expiration": exp_date,
                "calls": calls,
                "puts": puts,
                "source": "yahoo_http",
            }
        except Exception:
            return None

    # 1) Try yfinance if available
    if yf:
        print(f"[DEBUG] Fetching Yahoo Finance options for {symbol} using yfinance")
        try:
            ticker = yf.Ticker(symbol)

            info = ticker.info
            current_price = info.get("regularMarketPrice") or info.get("currentPrice")

            expirations = ticker.options
            if not expirations:
                print(f"[DEBUG] No expirations found for {symbol} via yfinance")
            else:
                nearest_exp = expirations[0]
                print(f"[DEBUG] Using expiration: {nearest_exp}")

                opt_chain = ticker.option_chain(nearest_exp)
                calls_df = opt_chain.calls
                puts_df = opt_chain.puts

                if current_price:
                    all_strikes = sorted(set(calls_df['strike'].tolist() + puts_df['strike'].tolist()))
                    strikes_below = [s for s in all_strikes if s < current_price][-15:]
                    strikes_above = [s for s in all_strikes if s >= current_price][:15]
                    relevant_strikes = set(strikes_below + strikes_above)
                    calls_df = calls_df[calls_df['strike'].isin(relevant_strikes)]
                    puts_df = puts_df[puts_df['strike'].isin(relevant_strikes)]
                    print(f"[DEBUG] Filtered to {len(relevant_strikes)} strikes around price {current_price}")

                print(f"[DEBUG] Yahoo Finance fetch SUCCESS - {len(calls_df)} calls, {len(puts_df)} puts")

                return {
                    "symbol": symbol,
                    "price": current_price,
                    "expiration": nearest_exp,
                    "calls": calls_df.to_dict('records'),
                    "puts": puts_df.to_dict('records'),
                    "source": "yahoo_yfinance",
                }
        except Exception as e:
            print(f"[DEBUG] Yahoo Finance yfinance fetch FAILED: {e}")

    # 2) Fallback: use Yahoo's JSON endpoint directly
    print(f"[DEBUG] Fetching Yahoo Finance options for {symbol} using HTTP JSON endpoint")
    data = _fetch_yahoo_options_http(symbol)
    if data:
        try:
            calls = data.get("calls") or []
            puts = data.get("puts") or []
            print(f"[DEBUG] Yahoo HTTP options SUCCESS - {len(calls)} calls, {len(puts)} puts")
        except Exception:
            pass
        return data

    print(f"[DEBUG] Yahoo HTTP options FAILED for {symbol}")
    return None


def get_spx_0dte_volume_levels_cached(max_age_seconds: int = 5 * 60) -> Dict[str, Any]:
    """Fetch SPX 0DTE levels from Yahoo options, using only the Volume column.

    Source is equivalent to the data shown on:
      https://finance.yahoo.com/quote/%5ESPX/options/?straddle=true

    Notes:
      - Yahoo can rate-limit (HTTP 429). This function is cached to reduce hits.
      - If today's expiration is not available, returns None.
    """

    def _cache_set(payload: Dict[str, Any]) -> Dict[str, Any]:
        _SPX_0DTE_VOLUME_CACHE["value"] = payload
        _SPX_0DTE_VOLUME_CACHE["fetched_at"] = time.time()
        return payload

    now_ts = time.time()
    cached = _SPX_0DTE_VOLUME_CACHE.get("value")
    fetched_at = float(_SPX_0DTE_VOLUME_CACHE.get("fetched_at") or 0.0)
    if isinstance(cached, dict) and (now_ts - fetched_at) <= max_age_seconds:
        return cached

    base: Dict[str, Any] = {
        "symbol": "SPX",
        "source": "yahoo",
        "metric": "volume",
        "time": None,
        "note": "Yahoo ^SPX options 0DTE; levels based on Volume",
    }

    if not yf:
        return _cache_set({**base, "error": "yfinance non disponibile"})

    today_str = _dt.date.today().isoformat()

    def _fetch() -> Dict[str, Any]:
        ticker = yf.Ticker("^SPX")

        # Avoid hard dependency on .info (can be slow / rate-limited).
        current_price = None
        try:
            fi = getattr(ticker, "fast_info", None)
            if isinstance(fi, dict):
                current_price = fi.get("last_price") or fi.get("lastPrice")
            else:
                current_price = getattr(fi, "last_price", None) or getattr(fi, "lastPrice", None)
        except Exception:
            current_price = None

        if current_price is None:
            info = ticker.info or {}
            current_price = info.get("regularMarketPrice") or info.get("currentPrice")

        expirations = ticker.options or []
        if not expirations:
            return {**base, "error": "Nessuna scadenza SPX disponibile da Yahoo", "today": today_str}

        if today_str not in expirations:
            # Enforce 0DTE requirement: if no expiry today, do not fall back.
            return {
                **base,
                "error": "Nessuna scadenza 0DTE oggi su Yahoo",
                "today": today_str,
                "available_expirations": expirations[:8],
            }

        opt_chain = ticker.option_chain(today_str)
        calls_df = opt_chain.calls
        puts_df = opt_chain.puts
        if calls_df is None or puts_df is None or calls_df.empty or puts_df.empty:
            return {**base, "error": "Option chain SPX 0DTE vuota su Yahoo", "today": today_str}

        # Build analyzer DF: map Volume -> Call_OI/Put_OI columns.
        strike_data: Dict[float, Dict[str, float]] = {}

        for _, row in calls_df.iterrows():
            try:
                strike = float(row.get("strike"))
            except Exception:
                continue
            if strike <= 0:
                continue
            try:
                vol = float(row.get("volume") or 0)
            except Exception:
                vol = 0.0
            strike_data.setdefault(strike, {"call": 0.0, "put": 0.0})["call"] = vol

        for _, row in puts_df.iterrows():
            try:
                strike = float(row.get("strike"))
            except Exception:
                continue
            if strike <= 0:
                continue
            try:
                vol = float(row.get("volume") or 0)
            except Exception:
                vol = 0.0
            strike_data.setdefault(strike, {"call": 0.0, "put": 0.0})["put"] = vol

        if not strike_data:
            return {**base, "error": "Nessun dato volume utilizzabile su Yahoo", "today": today_str}

        strikes = sorted(strike_data.keys())
        calls = [float(strike_data[s]["call"]) for s in strikes]
        puts = [float(strike_data[s]["put"]) for s in strikes]
        gammas = [(c - p) * 100.0 for c, p in zip(calls, puts)]

        df = pd.DataFrame(
            {
                "Strike": strikes,
                "Call_OI": calls,
                "Put_OI": puts,
                "Gamma_Exposure": gammas,
            }
        ).sort_values("Strike").reset_index(drop=True)

        results = analyze_0dte(
            df,
            current_price=float(current_price) if current_price is not None else None,
            levels_mode="price",
            prefer_strike_multiple=None,
        )

        snapshot: Dict[str, Any] = {
            **base,
            "expiration": today_str,
            "expiration_date": today_str,
            "price": float(current_price) if current_price is not None else None,
        }
        if isinstance(results, dict):
            snapshot.update(results)
        return snapshot

    try:
        # Keep the endpoint responsive even if Yahoo is slow.
        payload = _run_with_timeout(_fetch, timeout_seconds=12.0)
        return _cache_set(payload if isinstance(payload, dict) else {**base, "error": "Risposta Yahoo non valida"})
    except TimeoutError as e:
        # Cache timeouts briefly to avoid repeated long hangs.
        return _cache_set({**base, "error": f"Yahoo timeout: {e}", "today": today_str})
    except Exception as e:
        msg = str(e) or e.__class__.__name__
        return _cache_set({**base, "error": f"Yahoo errore: {msg}", "today": today_str})


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
    levels_mode: str = "price",
) -> Optional[Dict[str, Any]]:
    """Generic Nasdaq option-chain snapshot for a US stock symbol."""

    now_ts = time.time()
    requested = (levels_mode or "price").strip().lower()
    mode_key = "flip" if requested in {"flip", "gamma", "gamma_flip", "flip_zone"} else "price"

    fetched_at = float(cache.get("fetched_at") or 0.0)
    if (now_ts - fetched_at) <= max_age_seconds:
        by_mode = cache.get("value_by_mode")
        if isinstance(by_mode, dict) and by_mode.get(mode_key):
            return by_mode.get(mode_key)

        cached = cache.get("value")
        if isinstance(cached, dict) and (cached.get("levels_mode") == mode_key or cached.get("levels_mode_requested") == mode_key):
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

    strike_data: Dict[float, Dict[str, float]] = {}

    for row in rows:
        if (row.get("expiryDate") or "").strip() != nearest_exp_label:
            continue

        strike = _parse_pdf_number(row.get("strike"))
        if strike <= 0:
            continue

        call_oi = _parse_pdf_number(row.get("c_Openinterest"))
        put_oi = _parse_pdf_number(row.get("p_Openinterest"))
        call_vol = _parse_pdf_number(
            row.get("c_Volume")
            or row.get("c_volume")
            or row.get("c_Vol")
            or row.get("c_vol")
            or 0
        )
        put_vol = _parse_pdf_number(
            row.get("p_Volume")
            or row.get("p_volume")
            or row.get("p_Vol")
            or row.get("p_vol")
            or 0
        )
        gamma_exposure = (call_oi - put_oi) * 1000

        strike_data[float(strike)] = {
            "call_oi": float(call_oi),
            "put_oi": float(put_oi),
            "call_vol": float(call_vol),
            "put_vol": float(put_vol),
        }

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
        "symbol": sym,
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
    }

    # Precompute both variants so the frontend can show CP+GF together.
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

    cache["value_by_mode"] = by_mode
    cache["value"] = by_mode.get(mode_key) or by_mode.get("price")
    cache["fetched_at"] = now_ts
    return cache["value"]


def get_aapl_snapshot_cached(max_age_seconds: int = 60, levels_mode: str = "price") -> Optional[Dict[str, Any]]:
    return _get_nasdaq_stock_snapshot_cached("AAPL", _AAPL_SNAPSHOT_CACHE, max_age_seconds=max_age_seconds, levels_mode=levels_mode)


def get_goog_snapshot_cached(max_age_seconds: int = 60, levels_mode: str = "price") -> Optional[Dict[str, Any]]:
    return _get_nasdaq_stock_snapshot_cached("GOOG", _GOOG_SNAPSHOT_CACHE, max_age_seconds=max_age_seconds, levels_mode=levels_mode)


def get_amzn_snapshot_cached(max_age_seconds: int = 60, levels_mode: str = "price") -> Optional[Dict[str, Any]]:
    return _get_nasdaq_stock_snapshot_cached("AMZN", _AMZN_SNAPSHOT_CACHE, max_age_seconds=max_age_seconds, levels_mode=levels_mode)

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


def _compute_spx_hybrid_levels(
    strike_data: Dict[float, Dict[str, float]],
    current_price: Optional[float],
    window_each_side: int = 15,
) -> Dict[str, Any]:
    """Select SPX levels using BOTH open interest and volume.

    - Look at the nearest `window_each_side` strikes below and above `current_price`.
    - For each side, pick:
        1) strike with max total OI (call_oi + put_oi)
        2) strike with max total volume (call_vol + put_vol)

    Returns a dict with `supports` and `resistances` lists.
    """

    try:
        px = float(current_price) if current_price is not None else None
    except Exception:
        px = None

    strikes = sorted([float(s) for s in (strike_data or {}).keys() if s is not None])
    if not strikes or px is None:
        return {"supports": [], "resistances": [], "window_each_side": int(window_each_side)}

    below = [s for s in strikes if s < px][-int(window_each_side):]
    above = [s for s in strikes if s >= px][:int(window_each_side)]

    def _total(entry: Dict[str, float], kind: str) -> float:
        try:
            if kind == "oi":
                return float(entry.get("call_oi", 0.0) or 0.0) + float(entry.get("put_oi", 0.0) or 0.0)
            if kind == "vol":
                return float(entry.get("call_vol", 0.0) or 0.0) + float(entry.get("put_vol", 0.0) or 0.0)
        except Exception:
            return 0.0
        return 0.0

    def _fmt(strike: float, picked_by: str) -> Dict[str, Any]:
        d = strike_data.get(strike) or {}
        call_oi = float(d.get("call_oi", 0.0) or 0.0)
        put_oi = float(d.get("put_oi", 0.0) or 0.0)
        call_vol = float(d.get("call_vol", 0.0) or 0.0)
        put_vol = float(d.get("put_vol", 0.0) or 0.0)
        return {
            "strike": float(strike),
            "call_oi": call_oi,
            "put_oi": put_oi,
            "call_vol": call_vol,
            "put_vol": put_vol,
            "total_oi": call_oi + put_oi,
            "total_vol": call_vol + put_vol,
            "picked_by": picked_by,
        }

    def _pick(side_strikes: list[float]) -> list[Dict[str, Any]]:
        if not side_strikes:
            return []
        best_oi = max(side_strikes, key=lambda s: _total(strike_data.get(s, {}), "oi"))
        best_vol = max(side_strikes, key=lambda s: _total(strike_data.get(s, {}), "vol"))
        out: list[Dict[str, Any]] = []
        out.append(_fmt(best_oi, "max_oi"))
        if best_vol != best_oi:
            out.append(_fmt(best_vol, "max_vol"))
        return out

    supports = _pick(below)
    resistances = _pick(above)

    supports = sorted(supports, key=lambda x: float(x.get("strike") or 0.0), reverse=True)
    resistances = sorted(resistances, key=lambda x: float(x.get("strike") or 0.0))

    return {
        "supports": supports,
        "resistances": resistances,
        "window_each_side": int(window_each_side),
    }


def _build_spx_window_levels(
    strike_data: Dict[float, Dict[str, float]],
    current_price: Optional[float],
    window_each_side: int = 15,
) -> list[Dict[str, Any]]:
    """Return per-strike OI/Vol meta for the ±window around price.

    This is used to enrich UI tooltips and baseline conversions for strikes that are not
    necessarily the single max-OI/max-Vol picks.
    """

    strikes = sorted([float(s) for s in (strike_data or {}).keys() if s is not None])
    if not strikes:
        return []
    px = float(current_price) if isinstance(current_price, (int, float)) else None
    if px is None:
        return []

    below = [s for s in strikes if s < px][-int(window_each_side):]
    above = [s for s in strikes if s >= px][:int(window_each_side)]
    window = below + above

    out: list[Dict[str, Any]] = []
    for s in window:
        d = strike_data.get(s) or {}
        call_oi = float(d.get("call_oi", 0.0) or 0.0)
        put_oi = float(d.get("put_oi", 0.0) or 0.0)
        call_vol = float(d.get("call_vol", 0.0) or 0.0)
        put_vol = float(d.get("put_vol", 0.0) or 0.0)
        out.append({
            "strike": float(s),
            "call_oi": call_oi,
            "put_oi": put_oi,
            "call_vol": call_vol,
            "put_vol": put_vol,
            "total_oi": call_oi + put_oi,
            "total_vol": call_vol + put_vol,
        })
    return out


def get_spx_snapshot_cached(metric: str = 'volume', max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch SPX last price + option-chain derived gamma flip for the nearest expiry.

    Yahoo Finance data is fetched only at 8:00 AM and 2:30 PM ET to avoid rate limits.
    Between these times, cached data is served.
    
    Args:
        metric: 'volume' | 'openInterest' | 'hybrid'
            - volume/openInterest: legacy single-metric behavior
            - hybrid: pick max OI and max Volume within ±15 strikes around price
        max_age_seconds: maximum age of cached data in seconds
    """

    now_ts = time.time()

    # Use NY time for the scheduled fetch windows (server may not run in ET).
    now_dt_local = _dt.datetime.now()
    now_dt_ny = now_dt_local
    if ZoneInfo is not None:
        try:
            now_dt_ny = _dt.datetime.now(tz=ZoneInfo("America/New_York"))
        except Exception:
            now_dt_ny = now_dt_local

    # Force refresh path: when callers pass max_age_seconds <= 0 (e.g. /api/spx-snapshot?force=1),
    # allow a Yahoo fetch attempt even outside the scheduled windows.
    force_refresh = False
    try:
        force_refresh = int(max_age_seconds) <= 0
    except Exception:
        force_refresh = False

    # Check if we should fetch Yahoo data (only at 8:00 AM or 2:30 PM ET).
    should_fetch_yahoo = False
    current_hour = now_dt_ny.hour
    current_minute = now_dt_ny.minute
    # 8:00 AM window (8:00-8:05)
    if current_hour == 8 and current_minute < 5:
        should_fetch_yahoo = True
    # 2:30 PM window (14:30-14:35)
    elif current_hour == 14 and 30 <= current_minute < 35:
        should_fetch_yahoo = True
    if force_refresh:
        should_fetch_yahoo = True
    
    metric_norm = (metric or "volume").strip()
    if metric_norm not in {"volume", "openInterest", "hybrid"}:
        metric_norm = "volume"

    # Use metric-specific cache key
    cache_key = f"value_{metric_norm}"
    fetched_at_key = f"fetched_at_{metric_norm}"

    last_good_key = f"last_good_{metric_norm}"
    last_good_fetched_at_key = f"last_good_fetched_at_{metric_norm}"
    
    cached = _SPX_SNAPSHOT_CACHE.get(cache_key)
    fetched_at = float(_SPX_SNAPSHOT_CACHE.get(fetched_at_key) or 0.0)

    def _is_proxy_snapshot(s: Any) -> bool:
        if not isinstance(s, dict):
            return False
        note = (s.get("note") or "")
        try:
            return "proxy" in str(note).lower()
        except Exception:
            return False

    cached_age = now_ts - fetched_at if fetched_at else None
    cached_is_proxy = _is_proxy_snapshot(cached)
    
    # If we're in a Yahoo fetch window and haven't fetched recently (within 5 minutes), continue.
    if should_fetch_yahoo and (now_ts - fetched_at) > 300:
        print(f"[DEBUG] SPX scheduled fetch time (NY): {current_hour}:{current_minute:02d} with metric={metric}")
        pass
    # Otherwise, return cached data when it's not proxy.
    # If the cached snapshot is a proxy (SPY), allow refresh attempts once it's older than max_age_seconds.
    elif cached and not force_refresh:
        if not cached_is_proxy:
            # Preserve previous behavior: serve cached between scheduled windows.
            print(f"[DEBUG] Using cached SPX data with metric={metric} (fetched {int((now_ts - fetched_at)/60)} minutes ago)")
            return cached

        # Cached is proxy: only keep it briefly to avoid hammering providers.
        try:
            max_age = int(max_age_seconds)
        except Exception:
            max_age = 60
        if max_age > 0 and cached_age is not None and cached_age <= max_age:
            print(f"[DEBUG] Using cached SPX PROXY data with metric={metric} (age {int(cached_age)}s)")
            return cached

    # Try Yahoo Finance only during scheduled fetch windows.
    # Outside these windows we prefer Nasdaq to avoid rate-limits and slow/hanging requests.
    if should_fetch_yahoo:
        yahoo_data = _fetch_yahoo_options("^SPX")
        print(f"[DEBUG] Yahoo Finance data received: {yahoo_data is not None}")
        if yahoo_data:
            print(f"[DEBUG] Yahoo data keys: {yahoo_data.keys()}")
            try:
                calls = yahoo_data.get("calls", [])
                puts = yahoo_data.get("puts", [])
                last_price = yahoo_data.get("price")
                expiration_str = yahoo_data.get("expiration")

                # Price is sometimes missing on Yahoo options responses.
                # Fill it from Stooq ^spx (delayed/indicative) to avoid triggering proxy fallback.
                if not last_price:
                    stooq_px = _fetch_stooq_latest_close("^spx")
                    if isinstance(stooq_px, dict):
                        try:
                            last_price = float(stooq_px.get("price"))
                        except Exception:
                            last_price = None

                if calls and puts and last_price:
                    # Parse expiration date (YYYY-MM-DD format from yfinance/yahoo_http)
                    expiration_date = None
                    try:
                        expiration_date = _dt.datetime.strptime(expiration_str, "%Y-%m-%d").date() if expiration_str else None
                    except Exception:
                        expiration_date = None

                    # Combine calls and puts per strike keeping BOTH OI and Volume.
                    strike_data: Dict[float, Dict[str, float]] = {}
                    for call in calls:
                        try:
                            strike = float(call.get("strike", 0) or 0)
                        except Exception:
                            continue
                        if strike <= 0:
                            continue
                        d = strike_data.setdefault(strike, {"call_oi": 0.0, "put_oi": 0.0, "call_vol": 0.0, "put_vol": 0.0})
                        try:
                            d["call_oi"] = float(call.get("openInterest", 0) or 0)
                        except Exception:
                            d["call_oi"] = 0.0
                        try:
                            d["call_vol"] = float(call.get("volume", 0) or 0)
                        except Exception:
                            d["call_vol"] = 0.0

                    for put in puts:
                        try:
                            strike = float(put.get("strike", 0) or 0)
                        except Exception:
                            continue
                        if strike <= 0:
                            continue
                        d = strike_data.setdefault(strike, {"call_oi": 0.0, "put_oi": 0.0, "call_vol": 0.0, "put_vol": 0.0})
                        try:
                            d["put_oi"] = float(put.get("openInterest", 0) or 0)
                        except Exception:
                            d["put_oi"] = 0.0
                        try:
                            d["put_vol"] = float(put.get("volume", 0) or 0)
                        except Exception:
                            d["put_vol"] = 0.0

                    if strike_data:
                        snapshot = {
                            "symbol": "SPX",
                            "source": "yahoo",
                            "expiration": expiration_date.strftime("%B %d, %Y") if expiration_date else (expiration_str or ""),
                            "expiration_date": expiration_date.isoformat() if expiration_date else None,
                            "price": last_price,
                            "time": None,
                            "metric": metric_norm,
                        }

                        if metric_norm == "hybrid":
                            snapshot.update(_compute_spx_hybrid_levels(strike_data, current_price=last_price, window_each_side=15))
                            snapshot["note"] = "Hybrid levels: max OI + max Volume within ±15 strikes"
                            snapshot["window_levels"] = _build_spx_window_levels(strike_data, current_price=last_price, window_each_side=15)
                        else:
                            strikes = []
                            call_vals = []
                            put_vals = []
                            gammas = []
                            for strike in sorted(strike_data.keys()):
                                d = strike_data[strike]
                                if metric_norm == "openInterest":
                                    c = float(d.get("call_oi", 0.0) or 0.0)
                                    p = float(d.get("put_oi", 0.0) or 0.0)
                                else:
                                    c = float(d.get("call_vol", 0.0) or 0.0)
                                    p = float(d.get("put_vol", 0.0) or 0.0)
                                strikes.append(float(strike))
                                call_vals.append(c)
                                put_vals.append(p)
                                gammas.append((c - p) * 100)

                            if strikes:
                                df = pd.DataFrame({
                                    "Strike": strikes,
                                    "Call_OI": call_vals,
                                    "Put_OI": put_vals,
                                    "Gamma_Exposure": gammas,
                                }).sort_values("Strike").reset_index(drop=True)

                                results = analyze_0dte(df, current_price=last_price)
                                if isinstance(results, dict):
                                    snapshot.update(results)

                        _SPX_SNAPSHOT_CACHE[cache_key] = snapshot
                        _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
                        _SPX_SNAPSHOT_CACHE[last_good_key] = snapshot
                        _SPX_SNAPSHOT_CACHE[last_good_fetched_at_key] = now_ts
                        print(f"[DEBUG] Yahoo Finance SUCCESS - returning SPX snapshot with price {last_price} and metric={metric_norm}")
                        if metric_norm in {"openInterest", "hybrid"}:
                            try:
                                _maybe_capture_es_spx_conversion(snapshot, now_dt=now_dt_local)
                            except Exception:
                                pass
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
        # Prefer serving the last known good SPX snapshot over falling back to SPY proxy.
        last_good = _SPX_SNAPSHOT_CACHE.get(last_good_key)
        last_good_fetched_at = float(_SPX_SNAPSHOT_CACHE.get(last_good_fetched_at_key) or 0.0)
        if isinstance(last_good, dict) and not _is_proxy_snapshot(last_good):
            out = dict(last_good)
            out["stale"] = True
            out["stale_reason"] = "Serving last good non-proxy SPX snapshot; providers unavailable"
            out["stale_age_seconds"] = int(max(0.0, now_ts - last_good_fetched_at)) if last_good_fetched_at else None
            _SPX_SNAPSHOT_CACHE[cache_key] = out
            _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
            return out

        # Final fallback to SPY
        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "SPX"
        snapshot["note"] = "Proxy (SPY option chain) used when SPX unavailable"
        snapshot["metric"] = metric_norm
        _SPX_SNAPSHOT_CACHE[cache_key] = snapshot
        _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
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
        last_good = _SPX_SNAPSHOT_CACHE.get(last_good_key)
        last_good_fetched_at = float(_SPX_SNAPSHOT_CACHE.get(last_good_fetched_at_key) or 0.0)
        if isinstance(last_good, dict) and not _is_proxy_snapshot(last_good):
            out = dict(last_good)
            out["stale"] = True
            out["stale_reason"] = "Serving last good non-proxy SPX snapshot; expiries unavailable"
            out["stale_age_seconds"] = int(max(0.0, now_ts - last_good_fetched_at)) if last_good_fetched_at else None
            _SPX_SNAPSHOT_CACHE[cache_key] = out
            _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
            return out

        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "SPX"
        snapshot["note"] = "Proxy (SPY option chain) used when SPX expiries unavailable"
        snapshot["metric"] = metric_norm
        _SPX_SNAPSHOT_CACHE[cache_key] = snapshot
        _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
        return snapshot

    nearest_exp_label, nearest_exp_date = sorted(expiry_candidates.items(), key=lambda kv: kv[1])[0]

    strikes: list[float] = []
    calls: list[float] = []
    puts: list[float] = []
    gammas: list[float] = []

    strike_data: Dict[float, Dict[str, float]] = {}

    for row in rows:
        if (row.get("expiryDate") or "").strip() != nearest_exp_label:
            continue

        strike = _parse_pdf_number(row.get("strike"))
        if strike <= 0:
            continue

        call_oi = _parse_pdf_number(row.get("c_Openinterest"))
        put_oi = _parse_pdf_number(row.get("p_Openinterest"))

        # Best-effort volume parsing: often unavailable on Nasdaq.
        call_vol = _parse_pdf_number(
            row.get("c_Volume")
            or row.get("c_Vol")
            or row.get("c_Volume".lower())
            or row.get("c_Vol".lower())
        )
        put_vol = _parse_pdf_number(
            row.get("p_Volume")
            or row.get("p_Vol")
            or row.get("p_Volume".lower())
            or row.get("p_Vol".lower())
        )

        strike_data[float(strike)] = {
            "call_oi": float(call_oi),
            "put_oi": float(put_oi),
            "call_vol": float(call_vol),
            "put_vol": float(put_vol),
        }
        gamma_exposure = (call_oi - put_oi) * 1000

        strikes.append(float(strike))
        calls.append(float(call_oi))
        puts.append(float(put_oi))
        gammas.append(float(gamma_exposure))

    if not strikes:
        last_good = _SPX_SNAPSHOT_CACHE.get(last_good_key)
        last_good_fetched_at = float(_SPX_SNAPSHOT_CACHE.get(last_good_fetched_at_key) or 0.0)
        if isinstance(last_good, dict) and not _is_proxy_snapshot(last_good):
            out = dict(last_good)
            out["stale"] = True
            out["stale_reason"] = "Serving last good non-proxy SPX snapshot; strikes unavailable"
            out["stale_age_seconds"] = int(max(0.0, now_ts - last_good_fetched_at)) if last_good_fetched_at else None
            _SPX_SNAPSHOT_CACHE[cache_key] = out
            _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
            return out

        proxy = get_spy_snapshot_cached(max_age_seconds=max_age_seconds)
        if not proxy:
            return None
        snapshot = dict(proxy)
        snapshot["symbol"] = "SPX"
        snapshot["note"] = "Proxy (SPY option chain) used when SPX strikes unavailable"
        snapshot["metric"] = metric_norm
        _SPX_SNAPSHOT_CACHE[cache_key] = snapshot
        _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
        return snapshot

    df = pd.DataFrame({
        "Strike": strikes,
        "Call_OI": calls,
        "Put_OI": puts,
        "Gamma_Exposure": gammas,
    }).sort_values("Strike").reset_index(drop=True)

    snapshot: Dict[str, Any] = {
        "symbol": "SPX",
        "source": "nasdaq",
        "expiration": nearest_exp_label,
        "expiration_date": nearest_exp_date.isoformat(),
        "price": float(last_sale_price) if last_sale_price else None,
        "time": last_sale_time or None,
        "metric": metric_norm,
    }

    # If Nasdaq didn't provide a usable price, fill it from Stooq ^spx so hybrid window selection works.
    if not snapshot.get("price"):
        stooq_px = _fetch_stooq_latest_close("^spx")
        if isinstance(stooq_px, dict):
            try:
                snapshot["price"] = float(stooq_px.get("price"))
                snapshot["price_source"] = "stooq^spx"
            except Exception:
                pass

    if metric_norm == "hybrid":
        cur_px = None
        try:
            cur_px = float(snapshot.get("price")) if snapshot.get("price") is not None else None
        except Exception:
            cur_px = None
        snapshot.update(_compute_spx_hybrid_levels(strike_data, current_price=cur_px, window_each_side=15))
        snapshot["note"] = "Hybrid levels: max OI + max Volume within ±15 strikes (volume may be missing on Nasdaq)"
        snapshot["window_levels"] = _build_spx_window_levels(strike_data, current_price=cur_px, window_each_side=15)
    else:
        cur_px = None
        try:
            cur_px = float(snapshot.get("price")) if snapshot.get("price") is not None else None
        except Exception:
            cur_px = None
        results = analyze_0dte(df, current_price=cur_px)
        if isinstance(results, dict):
            snapshot.update(results)

    _SPX_SNAPSHOT_CACHE[cache_key] = snapshot
    _SPX_SNAPSHOT_CACHE[fetched_at_key] = now_ts
    if metric_norm in {'openInterest', 'hybrid'}:
        try:
            _maybe_capture_es_spx_conversion(snapshot, now_dt=now_dt_local)
        except Exception:
            pass

    if not _is_proxy_snapshot(snapshot):
        _SPX_SNAPSHOT_CACHE[last_good_key] = snapshot
        _SPX_SNAPSHOT_CACHE[last_good_fetched_at_key] = now_ts
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


def get_spx_index_price_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    """Fetch SPX index price from Stooq (^spx) without proxy fallback."""

    now = time.time()
    cached = _SPX_INDEX_PRICE_CACHE.get("value")
    fetched_at = float(_SPX_INDEX_PRICE_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    # Operator override (useful when providers are rate-limited).
    override = (os.getenv("SPX_PRICE_OVERRIDE") or os.getenv("SPX_PRICE") or "").strip()
    if override:
        seeded = _seed_spx_price_manual(override, note="env override")
        if seeded:
            return seeded

    data = _fetch_stooq_latest_close("^spx")
    if not data:
        # Fallback: Yahoo quote endpoint.
        data = _fetch_yahoo_quote_price("^GSPC")
    if not data:
        # Final fallback: use the price embedded in the SPX snapshot (Nasdaq/Yahoo options)
        # ONLY if it looks like a real SPX price (not a SPY proxy).
        try:
            snap = get_spx_snapshot_cached(metric='volume', max_age_seconds=max_age_seconds) or {}
        except Exception:
            snap = {}

        note = (snap.get('note') or '') if isinstance(snap, dict) else ''
        if 'proxy' in note.lower():
            return None

        try:
            px = float(snap.get('price'))
        except Exception:
            px = None

        # SPX index is typically in the thousands; reject obviously wrong values.
        if px is None or px < 1000:
            return None

        data = {
            "symbol": "^GSPC",
            "price": px,
            "date": "",
            "time": "",
            "source": "spx_snapshot",
        }

    data["instrument"] = "SPX Index"
    src = (data.get("source") or "").strip().lower()
    if src == "stooq":
        data["note"] = "Stooq ^spx; quote may be delayed"
    elif src == "yahoo_quote":
        data["note"] = "Yahoo ^GSPC quote (fallback when Stooq unavailable)"
    elif src == "spx_snapshot":
        data["note"] = "SPX snapshot price (fallback when Stooq/Yahoo quote unavailable)"
    else:
        data["note"] = "SPX index price (fallback source)"
    _SPX_INDEX_PRICE_CACHE["value"] = data
    _SPX_INDEX_PRICE_CACHE["fetched_at"] = now
    return data


def get_es_spx_spread_cached(max_age_seconds: int = 60 * 60) -> Optional[Dict[str, Any]]:
    """Compute ES–SPX spread (ES price minus SPX index) and cache it hourly.

    Uses Stooq for both legs to keep the spread source consistent.
    """

    now = time.time()
    cached = _ES_SPX_SPREAD_CACHE.get("value")
    fetched_at = float(_ES_SPX_SPREAD_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    es = get_es_price_cached(max_age_seconds=60)
    spx = get_spx_index_price_cached(max_age_seconds=60)
    if not es or not spx:
        # Stale-tolerant: if we have a previously computed spread, serve it.
        if isinstance(cached, dict):
            out = dict(cached)
            out["stale"] = True
            out["stale_reason"] = "Missing ES/SPX price for refresh"
            out["stale_age_seconds"] = int(max(0.0, now - fetched_at)) if fetched_at else None
            # Throttle retries.
            _ES_SPX_SPREAD_CACHE["fetched_at"] = now
            return out
        return None

    try:
        es_price = float(es.get("price"))
        spx_price = float(spx.get("price"))
    except Exception:
        return None

    # Sanity checks to avoid publishing nonsense due to proxy/mis-parsing.
    if es_price < 1000 or spx_price < 1000:
        if isinstance(cached, dict):
            out = dict(cached)
            out["stale"] = True
            out["stale_reason"] = "Invalid ES/SPX price for spread"
            out["invalid_prices"] = {"es_price": es_price, "spx_price": spx_price}
            out["stale_age_seconds"] = int(max(0.0, now - fetched_at)) if fetched_at else None
            _ES_SPX_SPREAD_CACHE["fetched_at"] = now
            return out
        return None

    spread = es_price - spx_price

    # Reject extreme spreads (almost certainly proxy/mismatch of instruments).
    if abs(spread) > 1000:
        if isinstance(cached, dict):
            out = dict(cached)
            out["stale"] = True
            out["stale_reason"] = "Spread out of expected range"
            out["invalid_spread"] = spread
            out["invalid_prices"] = {"es_price": es_price, "spx_price": spx_price}
            out["stale_age_seconds"] = int(max(0.0, now - fetched_at)) if fetched_at else None
            _ES_SPX_SPREAD_CACHE["fetched_at"] = now
            return out
        return None
    payload = {
        "es_price": es_price,
        "spx_price": spx_price,
        "spread": spread,
        "es": es,
        "spx": spx,
        "computed_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "stale": False,
    }

    _ES_SPX_SPREAD_CACHE["value"] = payload
    _ES_SPX_SPREAD_CACHE["fetched_at"] = now
    return payload


def get_es_price_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _ES_PRICE_CACHE.get("value")
    fetched_at = float(_ES_PRICE_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    # Operator override (useful when providers are rate-limited).
    override = (os.getenv("ES_PRICE_OVERRIDE") or os.getenv("ES_PRICE") or "").strip()
    if override:
        try:
            _seed_es_price_manual(float(override), note="env override")
            return _ES_PRICE_CACHE.get("value")
        except Exception:
            pass

    # ES continuous future on Stooq.
    data = _fetch_stooq_latest_close("es.f")
    if not data:
        # Fallback: Yahoo quote endpoint.
        data = _fetch_yahoo_quote_price("ES=F")
        if data:
            data["instrument"] = "ES Futures"
            data["note"] = "Yahoo ES=F quote (fallback when Stooq unavailable)"
            data["stale"] = False
            _ES_PRICE_CACHE["value"] = data
            _ES_PRICE_CACHE["fetched_at"] = now
            _ES_PRICE_CACHE["last_success_at"] = now
            return data

        # Fallback: last analysis current_price (Mongo) if available.
        try:
            doc = _load_last_analysis() or {}
            analysis = doc.get('analysis') if isinstance(doc, dict) else {}
            if isinstance(analysis, dict) and analysis.get('current_price') is not None:
                _seed_es_price_manual(float(analysis.get('current_price')), note="last analysis")
                return _ES_PRICE_CACHE.get("value")
        except Exception:
            pass

        # If Stooq is temporarily unavailable, serve the last known value (stale-tolerant)
        # instead of returning 503 to the UI.
        if cached:
            last_success_at = float(_ES_PRICE_CACHE.get("last_success_at") or fetched_at or 0.0)
            stale_age = max(0.0, now - last_success_at) if last_success_at else None
            out = dict(cached)
            out["stale"] = True
            if stale_age is not None:
                out["stale_age_seconds"] = int(stale_age)
            out["note"] = (out.get("note") or "") + " | stale (Stooq temporarily unavailable)"

            # Throttle retries: treat this as a fresh cache window so we don't hammer Stooq.
            _ES_PRICE_CACHE["fetched_at"] = now
            return out

        return None

    data["instrument"] = "ES Futures"
    data["note"] = "Stooq es.f (continuous); quote may be delayed"
    data["stale"] = False
    _ES_PRICE_CACHE["value"] = data
    _ES_PRICE_CACHE["fetched_at"] = now
    _ES_PRICE_CACHE["last_success_at"] = now
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
# TRADING CHECKLIST — MongoDB helpers
# ============================================================================

_MONGO_CHECKLIST_COLLECTION = None


def _get_checklist_collection():
    """Return Mongo collection for trading checklists, or None if not configured."""
    global _MONGO_CLIENT, _MONGO_CHECKLIST_COLLECTION
    if _MONGO_CHECKLIST_COLLECTION is not None:
        return _MONGO_CHECKLIST_COLLECTION

    if MongoClient is None:
        return None

    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None

    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_CHECKLIST_COLLECTION") or "trading_checklist").strip()

    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)

        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]

        # Unique per date_key
        try:
            coll.create_index([("date_key", 1)], unique=True)
        except Exception:
            pass

        _MONGO_CHECKLIST_COLLECTION = coll
        return _MONGO_CHECKLIST_COLLECTION
    except Exception:
        return None


def _checklist_upsert(date_key: str, checklist_data: dict) -> bool:
    coll = _get_checklist_collection()
    if coll is None:
        return False

    now_dt = _dt.datetime.utcnow()
    try:
        coll.update_one(
            {"date_key": date_key},
            {
                "$set": {
                    "date_key": date_key,
                    "checklist": checklist_data,
                    "updated_at": now_dt,
                },
                "$setOnInsert": {"created_at": now_dt},
            },
            upsert=True,
        )
        return True
    except Exception:
        return False


def _checklist_get(date_key: str) -> Optional[dict]:
    coll = _get_checklist_collection()
    if coll is None:
        return None

    try:
        doc = coll.find_one({"date_key": date_key})
    except Exception:
        return None

    if not doc or not isinstance(doc, dict):
        return None

    return {
        "date_key": doc.get("date_key"),
        "checklist": doc.get("checklist") or {},
        "updated_at": doc.get("updated_at").isoformat() if doc.get("updated_at") else None,
    }


def _checklist_history(limit: int = 30) -> list:
    coll = _get_checklist_collection()
    if coll is None:
        return []

    try:
        docs = list(coll.find({}, sort=[("date_key", -1)], limit=limit))
    except TypeError:
        docs = list(coll.find({}).sort("date_key", -1).limit(limit))
    except Exception:
        return []

    out = []
    for doc in docs:
        cl = doc.get("checklist") or {}
        session = cl.get("session") or {}
        trades = cl.get("trades") or []

        pnl = None
        try:
            pnl = float(session.get("daily_pnl")) if session.get("daily_pnl") not in (None, "", "null") else None
        except Exception:
            pass

        out.append({
            "date_key": doc.get("date_key"),
            "trade_count": len(trades),
            "session_pnl": pnl,
            "updated_at": doc.get("updated_at").isoformat() if doc.get("updated_at") else None,
        })
    return out


# ============================================================================
# WEB ROUTES - Authentication & Admin
# ============================================================================


@app.route('/')
def index():
    return render_template('index.html')


# ============================================================================
# TRADING JOURNAL (TradeZella-style interface)
# ============================================================================

@app.route('/journal')
def journal_dashboard():
    return render_template('journal_dashboard.html', active_page='dashboard')


@app.route('/journal/trade-view')
def journal_trade_view():
    return render_template('journal_trade_view.html', active_page='trade-view')


@app.route('/journal/day-view')
def journal_day_view():
    return render_template('journal_day_view.html', active_page='day-view')


@app.route('/favicon.ico')
def favicon():
    # Avoid a 404 page if the browser requests /favicon.ico.
    # Returning 204 is fine; you can later add a real icon under /static.
    return ('', 204)


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

        next_url = _sanitize_next_url(session.pop('next_url', None))
        if next_url:
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
    # Aggregation: only the most recent 'login' event per user email.
    # (Using Mongo's aggregation pipeline keeps memory bounded even at scale.)
    try:
        docs = list(coll.aggregate([
            {'$match': {'event': 'login', 'user.email': {'$ne': None}}},
            {'$sort': {'created_at': -1}},
            {'$group': {
                '_id': {'$toLower': '$user.email'},
                'latest': {'$first': '$$ROOT'},
            }},
            {'$replaceRoot': {'newRoot': '$latest'}},
            {'$sort': {'created_at': -1}},
            {'$limit': limit},
        ]))
    except Exception:
        docs = []

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


@app.route('/api/spx-index-price', methods=['GET'])
def spx_index_price():
    """Return SPX index price (prefers ^SPX; avoids SPY proxy fallbacks).

    This is intended for ES–SPX spread calculations during cash hours.
    """

    force = (request.args.get('force') or '').strip() == '1'
    try:
        data = get_spx_index_price_cached(max_age_seconds=0 if force else 60)
    except Exception as e:
        return jsonify({"error": f"Impossibile recuperare il prezzo SPX: {e}"})

    if not data:
        return jsonify({"error": "Impossibile recuperare il prezzo SPX in questo momento"})

    return jsonify(data)


@app.route('/api/es-price', methods=['GET'])
def es_price():
    data = get_es_price_cached()
    if not data:
        # Return 200 with error payload to avoid noisy "Failed to load resource" in browsers.
        return jsonify({"error": "Impossibile recuperare il prezzo ES in questo momento"})

    return jsonify(data)


@app.route('/api/es-spx-spread', methods=['GET'])
def es_spx_spread():
    """Return ES–SPX spread (ES minus SPX index), cached hourly."""

    force = (request.args.get('force') or '').strip() == '1'
    try:
        data = get_es_spx_spread_cached(max_age_seconds=0 if force else 60 * 60)
    except Exception as e:
        return jsonify({"error": f"Impossibile calcolare lo spread ES–SPX: {e}"})

    if not data:
        # Return 200 with error payload to avoid noisy "Failed to load resource" in browsers.
        return jsonify({"error": "Impossibile calcolare lo spread ES–SPX in questo momento"})

    # Attach cache age (best-effort)
    try:
        fetched_at = float(_ES_SPX_SPREAD_CACHE.get('fetched_at') or 0.0)
        if fetched_at:
            data = dict(data)
            data['cache_age_seconds'] = int(max(0.0, time.time() - fetched_at))
    except Exception:
        pass

    return jsonify(data)


@app.route('/api/es-spx-overnight-basis', methods=['GET'])
def es_spx_overnight_basis():
    """Return stable ES/SPX close basis for after-hours monitoring.

    The UI uses this to freeze SPX OI→ES converted levels overnight.
    """

    force = (request.args.get('force') or '').strip() == '1'
    try:
        data = get_es_spx_overnight_basis_cached(max_age_seconds=0 if force else 10 * 60)
    except Exception as e:
        return jsonify({"error": f"Impossibile recuperare la base overnight ES/SPX: {e}"})

    if not data:
        return jsonify({"error": "Impossibile recuperare la base overnight ES/SPX in questo momento"})

    try:
        fetched_at = float(_ES_SPX_OVERNIGHT_BASIS_CACHE.get('fetched_at') or 0.0)
        if fetched_at:
            data = dict(data)
            data['cache_age_seconds'] = int(max(0.0, time.time() - fetched_at))
    except Exception:
        pass

    return jsonify(data)


@app.route('/api/cot-sp500', methods=['GET'])
def api_cot_sp500():
    """Return weekly COT report for S&P 500 (non-commercial focus)."""

    force = (request.args.get('force') or '').strip() == '1'
    try:
        data = get_cot_sp500_cached(max_age_seconds=0 if force else 60 * 60)
    except Exception as e:
        return jsonify({"error": f"Impossibile recuperare il COT S&P 500: {e}"})

    if not data:
        return jsonify({"error": "Impossibile recuperare il COT S&P 500 in questo momento"})

    try:
        fetched_at = float(_COT_SP500_CACHE.get('fetched_at') or 0.0)
        if fetched_at:
            data = dict(data)
            data['cache_age_seconds'] = int(max(0.0, time.time() - fetched_at))
    except Exception:
        pass

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
    # Always return both CP (price) and GF (flip) so the UI can show all levels together.
    data_price = get_nvda_snapshot_cached(levels_mode='price')
    data_flip = get_nvda_snapshot_cached(levels_mode='flip')
    if not data_price and not data_flip:
        return jsonify({"error": "Impossibile recuperare NVDA option chain in questo momento"}), 503

    combined = {
        "symbol": "NVDA",
        "price": data_price,
        "flip": data_flip,
    }
    return jsonify(combined)


@app.route('/api/spy-snapshot', methods=['GET'])
def spy_snapshot():
    data = get_spy_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare SPY option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/msft-snapshot', methods=['GET'])
def msft_snapshot():
    # Always return both CP (price) and GF (flip) so the UI can show all levels together.
    data_price = get_msft_snapshot_cached(levels_mode='price')
    data_flip = get_msft_snapshot_cached(levels_mode='flip')
    if not data_price and not data_flip:
        return jsonify({"error": "Impossibile recuperare MSFT option chain in questo momento"}), 503

    combined = {
        "symbol": "MSFT",
        "price": data_price,
        "flip": data_flip,
    }
    return jsonify(combined)


@app.route('/api/spx-snapshot', methods=['GET'])
def spx_snapshot():
    # Allow force refresh for testing (add ?force=1 to URL)
    force = request.args.get('force') == '1'
    if force:
        print("[DEBUG] Force refresh SPX data requested")
        # Reset metric-specific cache timestamps
        _SPX_SNAPSHOT_CACHE["fetched_at_volume"] = 0.0
        _SPX_SNAPSHOT_CACHE["fetched_at_openInterest"] = 0.0
        _SPX_SNAPSHOT_CACHE["fetched_at_hybrid"] = 0.0
        _SPX_SNAPSHOT_CACHE["value_volume"] = None
        _SPX_SNAPSHOT_CACHE["value_openInterest"] = None
        _SPX_SNAPSHOT_CACHE["value_hybrid"] = None
    
    # Get metric parameter (volume | openInterest | hybrid)
    metric = request.args.get('metric', 'volume')
    if metric not in ['volume', 'openInterest', 'hybrid']:
        metric = 'volume'
    
    try:
        data = get_spx_snapshot_cached(metric=metric, max_age_seconds=0 if force else 60)
    except Exception as e:
        # Never let the request crash/abort the connection (which becomes a fetch "Load failed" client-side).
        return jsonify({"error": f"SPX snapshot failed: {e}", "metric": metric}), 200

    if not data:
        return jsonify({"error": "Impossibile recuperare SPX option chain in questo momento", "metric": metric}), 200
    return jsonify(data)


@app.route('/api/debug/yahoo-options', methods=['GET'])
def api_debug_yahoo_options():
    """Debug endpoint: show the JSON-shaped options payload used by the app.

    This is NOT HTML scraping. It reflects the same underlying Yahoo options data
    that powers https://finance.yahoo.com/quote/%5ESPX/options/?straddle=true.

    Query params:
      - symbol: defaults to ^SPX
      - limit: number of calls/puts rows to include (default 3)
    """

    symbol = (request.args.get('symbol') or '^SPX').strip() or '^SPX'
    try:
        limit = int((request.args.get('limit') or '3').strip())
    except Exception:
        limit = 3
    limit = max(0, min(limit, 25))

    data = _fetch_yahoo_options(symbol)
    if not isinstance(data, dict):
        return jsonify({
            "error": "Yahoo options unavailable",
            "symbol": symbol,
            "hint": "If you see this intermittently, Yahoo may be rate-limiting or blocking direct access.",
        }), 200

    calls = data.get('calls') or []
    puts = data.get('puts') or []

    def _slim_rows(rows: list) -> list:
        out = []
        for r in rows[:limit]:
            if not isinstance(r, dict):
                continue
            out.append({
                "contractSymbol": r.get('contractSymbol') or r.get('contract_symbol'),
                "strike": r.get('strike'),
                "bid": r.get('bid'),
                "ask": r.get('ask'),
                "lastPrice": r.get('lastPrice') or r.get('last_price'),
                "volume": r.get('volume'),
                "openInterest": r.get('openInterest') or r.get('open_interest'),
                "impliedVolatility": r.get('impliedVolatility') or r.get('implied_volatility'),
                "inTheMoney": r.get('inTheMoney') or r.get('in_the_money'),
            })
        return out

    return jsonify({
        "symbol": symbol,
        "source": data.get('source') or 'yahoo',
        "price": data.get('price'),
        "expiration": data.get('expiration'),
        "counts": {"calls": len(calls), "puts": len(puts)},
        "sample": {
            "calls": _slim_rows(calls),
            "puts": _slim_rows(puts),
        },
        "yahoo_options_page": f"https://finance.yahoo.com/quote/{urllib.parse.quote(symbol)}/options/?straddle=true",
        "yahoo_json_endpoint_base": f"https://query2.finance.yahoo.com/v7/finance/options/{urllib.parse.quote(symbol)}",
    }), 200


@app.route('/api/spx-0dte-volume', methods=['GET'])
def spx_0dte_volume():
    """SPX 0DTE key levels from Yahoo options using Volume only."""

    force = (request.args.get('force') or '').strip() == '1'
    try:
        data = get_spx_0dte_volume_levels_cached(max_age_seconds=0 if force else 5 * 60)
    except Exception as e:
        return jsonify({"error": f"SPX 0DTE volume failed: {e}"}), 200

    if not isinstance(data, dict):
        return jsonify({"error": "Impossibile recuperare SPX 0DTE (volume) in questo momento"}), 200

    return jsonify(data), 200


@app.route('/api/es-spx-oi-to-es', methods=['GET'])
def api_es_spx_oi_to_es_get():
    """Return ES levels converted from SPX OI supports/resistances.

    - If today's 14:30 capture exists in DB, return it.
    - Otherwise compute a "morning" provisional conversion from the most recent stored baseline.

    Query params:
      - date: YYYY-MM-DD (default: today)
      - kind: auto|1430|close|morning
    """

    coll = _get_mongo_conversions_collection()
    if coll is None:
        return jsonify({'error': 'MongoDB non configurato'}), 503

    date_key = (request.args.get('date') or '').strip()
    if not date_key:
        date_key = _dt.date.today().isoformat()

    kind = (request.args.get('kind') or 'auto').strip().lower()
    if kind == 'auto':
        stored = _conv_mongo_get(date_key, '1430')
        if stored:
            # Refresh spread intraday (hourly cache) without mutating DB baseline.
            if date_key == _dt.date.today().isoformat():
                spread_payload = get_es_spx_spread_cached(max_age_seconds=60 * 60)
                if spread_payload and isinstance(spread_payload, dict):
                    try:
                        spread = float(spread_payload.get('spread'))
                        es_price = float(spread_payload.get('es_price'))
                        spx_price = float(spread_payload.get('spx_price'))
                    except Exception:
                        spread = None
                        es_price = None
                        spx_price = None

                    raw_s = stored.get('spx_supports_raw') if isinstance(stored.get('spx_supports_raw'), list) else []
                    raw_r = stored.get('spx_resistances_raw') if isinstance(stored.get('spx_resistances_raw'), list) else []
                    if spread is not None and raw_s and raw_r:
                        try:
                            converted_s = [float(v) + spread for v in raw_s]
                            converted_r = [float(v) + spread for v in raw_r]
                            out = dict(stored)
                            out['spread'] = spread
                            out['es_price'] = es_price
                            out['spx_price'] = spx_price
                            out['supports'] = converted_s
                            out['resistances'] = converted_r
                            out['spread_updated_at'] = _dt.datetime.now().strftime('%H:%M')
                            return jsonify(out)
                        except Exception:
                            pass

            return jsonify(stored)
        computed = _compute_es_spx_conversion_from_baseline(date_key)
        if computed:
            return jsonify(computed)
        # First-run fallback: compute from current SPX snapshot (no persistence).
        current = _compute_es_spx_conversion_from_current_snapshot(date_key)
        if current:
            # Optional: persist a seeded baseline for today (admin-only) so we won't keep recomputing
            # and tomorrow morning has a usable baseline even before a real 14:30/cash-close capture.
            if _is_admin():
                seed_doc = dict(current)
                seed_doc['date_key'] = date_key
                seed_doc['capture_kind'] = '1430'
                seed_doc['captured_at'] = _dt.datetime.now().strftime('%H:%M')
                seed_doc['is_seed'] = True
                if _conv_mongo_upsert(seed_doc):
                    stored_seed = _conv_mongo_get(date_key, '1430')
                    if stored_seed:
                        return jsonify(stored_seed)
            return jsonify(current)
        # Fallback: if we have a close for today (rare), return it.
        stored_close = _conv_mongo_get(date_key, 'close')
        if stored_close:
            return jsonify(stored_close)
        return jsonify({'error': 'No stored baseline available yet'}), 404

    if kind in ('1430', 'close'):
        stored = _conv_mongo_get(date_key, kind)
        if stored:
            return jsonify(stored)
        return jsonify({'error': 'Not found'}), 404

    if kind == 'morning':
        computed = _compute_es_spx_conversion_from_baseline(date_key)
        if computed:
            return jsonify(computed)
        current = _compute_es_spx_conversion_from_current_snapshot(date_key)
        if current:
            if _is_admin():
                seed_doc = dict(current)
                seed_doc['date_key'] = date_key
                seed_doc['capture_kind'] = '1430'
                seed_doc['captured_at'] = _dt.datetime.now().strftime('%H:%M')
                seed_doc['is_seed'] = True
                _conv_mongo_upsert(seed_doc)
            return jsonify(current)
        return jsonify({'error': 'No stored baseline available yet'}), 404

    return jsonify({'error': 'Invalid kind'}), 400


@app.route('/api/es-spx-oi-to-es', methods=['POST'])
@login_required
def api_es_spx_oi_to_es_post():
    """Persist a conversion record (typically 14:30 or close) into MongoDB.

    Expected JSON fields:
      date_key, capture_kind, captured_at, spread, es_price, spx_price,
      supports, resistances, spx_supports_raw, spx_resistances_raw
    """

    try:
        payload = request.get_json(silent=True) or {}
    except Exception:
        payload = {}

    if not isinstance(payload, dict):
        return jsonify({'error': 'Invalid payload'}), 400

    date_key = (payload.get('date_key') or '').strip()
    capture_kind = (payload.get('capture_kind') or '').strip()
    if not date_key or not capture_kind:
        return jsonify({'error': 'Missing date_key/capture_kind'}), 400

    if capture_kind not in ('1430', 'close', 'morning'):
        return jsonify({'error': 'Invalid capture_kind'}), 400

    # Coerce lists.
    def _as_num_list(v):
        out = []
        if not isinstance(v, list):
            return out
        for n in v:
            try:
                out.append(float(n))
            except Exception:
                continue
        return out

    doc = {
        'date_key': date_key,
        'capture_kind': capture_kind,
        'is_seed': bool(payload.get('is_seed')),
        'captured_at': (payload.get('captured_at') or ''),
        'based_on_date_key': (payload.get('based_on_date_key') or ''),
        'spread': payload.get('spread'),
        'es_price': payload.get('es_price'),
        'spx_price': payload.get('spx_price'),
        'supports': _as_num_list(payload.get('supports')),
        'resistances': _as_num_list(payload.get('resistances')),
        'spx_supports_raw': _as_num_list(payload.get('spx_supports_raw')),
        'spx_resistances_raw': _as_num_list(payload.get('spx_resistances_raw')),
        'spx_supports_meta': payload.get('spx_supports_meta') if isinstance(payload.get('spx_supports_meta'), list) else None,
        'spx_resistances_meta': payload.get('spx_resistances_meta') if isinstance(payload.get('spx_resistances_meta'), list) else None,
    }

    stored = _conv_mongo_upsert(doc)
    return jsonify({'ok': bool(stored)})


@app.route('/api/es-spx-oi-to-es/bootstrap', methods=['POST'])
@login_required
def api_es_spx_oi_to_es_bootstrap():
    """Create/overwrite today's baseline conversion record immediately.

    This is useful to seed Mongo so that "morning" conversions work tomorrow
    even before the first scheduled 14:30/cash-close capture has happened.

    Notes:
      - Stores as capture_kind=1430 on today's date_key.
      - A real 14:30 capture later will overwrite this record.
      - Admin-only (or any authenticated user if ADMIN_EMAILS is not set).
    """

    if not _is_admin():
        return jsonify({'error': 'Forbidden'}), 403

    coll = _get_mongo_conversions_collection()
    if coll is None:
        return jsonify({'error': 'MongoDB non configurato'}), 503

    now_dt = _dt.datetime.now()
    today_key = now_dt.date().isoformat()

    # Pull freshest snapshots.
    spx = get_spx_snapshot_cached(metric='hybrid', max_age_seconds=0) or {}
    if not spx or not isinstance(spx, dict) or spx.get('error'):
        return jsonify({'error': 'Impossibile recuperare SPX snapshot'}), 503

    es = get_es_price_cached(max_age_seconds=0) or {}
    es_price = es.get('price')
    spx_price = spx.get('price')
    try:
        es_price_f = float(es_price)
        spx_price_f = float(spx_price)
    except Exception:
        return jsonify({'error': 'Missing ES/SPX prices'}), 503

    spread = es_price_f - spx_price_f

    supports = spx.get('supports') if isinstance(spx.get('supports'), list) else []
    resistances = spx.get('resistances') if isinstance(spx.get('resistances'), list) else []

    raw_s = []
    raw_r = []
    for lvl in supports:
        if not isinstance(lvl, dict):
            continue
        try:
            raw_s.append(float(lvl.get('strike')))
        except Exception:
            continue
    for lvl in resistances:
        if not isinstance(lvl, dict):
            continue
        try:
            raw_r.append(float(lvl.get('strike')))
        except Exception:
            continue

    if not raw_s and not raw_r:
        return jsonify({'error': 'SPX snapshot missing levels'}), 503

    converted_s = [v + spread for v in raw_s]
    converted_r = [v + spread for v in raw_r]

    doc = {
        'date_key': today_key,
        'capture_kind': '1430',
        'is_seed': True,
        'captured_at': now_dt.strftime('%H:%M'),
        'spread': spread,
        'es_price': es_price_f,
        'spx_price': spx_price_f,
        'supports': converted_s,
        'resistances': converted_r,
        'spx_supports_raw': raw_s,
        'spx_resistances_raw': raw_r,
    }

    ok = _conv_mongo_upsert(doc)
    if not ok:
        return jsonify({'error': 'Failed to persist conversion baseline'}), 500

    stored = _conv_mongo_get(today_key, '1430')
    return jsonify({'ok': True, 'record': stored or doc})


@app.route('/api/xsp-snapshot', methods=['GET'])
def xsp_snapshot():
    data = get_xsp_snapshot_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare XSP option chain in questo momento"}), 503
    return jsonify(data)


@app.route('/api/aapl-snapshot', methods=['GET'])
def aapl_snapshot():
    # Always return both CP (price) and GF (flip) so the UI can show all levels together.
    data_price = get_aapl_snapshot_cached(levels_mode='price')
    data_flip = get_aapl_snapshot_cached(levels_mode='flip')
    if not data_price and not data_flip:
        return jsonify({"error": "Impossibile recuperare AAPL option chain in questo momento"}), 503

    return jsonify({
        "symbol": "AAPL",
        "price": data_price,
        "flip": data_flip,
    })


@app.route('/api/goog-snapshot', methods=['GET'])
def goog_snapshot():
    # Always return both CP (price) and GF (flip) so the UI can show all levels together.
    data_price = get_goog_snapshot_cached(levels_mode='price')
    data_flip = get_goog_snapshot_cached(levels_mode='flip')
    if not data_price and not data_flip:
        return jsonify({"error": "Impossibile recuperare GOOG option chain in questo momento"}), 503

    return jsonify({
        "symbol": "GOOG",
        "price": data_price,
        "flip": data_flip,
    })


@app.route('/api/amzn-snapshot', methods=['GET'])
def amzn_snapshot():
    # Always return both CP (price) and GF (flip) so the UI can show all levels together.
    data_price = get_amzn_snapshot_cached(levels_mode='price')
    data_flip = get_amzn_snapshot_cached(levels_mode='flip')
    if not data_price and not data_flip:
        return jsonify({"error": "Impossibile recuperare AMZN option chain in questo momento"}), 503

    return jsonify({
        "symbol": "AMZN",
        "price": data_price,
        "flip": data_flip,
    })


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

        # Seed ES price cache from user-provided input (best-effort fallback).
        if current_price is not None:
            try:
                _seed_es_price_manual(float(current_price), note="pdf input")
            except Exception:
                pass

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
# WEB ROUTES - Trading Checklist
# ============================================================================


@app.route('/checklist')
@login_required
def checklist_page():
    return render_template('checklist.html')


@app.route('/api/checklist/<date_key>', methods=['GET'])
@login_required
def api_checklist_get(date_key):
    """Return the checklist for a given date (YYYY-MM-DD)."""
    import re as _re
    if not _re.match(r'^\d{4}-\d{2}-\d{2}$', date_key):
        return jsonify({'error': 'Invalid date format'}), 400

    doc = _checklist_get(date_key)
    if not doc:
        return jsonify({'date_key': date_key, 'checklist': {}, 'found': False})

    return jsonify({'date_key': date_key, 'checklist': doc.get('checklist') or {}, 'found': True, 'updated_at': doc.get('updated_at')})


@app.route('/api/checklist/save', methods=['POST'])
@login_required
def api_checklist_save():
    """Save (upsert) the full checklist for a date."""
    payload = request.get_json(silent=True) or {}
    date_key = (payload.get('date_key') or '').strip()
    checklist_data = payload.get('checklist')

    import re as _re
    if not date_key or not _re.match(r'^\d{4}-\d{2}-\d{2}$', date_key):
        return jsonify({'error': 'Invalid date_key'}), 400

    if not isinstance(checklist_data, dict):
        return jsonify({'error': 'Invalid checklist data'}), 400

    ok = _checklist_upsert(date_key, checklist_data)
    if not ok:
        # MongoDB not configured: return success anyway (data is ephemeral in session)
        return jsonify({'ok': True, 'persisted': False, 'note': 'MongoDB not configured — data not persisted'})

    return jsonify({'ok': True, 'persisted': True})


@app.route('/api/checklist/history', methods=['GET'])
@login_required
def api_checklist_history():
    """Return a list of recent checklist dates with summary stats."""
    try:
        limit = int((request.args.get('limit') or '30').strip())
        limit = max(1, min(limit, 365))
    except Exception:
        limit = 30

    entries = _checklist_history(limit=limit)
    return jsonify({'entries': entries})


@app.route('/api/checklist/reset', methods=['POST'])
@login_required
def api_checklist_reset():
    """Delete (reset) the checklist for a given date."""
    payload = request.get_json(silent=True) or {}
    date_key = (payload.get('date_key') or '').strip()

    import re as _re
    if not date_key or not _re.match(r'^\d{4}-\d{2}-\d{2}$', date_key):
        return jsonify({'error': 'Invalid date_key'}), 400

    coll = _get_checklist_collection()
    if coll is None:
        return jsonify({'ok': True, 'persisted': False, 'note': 'MongoDB not configured'})

    try:
        coll.delete_one({'date_key': date_key})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'ok': True, 'date_key': date_key})


# ============================================================================
# APEX CSV IMPORT — parse Overcharts order export into round-trip trades
# ============================================================================

def _parse_apex_csv(content: str, date_filter: str = '') -> list:
    """Parse an Overcharts/Apex TSV export and return a list of round-trip trade dicts.

    Only 'Filled' rows are considered.  If date_filter is provided (YYYY-MM-DD),
    only fills whose fill date matches that day are included — this is essential
    because Overcharts exports often span multiple trading days.

    Fills are grouped by account, then sorted chronologically within each account.
    Position tracking (FIFO) groups fills into round trips per account.
    Each resulting trade is tagged with 'account' and 'imported': True.

    ES multiplier: 1 point = $50 (E-mini S&P 500).
    """
    import csv as _csv
    import io as _io
    from collections import defaultdict as _defaultdict

    _ES_MULTIPLIER = 50.0

    # Parse date_filter into a date object for comparison
    filter_date = None
    if date_filter:
        try:
            filter_date = _dt.datetime.strptime(date_filter, '%Y-%m-%d').date()
        except Exception:
            pass

    reader = _csv.reader(_io.StringIO(content), delimiter='\t')
    rows = list(reader)

    if len(rows) < 2:
        return []

    fills = []
    for row in rows[1:]:
        if len(row) < 17:
            continue
        state = row[8].strip()
        if state != 'Filled':
            continue

        side = row[1].strip()           # 'Buy' or 'Sell'
        symbol = row[0].strip()         # e.g. 'MESH26', 'MESH26' (ES), 'MESH26' (MES)
        filled_qty_str = row[5].strip()
        avg_price_str = row[6].strip()
        account = row[12].strip() if len(row) > 12 else 'Unknown'
        fill_date = row[15].strip()     # MM/DD/YYYY
        fill_time_raw = row[16].strip() # HH:MM:SS.mmm

        try:
            filled_qty = int(filled_qty_str)
            avg_price = float(avg_price_str)
        except (ValueError, TypeError):
            continue

        if filled_qty <= 0 or avg_price <= 0:
            continue

        try:
            dt = _dt.datetime.strptime(
                f"{fill_date} {fill_time_raw[:8]}", "%m/%d/%Y %H:%M:%S"
            )
        except Exception:
            continue

        # Filter by date if requested
        if filter_date is not None and dt.date() != filter_date:
            continue

        fills.append({
            'dt': dt,
            'time_str': fill_time_raw[:5],  # HH:MM
            'side': side,
            'qty': filled_qty,
            'price': avg_price,
            'account': account,
            'symbol': symbol,
        })

    if not fills:
        return []

    # Group fills by (account, symbol) so MES and MNQ positions are tracked independently
    accounts_fills: dict = _defaultdict(list)
    for f in fills:
        accounts_fills[(f['account'], f['symbol'])].append(f)

    result = []

    for (account_name, _sym_key), acct_fills in accounts_fills.items():
        # Sort chronologically within this account+symbol bucket
        acct_fills.sort(key=lambda x: x['dt'])

        # Group into round-trip trades using net-position tracking
        position = 0
        current_round: list = []
        round_trips: list = []

        for f in acct_fills:
            current_round.append(f)
            if f['side'] == 'Buy':
                position += f['qty']
            else:
                position -= f['qty']

            if position == 0:
                round_trips.append(list(current_round))
                current_round = []

        if current_round:
            round_trips.append(list(current_round))

        for rt in round_trips:
            buys = [f for f in rt if f['side'] == 'Buy']
            sells = [f for f in rt if f['side'] == 'Sell']

            if not (buys and sells):
                # Skip pure directional open positions with no fills on the other side
                continue

            is_long = rt[0]['side'] == 'Buy'

            total_buy_qty = sum(f['qty'] for f in buys)
            total_sell_qty = sum(f['qty'] for f in sells)
            closed_qty = min(total_buy_qty, total_sell_qty)
            is_open = total_buy_qty != total_sell_qty

            if is_long:
                entry_fills, exit_fills = buys, sells
            else:
                entry_fills, exit_fills = sells, buys

            total_entry_qty = sum(f['qty'] for f in entry_fills)
            total_exit_qty = sum(f['qty'] for f in exit_fills)

            avg_entry = (
                sum(f['qty'] * f['price'] for f in entry_fills) / total_entry_qty
                if total_entry_qty else 0.0
            )
            avg_exit = (
                sum(f['qty'] * f['price'] for f in exit_fills) / total_exit_qty
                if total_exit_qty else 0.0
            )

            if is_long:
                pnl_points = (avg_exit - avg_entry) * closed_qty
            else:
                pnl_points = (avg_entry - avg_exit) * closed_qty

            # Dynamic multiplier based on contract symbol
            # Micro contracts must be checked before their full-size counterparts
            _sym = (rt[0].get('symbol') or '').upper()
            if _sym.startswith('MNQ'):   _mult = 2.0    # Micro E-mini NASDAQ
            elif _sym.startswith('MES'): _mult = 5.0    # Micro E-mini S&P 500
            elif _sym.startswith('M2K'): _mult = 5.0    # Micro Russell 2000
            elif _sym.startswith('MGC'): _mult = 10.0   # Micro Gold
            elif _sym.startswith('MCL'): _mult = 100.0  # Micro Crude Oil
            elif _sym.startswith('NQ'):  _mult = 20.0   # E-mini NASDAQ
            elif _sym.startswith('RTY'): _mult = 50.0   # E-mini Russell 2000
            elif _sym.startswith('GC'):  _mult = 100.0  # Gold
            elif _sym.startswith('CL'):  _mult = 1000.0 # Crude Oil
            else:                        _mult = _ES_MULTIPLIER  # ES default $50

            pnl_dollars = round(pnl_points * _mult, 2)

            direction_label = 'Long' if is_long else 'Short'
            _sym_label = rt[0].get('symbol', '')
            note_parts = [
                f"{direction_label} {closed_qty}x {_sym_label}".strip(),
                f"entry {avg_entry:.2f} → exit {avg_exit:.2f}",
            ]
            if is_open:
                note_parts.append("(posizione aperta)")

            # Clean symbol for display: strip exchange suffix (e.g. MESH6.CME -> MESH6)
            _sym_display = _sym_label.split('.')[0] if _sym_label else ''

            result.append({
                'time': rt[0]['time_str'],
                'account': account_name,
                'symbol': _sym_display,
                'imported': True,
                'qty': closed_qty,
                'context': {'trend': 'long' if is_long else 'short'},
                'result': {
                    'executed': 'true',
                    'pnl': '' if is_open else str(pnl_dollars),
                    'notes': ' | '.join(note_parts),
                },
            })

    # Sort final list by time across all accounts
    result.sort(key=lambda x: x.get('time', ''))
    return result


@app.route('/api/checklist/import-apex', methods=['POST'])
@login_required
def api_import_apex():
    """Parse an Overcharts/Apex CSV export and return trade objects.

    Accepts multipart form fields:
      - file: the CSV/TSV file
      - date_key: YYYY-MM-DD — only fills matching this date are imported
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Nome file vuoto'}), 400

    date_key = (request.form.get('date_key') or '').strip()
    import re as _re
    if date_key and not _re.match(r'^\d{4}-\d{2}-\d{2}$', date_key):
        return jsonify({'error': 'date_key non valido'}), 400

    # Accept only plain text / CSV — guard against large uploads
    MAX_SIZE = 2 * 1024 * 1024  # 2 MB
    raw = f.read(MAX_SIZE + 1)
    if len(raw) > MAX_SIZE:
        return jsonify({'error': 'File troppo grande (max 2 MB)'}), 413

    try:
        content = raw.decode('utf-8-sig', errors='replace')
    except Exception:
        return jsonify({'error': 'Impossibile decodificare il file'}), 400

    try:
        trades = _parse_apex_csv(content, date_filter=date_key)
    except Exception as e:
        return jsonify({'error': f'Errore parsing: {e}'}), 500

    return jsonify({'ok': True, 'trades': trades, 'count': len(trades), 'date_filter': date_key})


# ============================================================================
# STOCKS — SEC EDGAR 13F tracker for superinvestor funds
# ============================================================================
#
# Tracks the latest 13F-HR filings for a curated list of "superinvestor"
# funds (concentrated long-only / activist / value-oriented, explicitly NOT
# passive index giants). For each fund, diffs the two most recent 13F-HR
# filings and surfaces:
#   - NEW positions (absent from the previous quarter)
#   - ADDED positions where share count increased >= STOCKS_ADDED_MIN_PCT
#
# Data source: SEC EDGAR (https://www.sec.gov/edgar), the free primary source.
# Results are cached per-fund in MongoDB so each quarterly filing is only
# downloaded and parsed once.

_SUPERINVESTORS_DEFAULT = [
    # (display name, CIK as 10-digit zero-padded string)
    # CIKs verified 2026-04: each returns a recent 13F-HR via EDGAR submissions API.
    ("Berkshire Hathaway (Buffett)",     "0001067983"),
    ("Pershing Square (Ackman)",         "0001336528"),
    ("Scion Asset Mgmt (Burry)",         "0001649339"),  # files inconsistently
    ("Baupost Group (Klarman)",          "0001061768"),
    ("Appaloosa LP (Tepper)",            "0001656456"),
    ("DME Capital Mgmt (Einhorn)",       "0001489933"),  # Greenlight's current 13F filer
    ("Third Point (Loeb)",               "0001040273"),
    ("Harris Associates (Oakmark)",      "0000807985"),
    ("Dodge & Cox",                      "0000200217"),
    ("Tiger Global Mgmt",                "0001167483"),
]

_STOCKS_ADDED_MIN_PCT = 0.20  # share-count increase threshold to flag "ADDED"
_STOCKS_CACHE_TTL_SECONDS = 24 * 60 * 60  # 13F data updates quarterly; 24h cache is plenty
_STOCKS_CACHE_SCHEMA = 3  # bump to invalidate stale cached entries after logic changes
_STOCKS_STALE_DAYS = 120   # filing older than this is flagged "stale" in UI

_MONGO_STOCKS_CACHE_COLLECTION = None


def _edgar_user_agent() -> str:
    # EDGAR requires a descriptive UA; override via env in production.
    return (os.getenv("EDGAR_USER_AGENT") or "ES Gamma Analyzer contact@bitsharp.it").strip()


def _edgar_get(url: str, timeout: int = 10) -> bytes:
    headers = {
        "User-Agent": _edgar_user_agent(),
        "Accept-Encoding": "gzip, deflate",
        "Host": urllib.parse.urlparse(url).netloc,
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            data = gzip.decompress(data)
    return data


def _edgar_fetch_json(url: str) -> Optional[dict]:
    try:
        return json.loads(_edgar_get(url).decode("utf-8"))
    except Exception:
        return None


def _edgar_fetch_text(url: str) -> Optional[str]:
    try:
        return _edgar_get(url).decode("utf-8", errors="replace")
    except Exception:
        return None


def _get_mongo_stocks_cache_collection():
    global _MONGO_CLIENT, _MONGO_STOCKS_CACHE_COLLECTION
    if _MONGO_STOCKS_CACHE_COLLECTION is not None:
        return _MONGO_STOCKS_CACHE_COLLECTION
    if MongoClient is None:
        return None
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_STOCKS_CACHE_COLLECTION") or "stocks_13f_cache").strip()
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]
        try:
            coll.create_index("cik", unique=True)
        except Exception:
            pass
        _MONGO_STOCKS_CACHE_COLLECTION = coll
        return coll
    except Exception:
        return None


def _get_recent_13f_accessions(cik: str, limit: int = 2) -> list:
    """Most recent 13F-HR filings for a CIK, from EDGAR submissions API."""
    cik10 = str(cik).strip().lstrip("0").zfill(10)
    data = _edgar_fetch_json(f"https://data.sec.gov/submissions/CIK{cik10}.json")
    if not data:
        return []
    recent = (data.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    accs = recent.get("accessionNumber") or []
    fdates = recent.get("filingDate") or []
    rdates = recent.get("reportDate") or []
    out = []
    for i, form in enumerate(forms):
        if (form or "").upper() != "13F-HR":
            continue
        out.append({
            "accession_no": accs[i] if i < len(accs) else "",
            "filing_date": fdates[i] if i < len(fdates) else "",
            "report_date": rdates[i] if i < len(rdates) else "",
        })
        if len(out) >= limit:
            break
    return out


def _fetch_13f_info_table(cik: str, accession_no: str) -> list:
    """Parse the information table XML for a filing → list of position dicts."""
    if not accession_no:
        return []
    cik_nolead = str(cik).strip().lstrip("0") or "0"
    acc_nodashes = accession_no.replace("-", "")
    idx = _edgar_fetch_json(
        f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{acc_nodashes}/index.json"
    )
    if not idx:
        return []
    items = ((idx.get("directory") or {}).get("item") or [])
    xml_name = None
    for it in items:
        n = (it.get("name") or "").lower()
        if n.endswith(".xml") and ("infotable" in n or "info_table" in n):
            xml_name = it.get("name")
            break
    if not xml_name:
        for it in items:
            n = (it.get("name") or "").lower()
            if n.endswith(".xml") and "primary_doc" not in n:
                xml_name = it.get("name")
                break
    if not xml_name:
        return []
    xml_text = _edgar_fetch_text(
        f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{acc_nodashes}/{xml_name}"
    )
    if not xml_text:
        return []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
    except Exception:
        return []
    # Strip XML namespaces so findall() works regardless of filing vintage.
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    def _text(parent, path, default=""):
        el = parent.find(path)
        return (el.text or "").strip() if el is not None and el.text is not None else default

    positions = []
    for info in root.findall("infoTable"):
        try:
            value_raw = float(_text(info, "value", "0") or "0")
        except Exception:
            value_raw = 0.0
        try:
            shares = int(float(_text(info, "shrsOrPrnAmt/sshPrnamt", "0") or "0"))
        except Exception:
            shares = 0
        positions.append({
            "issuer": _text(info, "nameOfIssuer"),
            "class": _text(info, "titleOfClass"),
            "cusip": _text(info, "cusip"),
            # Post-2022 SEC rule: value reported in full USD (pre-2022 was $thousands).
            "value_usd": value_raw,
            "shares": shares,
            "share_type": _text(info, "shrsOrPrnAmt/sshPrnamtType", "SH"),
            "put_call": _text(info, "putCall") or None,
        })
    return positions


def _position_key(p: dict) -> tuple:
    return (
        (p.get("cusip") or "").strip().upper(),
        (p.get("class") or "").strip().upper(),
        (p.get("put_call") or "") or "SHARES",
    )


def _aggregate_13f_positions(positions: list) -> list:
    """Merge duplicate rows for the same (cusip, class, put/call).

    A single 13F can list the same security multiple times — once per
    sub-advisor / otherManager (e.g. Berkshire reports APPLE under Buffett,
    Combs, and Weschler separately, and Berkshire's Liberty Live holdings
    are split across Buffett's and Weschler's buckets). The total fund
    position is the sum of shares and value across those rows.
    """
    merged = {}
    for p in positions or []:
        k = _position_key(p)
        if k not in merged:
            merged[k] = dict(p)
            continue
        cur = merged[k]
        cur["shares"] = (cur.get("shares") or 0) + (p.get("shares") or 0)
        cur["value_usd"] = (cur.get("value_usd") or 0.0) + (p.get("value_usd") or 0.0)
        # Preserve issuer/class from the first occurrence; they should match anyway.
    return list(merged.values())


def _diff_13f_positions(latest: list, previous: list, min_pct: float = _STOCKS_ADDED_MIN_PCT) -> dict:
    latest_agg = _aggregate_13f_positions(latest)
    prev_agg = _aggregate_13f_positions(previous)
    prev_map = {_position_key(p): p for p in prev_agg}
    new_positions, added_positions = [], []
    for p in latest_agg:
        k = _position_key(p)
        if k not in prev_map:
            new_positions.append(p)
            continue
        prev_shares = prev_map[k].get("shares") or 0
        cur_shares = p.get("shares") or 0
        if prev_shares <= 0 or cur_shares <= 0:
            continue
        delta_pct = (cur_shares - prev_shares) / float(prev_shares)
        if delta_pct >= min_pct:
            item = dict(p)
            item["prev_shares"] = prev_shares
            item["delta_pct"] = delta_pct
            added_positions.append(item)
    new_positions.sort(key=lambda x: x.get("value_usd") or 0, reverse=True)
    added_positions.sort(key=lambda x: x.get("value_usd") or 0, reverse=True)
    return {"new": new_positions, "added": added_positions}


def _fetch_13f_fund_data(cik: str, name: str) -> dict:
    accs = _get_recent_13f_accessions(cik, limit=2)
    if not accs:
        return {
            "name": name, "cik": cik,
            "error": "Nessun 13F-HR trovato per questo CIK.",
            "new": [], "added": [],
        }
    latest_acc = accs[0]
    latest_pos = _fetch_13f_info_table(cik, latest_acc.get("accession_no"))
    prev_pos = _fetch_13f_info_table(cik, accs[1].get("accession_no")) if len(accs) >= 2 else []
    diff = _diff_13f_positions(latest_pos, prev_pos)
    # Flag filings older than the freshness threshold so the UI can warn.
    stale_days = None
    try:
        fd = _dt.date.fromisoformat(latest_acc.get("filing_date") or "")
        stale_days = (_dt.date.today() - fd).days
    except Exception:
        pass
    return {
        "name": name,
        "cik": cik,
        "filing_date": latest_acc.get("filing_date"),
        "report_date": latest_acc.get("report_date"),
        "accession": latest_acc.get("accession_no"),
        "has_previous": bool(prev_pos),
        "total_positions": len(latest_pos),
        "stale_days": stale_days,
        "stale": (stale_days is not None and stale_days > _STOCKS_STALE_DAYS),
        "new": diff["new"],
        "added": diff["added"],
    }


def _stocks_cached_fund(cik: str, name: str, ttl_seconds: int = _STOCKS_CACHE_TTL_SECONDS) -> dict:
    coll = _get_mongo_stocks_cache_collection()
    now = _dt.datetime.utcnow()
    if coll is not None:
        try:
            cached = coll.find_one({"cik": cik})
        except Exception:
            cached = None
        if cached and cached.get("schema") == _STOCKS_CACHE_SCHEMA:
            fetched_at = cached.get("fetched_at")
            if fetched_at and (now - fetched_at).total_seconds() < ttl_seconds:
                data = dict(cached.get("data") or {})
                data["name"] = name
                data["cached"] = True
                return data
    try:
        data = _fetch_13f_fund_data(cik, name)
    except Exception as e:
        data = {"name": name, "cik": cik, "error": str(e), "new": [], "added": []}
    if coll is not None:
        try:
            coll.update_one(
                {"cik": cik},
                {"$set": {"cik": cik, "data": data, "fetched_at": now, "schema": _STOCKS_CACHE_SCHEMA}},
                upsert=True,
            )
        except Exception:
            pass
    return data


def _get_superinvestors() -> list:
    """Active list of superinvestor funds.

    Override via env: STOCKS_FUNDS_OVERRIDE="Name1:cik1,Name2:cik2,..."
    """
    override = (os.getenv("STOCKS_FUNDS_OVERRIDE") or "").strip()
    if override:
        out = []
        for part in override.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            name, cik = part.split(":", 1)
            out.append((name.strip(), cik.strip().zfill(10)))
        if out:
            return out
    return list(_SUPERINVESTORS_DEFAULT)


@app.route('/stocks')
@login_required
def stocks_page():
    return render_template('stocks.html')


@app.route('/api/stocks/top-buys', methods=['GET'])
@login_required
def api_stocks_top_buys():
    """Latest 13F-HR buys for the curated superinvestor list.

    For each fund returns NEW positions and ADDED positions
    (share count +>= 20%) from the latest filing vs. the prior quarter.
    Funds whose latest 13F-HR is older than _STOCKS_STALE_DAYS are dropped.
    """
    funds = _get_superinvestors()
    results = [None] * len(funds)
    from concurrent.futures import ThreadPoolExecutor

    def _work(i):
        name, cik = funds[i]
        try:
            return i, _stocks_cached_fund(cik, name)
        except Exception as e:
            return i, {"name": name, "cik": cik, "error": str(e), "new": [], "added": []}

    with ThreadPoolExecutor(max_workers=4) as ex:
        for i, data in ex.map(_work, range(len(funds))):
            results[i] = data

    # Drop funds with a stale latest filing (fund may have stopped filing 13Fs).
    fresh = [r for r in results if not r.get("stale")]
    hidden = [r.get("name") for r in results if r.get("stale")]

    return jsonify({
        "funds": fresh,
        "min_added_pct": _STOCKS_ADDED_MIN_PCT,
        "hidden_stale": hidden,
    })


# ============================================================================
# DAMODARAN STOCK SCREENER (Serafini strategy)
# ============================================================================

# Discounts applied to the base P/E theoretical (intercept 13.1).
# Formula: P/E_theo = 13.1 + 1.2 * growth_5y * 100 + country_disc + sector_disc
_SCREENER_COUNTRY_DISCOUNTS = {
    "US": 0, "EU": -5, "IT": -5,
    "CN": -10, "EM": -10, "JP": -3,
}

_SCREENER_SECTOR_DISCOUNTS = {
    "Tech": 0,
    "Industrial": 0,
    "Financial": -5,
    "Energy": -5,
    "Healthcare": -5,
    "RealEstate": -2,
    "Utilities": -2,
    "Comms": -1.5,
    "Lusso": 5,
    "Discretionary": 0,
    "Staples": 0,
    "Materials": 0,
}

# Mapping from yfinance GICS sector strings to our internal bucket.
_SCREENER_GICS_TO_BUCKET = {
    "Technology": "Tech",
    "Communication Services": "Comms",
    "Consumer Cyclical": "Discretionary",
    "Consumer Defensive": "Staples",
    "Financial Services": "Financial",
    "Healthcare": "Healthcare",
    "Industrials": "Industrial",
    "Energy": "Energy",
    "Basic Materials": "Materials",
    "Real Estate": "RealEstate",
    "Utilities": "Utilities",
}

# Top 30 US mega caps by market cap. Used on Vercel where serverless 60s
# timeout requires a tighter universe + parallel fetching.
_SCREENER_US_TOP30_UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "AVGO", "TSLA",
    "BRK-B", "LLY", "WMT", "JPM", "V", "ORCL", "MA", "UNH",
    "XOM", "JNJ", "PG", "HD", "NFLX", "COST", "ABBV", "BAC",
    "KO", "CVX", "AMD", "CRM", "TMUS", "ADBE",
]

# Curated US universe (full): large/mega caps with high analyst coverage,
# plus the tickers explicitly mentioned in the Serafini strategy talks.
# Used in local dev (no timeout, background thread).
_SCREENER_US_UNIVERSE = [
    # Mega/large caps S&P 100
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "JPM",
    "V", "WMT", "UNH", "XOM", "LLY", "JNJ", "MA", "PG", "ORCL", "HD",
    "BAC", "COST", "ABBV", "CVX", "KO", "PEP", "MRK", "NFLX", "CRM", "TMO",
    "AMD", "ADBE", "CSCO", "WFC", "ACN", "DIS", "ABT", "LIN", "DHR", "MCD",
    "INTC", "TXN", "VZ", "CMCSA", "QCOM", "NEE", "NKE", "RTX", "NOW", "HON",
    "PM", "UNP", "T", "UPS", "SBUX", "COP", "BMY", "C", "SCHW", "ELV",
    "MS", "SPGI", "CAT", "DE", "GS", "INTU", "BLK", "MDT", "PFE", "BA",
    "AMAT", "BX", "ISRG", "LMT", "PLD", "SYK", "TJX", "REGN", "GILD", "ADI",
    # Strategy picks from the transcripts
    "ARM", "PLTR", "MU", "STX", "WDC", "MOH", "CNC", "KLAC", "LRCX", "MPWR",
    "GEV", "DELL", "ANET", "CRWD", "MRVL", "KGS",
]

_SCREENER_IS_VERCEL = bool(os.getenv("VERCEL"))

# FTSE MIB top names — large/mid cap with reliable yfinance coverage.
_SCREENER_IT_UNIVERSE = [
    "ISP.MI", "UCG.MI", "ENEL.MI", "ENI.MI", "STLAM.MI", "G.MI", "RACE.MI",
    "LDO.MI", "PRY.MI", "MB.MI", "MONC.MI", "BAMI.MI", "TIT.MI", "TRN.MI",
    "SRG.MI", "CPR.MI", "MAIRE.MI", "DIA.MI", "REC.MI", "AMP.MI", "TEN.MI",
    "NEXI.MI", "FBK.MI", "AZM.MI", "POST.MI", "SPM.MI", "BPE.MI", "INW.MI",
]

# DAX 40 top names.
_SCREENER_DE_UNIVERSE = [
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MUV2.DE", "BAS.DE", "BMW.DE",
    "MBG.DE", "VOW3.DE", "DBK.DE", "ADS.DE", "DB1.DE", "DHL.DE", "HEN3.DE",
    "IFX.DE", "MRK.DE", "RHM.DE", "AIR.DE", "CON.DE", "HEI.DE", "DTG.DE",
    "BEI.DE", "QIA.DE", "VNA.DE", "BAYN.DE", "EOAN.DE", "FRE.DE", "HNR1.DE",
    "PUM.DE", "SY1.DE",
]

# NIFTY 50 top names — NSE listings (`.NS` suffix for yfinance).
_SCREENER_IN_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS", "SBIN.NS", "LT.NS", "HINDUNILVR.NS", "ITC.NS",
    "BAJFINANCE.NS", "KOTAKBANK.NS", "AXISBANK.NS", "M&M.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "HCLTECH.NS", "NTPC.NS", "TITAN.NS", "ULTRACEMCO.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "ASIANPAINT.NS", "BAJAJFINSV.NS", "BEL.NS",
    "NESTLEIND.NS", "ONGC.NS", "POWERGRID.NS", "TATAMOTORS.NS", "TATASTEEL.NS",
    "WIPRO.NS", "JSWSTEEL.NS", "COALINDIA.NS", "GRASIM.NS", "TECHM.NS",
    "HDFCLIFE.NS", "SBILIFE.NS", "BAJAJ-AUTO.NS", "BRITANNIA.NS", "EICHERMOT.NS",
    "INDUSINDBK.NS", "CIPLA.NS", "DRREDDY.NS", "APOLLOHOSP.NS", "HEROMOTOCO.NS",
    "TRENT.NS", "SHRIRAMFIN.NS", "TATACONSUM.NS", "HINDALCO.NS",
]

# Maps a screener market code to the country bucket used by the Damodaran
# discount table (controls country_disc).
_SCREENER_MARKET_TO_COUNTRY = {
    "US": "US",
    "IT": "IT",
    "DE": "EU",
    "IN": "EM",
}

_SCREENER_VALID_MARKETS = ("US", "IT", "DE", "IN")

# Map yfinance info["country"] (full country names) to our bucket codes.
# Used by the on-demand lookup endpoint where the user can enter any ticker.
_SCREENER_COUNTRY_TO_BUCKET = {
    "United States": "US",
    "Italy": "IT",
    "Germany": "EU", "France": "EU", "Spain": "EU", "Netherlands": "EU",
    "Switzerland": "EU", "United Kingdom": "EU", "Ireland": "EU",
    "Sweden": "EU", "Denmark": "EU", "Finland": "EU", "Belgium": "EU",
    "Austria": "EU", "Norway": "EU", "Portugal": "EU", "Luxembourg": "EU",
    "Japan": "JP",
    "China": "CN", "Hong Kong": "CN", "Taiwan": "CN",
    "India": "EM", "Brazil": "EM", "Mexico": "EM", "South Africa": "EM",
    "Russia": "EM", "Turkey": "EM", "Indonesia": "EM",
}


def _map_country_to_bucket(country_name: Optional[str]) -> str:
    return _SCREENER_COUNTRY_TO_BUCKET.get((country_name or "").strip(), "US")


def _screener_universe_for(market: str) -> list:
    """Universe for a given market code. US always uses the full ~95-ticker
    universe — Vercel can handle it within the 60s budget thanks to FMP
    Starter (no rate-limit fallbacks) + 8 parallel workers."""
    if market == "US":
        return _SCREENER_US_UNIVERSE
    if market == "IT":
        return _SCREENER_IT_UNIVERSE
    if market == "DE":
        return _SCREENER_DE_UNIVERSE
    if market == "IN":
        return _SCREENER_IN_UNIVERSE
    return []


def _screener_active_universe() -> list:
    """Backward-compatible wrapper (US-only callers)."""
    return _screener_universe_for("US")


# Per-market in-memory cache. Each market gets its own slot.
_SCREENER_CACHE: Dict[str, dict] = {}
_SCREENER_CACHE_TTL_SECONDS = int((os.getenv("SCREENER_CACHE_TTL") or "43200").strip() or 43200)  # 12h
_SCREENER_REFRESH_LOCKS: Dict[str, "threading.Lock"] = {}
_MONGO_SCREENER_COLLECTION = None
_MONGO_PORTFOLIO_COLLECTION = None


def _get_market_cache(market: str) -> dict:
    if market not in _SCREENER_CACHE:
        _SCREENER_CACHE[market] = {
            "results": [],
            "computed_at": 0.0,
            "errors": [],
            "in_progress": False,
            "loaded_from_mongo": False,
        }
    return _SCREENER_CACHE[market]


def _get_market_lock(market: str) -> "threading.Lock":
    if market not in _SCREENER_REFRESH_LOCKS:
        _SCREENER_REFRESH_LOCKS[market] = threading.Lock()
    return _SCREENER_REFRESH_LOCKS[market]


def _get_mongo_screener_collection():
    """Lazy getter for the screener results collection (per-ticker upsert)."""
    global _MONGO_CLIENT, _MONGO_SCREENER_COLLECTION
    if _MONGO_SCREENER_COLLECTION is not None:
        return _MONGO_SCREENER_COLLECTION
    if MongoClient is None:
        return None
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_SCREENER_COLLECTION") or "screener_results").strip()
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]
        try:
            coll.create_index("ticker", unique=True)
            coll.create_index([("market", 1), ("ratio_discount_vola", -1)])
        except Exception:
            pass
        _MONGO_SCREENER_COLLECTION = coll
        return coll
    except Exception:
        return None


def _get_mongo_portfolio_collection():
    """Lazy getter for the per-user portfolio collection.
    Each document = one ticker added by one user. Composite unique index
    on (user_key, ticker) prevents duplicates.
    """
    global _MONGO_CLIENT, _MONGO_PORTFOLIO_COLLECTION
    if _MONGO_PORTFOLIO_COLLECTION is not None:
        return _MONGO_PORTFOLIO_COLLECTION
    if MongoClient is None:
        return None
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_PORTFOLIO_COLLECTION") or "user_portfolio").strip()
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]
        try:
            coll.create_index([("user_key", 1), ("ticker", 1)], unique=True)
            coll.create_index("user_key")
        except Exception:
            pass
        _MONGO_PORTFOLIO_COLLECTION = coll
        return coll
    except Exception:
        return None


def _calculate_damodaran_target(
    avg_growth: float,
    forward_eps: float,
    current_price: float,
    country: str = "US",
    bucket: str = "Tech",
    dev_st_pct: Optional[float] = None,
) -> dict:
    """Pure Damodaran/Serafini calculation (matches the Excel formula).

    avg_growth: decimal (0.20 = 20%); typically the 5y analyst CAGR.
    forward_eps: EPS estimate for next fiscal year.
    """
    country_disc = _SCREENER_COUNTRY_DISCOUNTS.get(country, 0)
    sector_disc = _SCREENER_SECTOR_DISCOUNTS.get(bucket, 0)
    pe_theo = 13.1 + 1.2 * avg_growth * 100 + country_disc + sector_disc
    target = pe_theo * forward_eps
    discount = (target - current_price) / current_price if current_price else 0.0
    ratio = (discount / (dev_st_pct / 100.0)) if dev_st_pct else None
    return {
        "country_disc": country_disc,
        "sector_disc": sector_disc,
        "pe_theoretical": pe_theo,
        "target_y1": target,
        "discount_pct": discount,
        "ratio_discount_vola": ratio,
        "verdict": "UNDERVALUED" if discount > 0 else "OVERVALUED",
    }


_FMP_BASE_URL = "https://financialmodelingprep.com/stable"
_FMP_TIMEOUT_SECONDS = 8


def _fmp_get(path: str, **params) -> Optional[Any]:
    """GET wrapper for FMP `stable` API. Returns parsed JSON, or None on any
    failure (missing key, network error, non-200, error payload). Never raises."""
    api_key = (os.getenv("FMP_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        params["apikey"] = api_key
        qs = urllib.parse.urlencode(params)
        url = f"{_FMP_BASE_URL}/{path}?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "es-gamma-analyzer/1.0"})
        with urllib.request.urlopen(req, timeout=_FMP_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict) and (data.get("Error Message") or data.get("error")):
            return None
        return data
    except Exception:
        return None


def _fetch_ticker_fundamentals_fmp(ticker: str) -> Optional[dict]:
    """Fetch fundamentals for a single ticker via FinancialModelingPrep
    `stable` API. Returns the same dict shape as _fetch_ticker_fundamentals_yf,
    or None when required fields are missing or FMP is unavailable.
    """
    profile_data = _fmp_get("profile", symbol=ticker)
    if not profile_data or not isinstance(profile_data, list) or not profile_data:
        return None
    p = profile_data[0]

    current_price = p.get("price")
    market_cap = p.get("marketCap")
    beta = p.get("beta")
    sector = p.get("sector") or ""
    country_iso = p.get("country") or ""
    long_name = p.get("companyName") or ticker

    if not current_price or current_price <= 0:
        return None

    # Forward EPS + 5y CAGR from analyst estimates (consensus mean per fiscal year).
    estimates = _fmp_get("analyst-estimates", symbol=ticker, period="annual", limit=10)
    if not estimates or not isinstance(estimates, list):
        return None
    today_str = _dt.date.today().isoformat()
    future_eps = []
    for est in estimates:
        d = (est.get("date") or "")[:10]
        eps = est.get("epsAvg")
        if d > today_str and eps is not None and eps > 0:
            future_eps.append((d, float(eps)))
    if not future_eps:
        return None
    future_eps.sort()
    forward_eps = future_eps[0][1]

    # CAGR from year+1 to the furthest available forecast (cap at 5y span).
    growth = None
    if len(future_eps) >= 2:
        idx = min(len(future_eps) - 1, 4)
        last_eps = future_eps[idx][1]
        n = idx
        if last_eps > 0 and forward_eps > 0 and n > 0:
            growth = (last_eps / forward_eps) ** (1.0 / n) - 1.0
    if growth is None:
        return None

    # 1y annualized stdev of daily returns (volatility for ratio).
    dev_st_pct = None
    try:
        hist = _fmp_get("historical-price-eod/light", symbol=ticker)
        if hist and isinstance(hist, list):
            prices = [h.get("price") for h in hist if h.get("price")]
            if len(prices) > 30:
                import math
                prices = prices[:260]
                prices.reverse()  # FMP returns newest first
                rets = []
                for i in range(1, len(prices)):
                    if prices[i - 1] > 0:
                        rets.append((prices[i] - prices[i - 1]) / prices[i - 1])
                if len(rets) > 20:
                    mean = sum(rets) / len(rets)
                    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
                    dev_st_pct = float((var ** 0.5) * math.sqrt(252) * 100)
    except Exception:
        dev_st_pct = None

    bucket = _SCREENER_GICS_TO_BUCKET.get(sector, "Tech")

    return {
        "ticker": ticker,
        "name": long_name,
        "yf_sector": sector,
        "bucket": bucket,
        "country_iso": country_iso,
        "country": "US",
        "market_cap": float(market_cap) if market_cap else None,
        "beta": float(beta) if beta is not None else None,
        "current_price": float(current_price),
        "forward_eps": float(forward_eps),
        "growth_5y": float(growth),
        "dev_st_pct": dev_st_pct,
        "_source": "fmp",
    }


def _fetch_ticker_fundamentals(ticker: str) -> Optional[dict]:
    """Fundamentals for a ticker — FMP first, yfinance fallback.

    Returns None when neither source has the required fields (forward EPS,
    price, growth)."""
    fmp_res = _fetch_ticker_fundamentals_fmp(ticker)
    if fmp_res is not None:
        return fmp_res
    return _fetch_ticker_fundamentals_yf(ticker)


def _fetch_ticker_fundamentals_yf(ticker: str) -> Optional[dict]:
    """Fetch fundamentals for a single ticker via yfinance.

    Returns None when required fields (forward EPS, price, growth) are missing.
    """
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        try:
            info = t.info or {}
        except Exception:
            info = {}

        forward_eps = info.get("forwardEps")
        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        yf_sector = info.get("sector") or ""
        country_iso = info.get("country") or ""
        market_cap = info.get("marketCap")
        beta = info.get("beta")
        long_name = info.get("shortName") or info.get("longName") or ticker

        if not forward_eps or forward_eps <= 0:
            return None
        if not current_price or current_price <= 0:
            return None

        # Forward EPS growth estimate. Source priority (analyst consensus, all forward):
        #   1. LTG (Long Term Growth, ~5y CAGR) from growth_estimates
        #   2. +1y (next fiscal year) from growth_estimates
        #   3. computed (forwardEps - trailingEps) / trailingEps
        # We deliberately avoid info["earningsGrowth"] because it is TRAILING.
        growth = None

        def _read_growth_row(ge_df, row_label):
            """Read the per-stock growth value for a row of growth_estimates.
            Only stockTrend / stock_trend / growth columns are valid:
            indexTrend would return the market index growth (e.g. 12% for S&P).
            """
            try:
                if ge_df is None or ge_df.empty or row_label not in ge_df.index:
                    return None
                row = ge_df.loc[row_label]
                for col in ("stockTrend", "stock_trend", "growth"):
                    try:
                        v = row[col]
                        if v is not None and v == v:  # NaN check
                            return float(v)
                    except (KeyError, TypeError):
                        continue
            except Exception:
                pass
            return None

        try:
            ge = t.growth_estimates
            for label in ("LTG", "+5y", "5y", "longTerm", "+1y", "1y"):
                growth = _read_growth_row(ge, label)
                if growth is not None:
                    break
        except Exception:
            growth = None

        if growth is None:
            trailing_eps = info.get("trailingEps")
            if trailing_eps and trailing_eps > 0 and forward_eps > 0:
                growth = (forward_eps - trailing_eps) / trailing_eps

        if growth is None:
            return None

        # Annualized stdev of daily returns over the last year (volatility for ratio).
        dev_st_pct = None
        try:
            hist = t.history(period="1y", auto_adjust=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                returns = hist["Close"].pct_change().dropna()
                if len(returns) > 20:
                    import math
                    dev_st_pct = float(returns.std() * math.sqrt(252) * 100)
        except Exception:
            dev_st_pct = None

        bucket = _SCREENER_GICS_TO_BUCKET.get(yf_sector, "Tech")

        return {
            "ticker": ticker,
            "name": long_name,
            "yf_sector": yf_sector,
            "bucket": bucket,
            "country_iso": country_iso,
            "country": "US",
            "market_cap": float(market_cap) if market_cap else None,
            "beta": float(beta) if beta is not None else None,
            "current_price": float(current_price),
            "forward_eps": float(forward_eps),
            "growth_5y": float(growth),
            "dev_st_pct": dev_st_pct,
            "_source": "yf",
        }
    except Exception:
        return None


def _stock_strategy_check(r: dict) -> dict:
    """Strategy filters from the Serafini course + sanity guards on yfinance
    data quality. Returns {passes: bool, reasons: [str]} so the UI can show
    the user WHY a ticker is excluded.
    """
    reasons = []

    fwd_eps = r.get("forward_eps")
    if not fwd_eps or fwd_eps <= 0:
        reasons.append("EPS forward non positivo")
    elif fwd_eps > 60:
        reasons.append(f"EPS forward {fwd_eps:.2f} > 60 (probabile dato stale o split non gestito)")

    g = r.get("growth_5y")
    if g is None:
        reasons.append("Growth attesa non disponibile")
    elif g <= 0.05:
        reasons.append(f"Growth {g*100:+.1f}% ≤ 5% (insufficiente per il modello)")
    elif g >= 0.60:
        reasons.append(f"Growth {g*100:.1f}% ≥ 60% (irrealistica, di solito artefatto da trailing-recovery)")

    mc = r.get("market_cap")
    if mc is None:
        reasons.append("Market cap non disponibile")
    elif mc < 2_000_000_000:
        reasons.append(f"Market cap {mc/1e9:.2f}B < 2B (troppo piccola per la strategia)")

    d = r.get("discount_pct")
    if d is None:
        reasons.append("Discount non calcolabile")
    elif d <= 0:
        reasons.append(f"Sovracuotata vs target ({d*100:+.1f}%)")
    elif d >= 3.0:
        reasons.append(f"Discount {d*100:.0f}% ≥ 300% (probabile dato corrotto)")

    pe = r.get("pe_theoretical")
    if pe is None:
        reasons.append("P/E teorico non calcolabile")
    elif pe >= 100:
        reasons.append(f"P/E teorico {pe:.0f} ≥ 100 (combinazione growth+sconti irrealistica)")

    return {"passes": len(reasons) == 0, "reasons": reasons}


def _stock_passes_strategy(r: dict) -> bool:
    return _stock_strategy_check(r)["passes"]


def _refresh_screener_results(
    universe: Optional[list] = None,
    market: str = "US",
    max_workers: int = 1,
) -> dict:
    """Run the screener over a market's universe. Per-market cache + lock.
    Applies the correct country bucket (US/IT/EU) to override yfinance's
    raw country detection. Persists per-ticker rows to Mongo (with market tag).
    """
    cache = _get_market_cache(market)
    lock = _get_market_lock(market)
    if not lock.acquire(blocking=False):
        return {"status": "already_running"}
    try:
        cache["in_progress"] = True
        if universe is None:
            universe = _screener_universe_for(market)
        country_for_market = _SCREENER_MARKET_TO_COUNTRY.get(market, "US")
        coll = _get_mongo_screener_collection()

        def _process(ticker: str) -> Optional[dict]:
            try:
                fund = _fetch_ticker_fundamentals(ticker)
                if not fund:
                    return None
                # Apply the screener-context country (overrides default "US"
                # in the fetcher). Damodaran sconto_paese depends on this.
                fund["country"] = country_for_market
                calc = _calculate_damodaran_target(
                    avg_growth=fund["growth_5y"],
                    forward_eps=fund["forward_eps"],
                    current_price=fund["current_price"],
                    country=fund["country"],
                    bucket=fund["bucket"],
                    dev_st_pct=fund["dev_st_pct"],
                )
                row = {**fund, **calc, "market": market, "computed_at": time.time()}
                if coll is not None:
                    try:
                        coll.update_one(
                            {"ticker": ticker, "market": market},
                            {"$set": row},
                            upsert=True,
                        )
                    except Exception:
                        pass
                return row
            except Exception:
                return None

        results = []
        errors = []
        if max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for ticker, row in zip(universe, ex.map(_process, universe)):
                    if row is None:
                        errors.append(ticker)
                    else:
                        results.append(row)
        else:
            for ticker in universe:
                row = _process(ticker)
                if row is None:
                    errors.append(ticker)
                else:
                    results.append(row)

        cache["results"] = results
        cache["computed_at"] = time.time()
        cache["errors"] = errors
        return {"status": "ok", "market": market, "count": len(results), "errors_count": len(errors)}
    finally:
        cache["in_progress"] = False
        try:
            lock.release()
        except Exception:
            pass


def _ensure_screener_cache_fresh(
    market: str = "US",
    max_age_seconds: Optional[int] = None,
) -> bool:
    """Returns True if the cache for `market` is fresh.

    On Vercel (no persistent threads): SYNCHRONOUSLY refreshes in-request
    with the curated universe + 8 parallel workers (fits in 60s timeout).
    On local: spawns a background thread.

    Both modes seed the cache from Mongo (filtered by market) on cold start.
    """
    max_age = max_age_seconds if max_age_seconds is not None else _SCREENER_CACHE_TTL_SECONDS
    cache = _get_market_cache(market)
    if cache.get("in_progress"):
        return False
    age = time.time() - (cache.get("computed_at") or 0)
    has_results = bool(cache.get("results"))

    # Seed from Mongo on cold start
    if not has_results and not cache.get("loaded_from_mongo"):
        try:
            coll = _get_mongo_screener_collection()
            if coll is not None:
                docs = list(coll.find({"market": market}))
                if docs:
                    rows = []
                    for d in docs:
                        row = {k: v for k, v in d.items() if k != "_id"}
                        # Pre-FMP rows lack `_source`. They were all computed
                        # via yfinance — backfill so the UI pill renders correctly.
                        row.setdefault("_source", "yf")
                        rows.append(row)
                    cache["results"] = rows
                    cache["computed_at"] = min(
                        (d.get("computed_at") or 0) for d in docs
                    )
                    age = time.time() - cache["computed_at"]
                    has_results = True
        except Exception:
            pass
        cache["loaded_from_mongo"] = True

    if has_results and age <= max_age:
        return True

    # Stale or empty: refresh now.
    if _SCREENER_IS_VERCEL:
        try:
            _refresh_screener_results(
                universe=_screener_universe_for(market),
                market=market,
                max_workers=8,
            )
            return True
        except Exception:
            return False
    else:
        try:
            t = threading.Thread(
                target=_refresh_screener_results,
                kwargs={"market": market},
                name=f"screener-refresh-{market}",
                daemon=True,
            )
            t.start()
        except Exception:
            pass
        return False


@app.route('/screener')
@login_required
def screener_page():
    return render_template('screener.html')


@app.route('/api/screener/top', methods=['GET'])
@login_required
def api_screener_top():
    """Return the top N stocks for a market (US|IT|DE) that pass the Serafini
    strategy filters, ranked by ratio_discount_vola (fallback discount_pct).
    """
    try:
        limit = int(request.args.get('limit') or 5)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(50, limit))

    market = (request.args.get('market') or 'US').strip().upper()
    if market not in _SCREENER_VALID_MARKETS:
        return jsonify({"error": f"invalid market '{market}'", "valid": list(_SCREENER_VALID_MARKETS)}), 400

    fresh = _ensure_screener_cache_fresh(market=market)
    cache = _get_market_cache(market)
    results = list(cache.get("results") or [])
    qualified = [r for r in results if _stock_passes_strategy(r)]

    def _sort_key(r):
        ratio = r.get("ratio_discount_vola")
        if ratio is None:
            return r.get("discount_pct") or -999
        return ratio

    qualified.sort(key=_sort_key, reverse=True)
    top = qualified[:limit]

    return jsonify({
        "market": market,
        "computed_at": cache.get("computed_at"),
        "in_progress": cache.get("in_progress", False),
        "is_fresh": fresh,
        "ttl_seconds": _SCREENER_CACHE_TTL_SECONDS,
        "universe_size": len(_screener_universe_for(market)),
        "runtime": "vercel" if _SCREENER_IS_VERCEL else "local",
        "evaluated_count": len(results),
        "qualified_count": len(qualified),
        "errors_count": len(cache.get("errors") or []),
        "top": top,
    })


@app.route('/api/screener/search', methods=['GET'])
@login_required
def api_screener_search():
    """Typeahead search proxy to Yahoo Finance.
    Returns up to 10 matches: [{symbol, name, exchange, type}, ...].
    Always 200 even on upstream failure so the typeahead doesn't break the UI.
    """
    q = (request.args.get('q') or '').strip()
    if not q or len(q) > 50:
        return jsonify({"results": []})
    try:
        url = (
            "https://query2.finance.yahoo.com/v1/finance/search?"
            + urllib.parse.urlencode({"q": q, "quotesCount": 10, "newsCount": 0})
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status != 200:
                return jsonify({"results": []})
            data = json.loads(response.read().decode("utf-8")) or {}
        out = []
        for quote in (data.get("quotes") or []):
            sym = quote.get("symbol")
            if not sym:
                continue
            qtype = (quote.get("quoteType") or "").lower()
            # Only equities/ETFs are relevant for this screener
            if qtype not in ("equity", "etf", ""):
                continue
            out.append({
                "symbol": sym,
                "name": quote.get("longname") or quote.get("shortname") or "",
                "exchange": quote.get("exchDisp") or quote.get("exchange") or "",
                "type": quote.get("typeDisp") or quote.get("quoteType") or "",
            })
        return jsonify({"results": out})
    except Exception:
        return jsonify({"results": []})


@app.route('/api/screener/lookup/<ticker>', methods=['GET'])
@login_required
def api_screener_lookup(ticker):
    """On-demand single-ticker analysis. Always recomputed (no cache),
    so the user sees the live yfinance data for any symbol they search.

    Country is auto-detected from yfinance metadata (US, IT, EU, JP, CN, EM).
    Returns the same shape as a screener row plus 'passes_strategy'.
    """
    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm or not all(c.isalnum() or c in ".-" for c in ticker_norm) or len(ticker_norm) > 12:
        return jsonify({"error": "invalid ticker", "ticker": ticker_norm}), 400

    fund = _fetch_ticker_fundamentals(ticker_norm)
    if not fund:
        return jsonify({
            "error": "ticker not found or missing data (forward EPS / growth)",
            "ticker": ticker_norm,
        }), 404

    # Override the hardcoded "US" with the auto-detected bucket from yfinance metadata.
    detected_country = _map_country_to_bucket(fund.get("country_iso"))
    fund["country"] = detected_country

    calc = _calculate_damodaran_target(
        avg_growth=fund["growth_5y"],
        forward_eps=fund["forward_eps"],
        current_price=fund["current_price"],
        country=fund["country"],
        bucket=fund["bucket"],
        dev_st_pct=fund["dev_st_pct"],
    )
    row = {**fund, **calc, "market": detected_country}
    check = _stock_strategy_check(row)
    row["passes_strategy"] = check["passes"]
    row["strategy_reasons"] = check["reasons"]
    return jsonify(row)


@app.route('/api/screener/refresh', methods=['POST'])
@login_required
def api_screener_refresh():
    """Manual refresh trigger (admin only).
    On Vercel: runs SYNCHRONOUSLY in-request with parallel workers (background
    threads die with the serverless function, leaving in_progress stuck True).
    On local: spawns a background thread.
    Optional ?market=US|IT|DE|IN (default US).
    """
    if not _is_admin():
        return jsonify({"error": "admin only"}), 403
    market = (request.args.get('market') or 'US').strip().upper()
    if market not in _SCREENER_VALID_MARKETS:
        return jsonify({"error": f"invalid market '{market}'"}), 400
    cache = _get_market_cache(market)
    # Allow forcing a refresh even if a previous (likely orphaned) thread
    # left in_progress=True — the lock inside _refresh_screener_results is
    # the real guard against concurrent refreshes.
    force = (request.args.get('force') or '').lower() in ('1', 'true', 'yes')
    if cache.get("in_progress") and not force:
        return jsonify({"status": "already_running", "market": market})
    if _SCREENER_IS_VERCEL:
        try:
            res = _refresh_screener_results(
                universe=_screener_universe_for(market),
                market=market,
                max_workers=8,
            )
            return jsonify({"status": "completed", "market": market, **res})
        except Exception as e:
            cache["in_progress"] = False
            return jsonify({"error": str(e)}), 500
    try:
        t = threading.Thread(
            target=_refresh_screener_results,
            kwargs={"market": market},
            name=f"screener-refresh-manual-{market}",
            daemon=True,
        )
        t.start()
        return jsonify({"status": "started", "market": market})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# USER PORTFOLIO (per-user holdings + Damodaran analysis on demand)
# ============================================================================


def _analyze_portfolio_ticker(ticker: str, added_at: Optional[float]) -> dict:
    """Single-ticker analysis used by the portfolio endpoint.
    Mirrors the lookup endpoint logic but returns a row dict (no JSON).
    Country is auto-detected from yfinance metadata.
    """
    try:
        fund = _fetch_ticker_fundamentals(ticker)
        if not fund:
            return {
                "ticker": ticker,
                "added_at": added_at,
                "error": "data not available (forward EPS or growth missing)",
            }
        detected_country = _map_country_to_bucket(fund.get("country_iso"))
        fund["country"] = detected_country
        calc = _calculate_damodaran_target(
            avg_growth=fund["growth_5y"],
            forward_eps=fund["forward_eps"],
            current_price=fund["current_price"],
            country=fund["country"],
            bucket=fund["bucket"],
            dev_st_pct=fund["dev_st_pct"],
        )
        row = {**fund, **calc, "market": detected_country, "added_at": added_at}
        check = _stock_strategy_check(row)
        row["passes_strategy"] = check["passes"]
        row["strategy_reasons"] = check["reasons"]
        return row
    except Exception as e:
        return {"ticker": ticker, "added_at": added_at, "error": str(e)}


def _portfolio_aggregate_exposure(holdings: list) -> dict:
    """Aggregate exposure by sector and country.
    Equal-weight (one share = one vote) since we don't track quantities.
    Returns dicts for chart rendering.
    """
    by_sector = {}
    by_country = {}
    by_zone = {"Affare": 0, "Sconto": 0, "Equa": 0, "Cara": 0, "N/D": 0}
    valid = 0
    for h in holdings:
        if h.get("error"):
            continue
        valid += 1
        sector = h.get("bucket") or "N/D"
        country = h.get("country") or "N/D"
        by_sector[sector] = by_sector.get(sector, 0) + 1
        by_country[country] = by_country.get(country, 0) + 1
        # Zone classification (mirrors frontend computeZone)
        pe_now = (h["current_price"] / h["forward_eps"]) if h.get("forward_eps") else None
        pe_theo = h.get("pe_theoretical")
        if pe_now and pe_theo and pe_theo > 0:
            ratio = pe_now / pe_theo
            if ratio <= 0.35:
                by_zone["Affare"] += 1
            elif ratio <= 0.55:
                by_zone["Sconto"] += 1
            elif ratio <= 0.85:
                by_zone["Equa"] += 1
            else:
                by_zone["Cara"] += 1
        else:
            by_zone["N/D"] += 1
    return {
        "by_sector": by_sector,
        "by_country": by_country,
        "by_zone": by_zone,
        "valid_count": valid,
        "total_count": len(holdings),
    }


@app.route('/portfolio')
@login_required
def portfolio_page():
    return render_template('portfolio.html')


@app.route('/api/portfolio', methods=['GET'])
@login_required
def api_portfolio_get():
    """Get the authenticated user's portfolio with live Damodaran analysis
    for each holding. Computed in parallel (~5s for 10 tickers on Vercel)."""
    user_key = _current_user_key()
    if not user_key:
        return jsonify({"error": "no user"}), 401
    coll = _get_mongo_portfolio_collection()
    if coll is None:
        return jsonify({"holdings": [], "exposure": {}, "error": "mongo unavailable"}), 200
    try:
        docs = list(coll.find({"user_key": user_key}).sort("added_at", -1))
    except Exception:
        docs = []
    if not docs:
        return jsonify({"holdings": [], "exposure": _portfolio_aggregate_exposure([])})

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        holdings = list(ex.map(
            lambda d: _analyze_portfolio_ticker(d["ticker"], d.get("added_at")),
            docs,
        ))

    return jsonify({
        "holdings": holdings,
        "exposure": _portfolio_aggregate_exposure(holdings),
    })


@app.route('/api/portfolio', methods=['POST'])
@login_required
def api_portfolio_add():
    """Add a ticker to the user's portfolio. Body: {"ticker": "NVDA"}.
    Validates the ticker exists via yfinance before persisting."""
    user_key = _current_user_key()
    if not user_key:
        return jsonify({"error": "no user"}), 401
    data = request.get_json(silent=True) or {}
    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker or not all(c.isalnum() or c in ".-" for c in ticker) or len(ticker) > 12:
        return jsonify({"error": "invalid ticker"}), 400
    coll = _get_mongo_portfolio_collection()
    if coll is None:
        return jsonify({"error": "mongo unavailable"}), 503
    try:
        coll.update_one(
            {"user_key": user_key, "ticker": ticker},
            {"$set": {"user_key": user_key, "ticker": ticker, "added_at": time.time()}},
            upsert=True,
        )
        return jsonify({"status": "added", "ticker": ticker})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/portfolio/<ticker>', methods=['DELETE'])
@login_required
def api_portfolio_remove(ticker):
    """Remove a ticker from the user's portfolio."""
    user_key = _current_user_key()
    if not user_key:
        return jsonify({"error": "no user"}), 401
    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm:
        return jsonify({"error": "invalid ticker"}), 400
    coll = _get_mongo_portfolio_collection()
    if coll is None:
        return jsonify({"error": "mongo unavailable"}), 503
    try:
        result = coll.delete_one({"user_key": user_key, "ticker": ticker_norm})
        return jsonify({
            "status": "removed" if result.deleted_count else "not_found",
            "ticker": ticker_norm,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================


if __name__ == '__main__':
    port_env = os.getenv('PORT')
    try:
        port = int(port_env) if port_env else 5005
    except ValueError:
        port = 5005
    debug_env = (os.getenv('FLASK_DEBUG') or os.getenv('DEBUG') or '').strip().lower()
    # Default to debug=True (developer-friendly), but allow disabling for stable background runs.
    debug = False if debug_env in ('0', 'false', 'no') else True
    # If debug is enabled, keep the reloader only when explicitly allowed.
    reloader_env = (os.getenv('FLASK_USE_RELOADER') or '').strip().lower()
    use_reloader = True if (debug and reloader_env in ('1', 'true', 'yes')) else False

    # Best-effort background capture so conversions are stored even without an open browser.
    # Disabled on Vercel/serverless.
    enable_capture = (os.getenv('ENABLE_CONVERSION_CAPTURE_THREAD') or '1').strip().lower() not in ('0', 'false', 'no')
    if enable_capture and not os.getenv('VERCEL'):
        def _capture_loop():
            while True:
                try:
                    now_dt = _dt.datetime.now()
                    h, m = now_dt.hour, now_dt.minute
                    in_1430 = (h == 14 and 30 <= m < 35)
                    in_close = (h == 16 and m < 5)
                    if in_1430 or in_close:
                        snap = get_spx_snapshot_cached(metric='hybrid', max_age_seconds=0) or None
                        if snap:
                            _maybe_capture_es_spx_conversion(snap, now_dt=now_dt)
                except Exception:
                    pass
                time.sleep(20)

        try:
            t = threading.Thread(target=_capture_loop, name='conv-capture', daemon=True)
            t.start()
        except Exception:
            pass

    app.run(debug=debug, use_reloader=use_reloader, port=port)
