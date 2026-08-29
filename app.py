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
import html as _html
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, List, Optional
import datetime as _dt
import tempfile
import pdfplumber
import pandas as pd
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import re
import hmac
import base64
import hashlib
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


_CHANGELOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CHANGELOG.md")
# Matches "## [1.2.3] — 2026-05-13" with hyphen, em-dash or en-dash.
_CHANGELOG_HEADING_RE = re.compile(
    r"^##\s*\[(?P<version>\d+\.\d+\.\d+)\]\s*[—–-]\s*(?P<date>\d{4}-\d{2}-\d{2})",
    re.MULTILINE,
)


def _read_changelog_text() -> str:
    try:
        with open(_CHANGELOG_PATH, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception:
        return ""


def _parse_latest_release(text: str) -> dict:
    """Returns {version, date} for the topmost release in the changelog,
    or {} if no version heading is found."""
    if not text:
        return {}
    m = _CHANGELOG_HEADING_RE.search(text)
    if not m:
        return {}
    return {"version": m.group("version"), "date": m.group("date")}


def _compute_build_info() -> dict:
    """Identifier shown in the header. Reads CHANGELOG.md as the single source
    of truth for the user-visible version + release date. Falls back to the
    file mtime if the changelog isn't parseable for any reason.
    """
    text = _read_changelog_text()
    rel = _parse_latest_release(text)
    version = rel.get("version")
    date = rel.get("date")

    if not date and _APP_BUILD:
        date = _dt.datetime.utcfromtimestamp(_APP_BUILD).strftime("%Y-%m-%d")

    if version and date:
        label = f"v{version} · {date}"
    elif version:
        label = f"v{version}"
    elif date:
        label = f"build {date}"
    else:
        label = "build —"

    return {
        "version": version or "—",
        "date": date or "",
        "label": label,
        "has_notes": bool(text),
    }


def _render_changelog_html(text: str) -> str:
    """Tiny Markdown → HTML converter for our changelog. Handles only the
    constructs we use: H1–H4, bullet lists, **bold**, `code`, paragraphs,
    blank lines. Output is wrapped in <div class="rn-content">.
    Input is HTML-escaped first so no Markdown can produce raw HTML."""
    if not text:
        return '<div class="rn-content"><p>Note di rilascio non disponibili.</p></div>'

    def esc(s):
        return (s.replace("&", "&amp;").replace("<", "&lt;")
                 .replace(">", "&gt;").replace('"', "&quot;"))

    _inline_bold = re.compile(r"\*\*(.+?)\*\*")
    _inline_code = re.compile(r"`([^`]+)`")
    _inline_link = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

    def inline(s):
        s = _inline_code.sub(lambda m: f"<code>{m.group(1)}</code>", s)
        s = _inline_bold.sub(lambda m: f"<strong>{m.group(1)}</strong>", s)
        # Links: only allow http(s) and relative URLs
        def _link_repl(m):
            label, href = m.group(1), m.group(2)
            if href.startswith(("http://", "https://", "/", "#")):
                return f'<a href="{href}" target="_blank" rel="noopener">{label}</a>'
            return m.group(0)
        s = _inline_link.sub(_link_repl, s)
        return s

    out = []
    in_list = False
    in_para = []

    def flush_para():
        if in_para:
            out.append("<p>" + inline(" ".join(in_para)) + "</p>")
            in_para.clear()

    def close_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw_line in text.split("\n"):
        line = esc(raw_line.rstrip())
        if not line.strip():
            flush_para()
            close_list()
            continue
        if line.startswith("#### "):
            flush_para(); close_list()
            out.append(f"<h5>{inline(line[5:])}</h5>")
        elif line.startswith("### "):
            flush_para(); close_list()
            out.append(f"<h4>{inline(line[4:])}</h4>")
        elif line.startswith("## "):
            flush_para(); close_list()
            out.append(f"<h3>{inline(line[3:])}</h3>")
        elif line.startswith("# "):
            flush_para(); close_list()
            out.append(f"<h2>{inline(line[2:])}</h2>")
        elif line.lstrip().startswith("- "):
            flush_para()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line.lstrip()[2:])}</li>")
        else:
            close_list()
            in_para.append(line.strip())
    flush_para()
    close_list()
    return '<div class="rn-content">\n' + "\n".join(out) + "\n</div>"


_BUILD_INFO = _compute_build_info()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Behind reverse proxies (e.g., Vercel), trust forwarded headers so url_for(..., _external=True)
# produces the correct https://<host>/... callback URLs.
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)


@app.context_processor
def _inject_build_info():
    """Make build_info available in every template (used by the navbar badge)."""
    return {"build_info": _BUILD_INFO}

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

    if path == '/api/health' or path == '/api/release-notes' or path in public_paths or path.startswith(public_prefixes) or path.startswith('/static'):
        return None

    # Endpoint macchina-a-macchina: il job che sincronizza IBKR gira headless e
    # non ha un cookie di sessione da presentare, si autentica col bearer token
    # condiviso. Il controllo vero resta dentro la rotta — qui si evita solo che
    # la guardia di sessione risponda 401 prima che la rotta veda la richiesta.
    if path.startswith('/api/ibkr/') and _ibkr_cron_authorized():
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


# ATR (Average True Range) per-simbolo. L'ATR giornaliero non cambia dentro
# la seduta, quindi la cache ha un TTL generoso.
_ES_ATR_CACHE = {
    "value": {},  # {symbol: {"atr": float, "fetched_at": float}}
}


def _compute_atr_cached(symbol: str = "ES=F", period: int = 14, max_age_seconds: int = 30 * 60) -> Optional[float]:
    """Wilder ATR(period) del simbolo, espresso nei punti del sottostante.

    Usa l'OHLC giornaliero via yfinance. Cache per-simbolo con TTL. Ritorna
    None quando i dati non sono disponibili: il chiamante deve degradare senza
    rompere il request handler (convenzione "silent failures").

    NB: l'unità dell'ATR deve combaciare con quella di prezzo/flip. Per la card
    ES usare "ES=F" (punti ES); per SPX un simbolo indice (^GSPC).
    """
    if yf is None:
        return None

    store = _ES_ATR_CACHE.setdefault("value", {})
    entry = store.get(symbol)
    now = time.time()
    if entry and (now - entry.get("fetched_at", 0.0)) < max_age_seconds and entry.get("atr"):
        return entry["atr"]

    try:
        hist = yf.Ticker(symbol).history(period="3mo", interval="1d", auto_adjust=False)
        if hist is None or hist.empty or len(hist) < period + 1:
            return entry["atr"] if entry else None

        high = hist["High"].astype(float)
        low = hist["Low"].astype(float)
        prev_close = hist["Close"].astype(float).shift(1)

        true_range = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

        # Smoothing di Wilder ≈ EMA con alpha = 1/period (adjust=False).
        atr_series = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        atr = float(atr_series.iloc[-1])
        if not (atr > 0):
            return entry["atr"] if entry else None

        store[symbol] = {"atr": round(atr, 2), "fetched_at": now}
        return store[symbol]["atr"]
    except Exception:
        return entry["atr"] if entry else None


# ============================================================================
# CBOE delayed quotes — GEX SPX reale (OI + gamma + IV per strike, no PDF)
# ============================================================================
# Fonte: https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json
# Gratis, ~15 min di ritardo, OI a chiusura precedente. Contiene per ogni
# contratto open_interest, gamma, iv e le greche — quindi il Net GEX si calcola
# direttamente, senza Black-Scholes. Vedi memory/argo-roadmap.

_SPX_GAMMA_CBOE_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}

_CBOE_OCC_RE = re.compile(r'^([A-Z]+?)(\d{6})([CP])(\d{8})$')

# Soglie regime Net GEX in $B PER PUNTO di SPX, come nel processo "Argo" del
# video: |x| <= 0.5 bivio, 0.5-1 debole, 1-3 moderato, >3 estremo.
_GEX_T_BIVIO = 0.5
_GEX_T_MODERATE = 1.0
_GEX_T_EXTREME = 3.0


def _fetch_cboe_json(url: str) -> Optional[Dict[str, Any]]:
    """GET del JSON 'delayed quotes' di CBOE. None su qualsiasi errore."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.cboe.com/delayed_quotes/",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception:
        return None


def _parse_cboe_occ(sym: str):
    """('SPXW', date, 'C', 7575.0) da un simbolo OCC CBOE, o None."""
    m = _CBOE_OCC_RE.match(sym or "")
    if not m:
        return None
    yymmdd, cp, strike_milli = m.group(2), m.group(3), m.group(4)
    try:
        exp = _dt.date(2000 + int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
    except ValueError:
        return None
    return m.group(1), exp, cp, int(strike_milli) / 1000.0


def get_spx_gamma_cboe_cached(max_age_seconds: int = 8 * 60) -> Optional[Dict[str, Any]]:
    """Chain SPX con OI + gamma reali da CBOE, aggregata per due scope.

    Ritorna: {"spot", "nearest_expiry", "source", "scopes": {"0dte": [rows],
    "all": [rows]}} dove ogni row è {strike, call_oi, put_oi, call_gamma_oi,
    put_gamma_oi}. `*_gamma_oi` = Σ(gamma·OI) già pesato, pronto per il GEX.
    None (o l'ultimo valido) se CBOE non è raggiungibile.
    """
    now = time.time()
    cached = _SPX_GAMMA_CBOE_CACHE.get("value")
    if cached and (now - _SPX_GAMMA_CBOE_CACHE.get("fetched_at", 0.0)) < max_age_seconds:
        return cached

    payload = _fetch_cboe_json("https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json")
    if not payload:
        return cached  # stale-if-error

    data = payload.get("data") or {}
    spot = data.get("current_price")
    options = data.get("options") or []
    if not spot or not options:
        return cached

    today = _dt.date.today()

    parsed = []
    nearest_exp = None
    for o in options:
        p = _parse_cboe_occ(o.get("option", ""))
        if not p:
            continue
        _root, exp, cp, strike = p
        parsed.append((exp, cp, strike, o))
        if exp >= today and (nearest_exp is None or exp < nearest_exp):
            nearest_exp = exp

    def _blank():
        return {"call_oi": 0.0, "put_oi": 0.0, "call_gamma_oi": 0.0, "put_gamma_oi": 0.0}

    agg_all: Dict[float, dict] = {}
    agg_0dte: Dict[float, dict] = {}

    for exp, cp, strike, o in parsed:
        oi = float(o.get("open_interest") or 0.0)
        if oi <= 0:
            continue
        gamma = float(o.get("gamma") or 0.0)
        targets = [agg_all]
        if nearest_exp is not None and exp == nearest_exp:
            targets.append(agg_0dte)
        for store in targets:
            row = store.setdefault(strike, _blank())
            if cp == "C":
                row["call_oi"] += oi
                row["call_gamma_oi"] += gamma * oi
            else:
                row["put_oi"] += oi
                row["put_gamma_oi"] += gamma * oi

    def _rows(store):
        return [
            {
                "strike": float(k),
                "call_oi": float(store[k]["call_oi"]),
                "put_oi": float(store[k]["put_oi"]),
                "call_gamma_oi": float(store[k]["call_gamma_oi"]),
                "put_gamma_oi": float(store[k]["put_gamma_oi"]),
            }
            for k in sorted(store.keys())
        ]

    result = {
        "spot": float(spot),
        "nearest_expiry": nearest_exp.isoformat() if nearest_exp else None,
        "source": "cboe_delayed",
        "scopes": {"0dte": _rows(agg_0dte), "all": _rows(agg_all)},
    }
    _SPX_GAMMA_CBOE_CACHE["value"] = result
    _SPX_GAMMA_CBOE_CACHE["fetched_at"] = now
    return result


def _gex_regime_band(net_gex_b: float) -> str:
    """Etichetta di regime dal Net GEX in $B/punto, soglie del video (0.5/1/3)."""
    a = abs(net_gex_b)
    if a <= _GEX_T_BIVIO:
        return "Bivio"
    sign = "Positivo" if net_gex_b > 0 else "Negativo"
    if a < _GEX_T_MODERATE:
        return f"{sign} debole"
    if a < _GEX_T_EXTREME:
        return f"{sign} moderato"
    return f"{sign} estremo"


def _compute_gex_profile(rows: list, spot: float) -> Optional[dict]:
    """Net GEX ($B), regime, flip proxy e profilo per-strike da righe con gamma.

    GEX per strike = (call_gamma_oi - put_gamma_oi) · 100 · spot
    ($ di hedging per 1 PUNTO di SPX, scala del video; dealer long-call/short-put).
    Il flip proxy è lo strike (più vicino allo spot) dove il gamma netto per-strike
    cambia segno; l'upgrade rigoroso (ricalcolo su griglia di spot con l'IV CBOE)
    è documentato in roadmap.
    """
    if not rows or not spot or spot <= 0:
        return None

    scale = 100.0 * float(spot)
    gex_by_strike = []
    net = 0.0
    for r in rows:
        g = (r["call_gamma_oi"] - r["put_gamma_oi"]) * scale
        net += g
        gex_by_strike.append({"strike": r["strike"], "gex": g})

    # Flip proxy: cambio di segno del gamma netto per-strike più vicino allo spot.
    flip = None
    best_dist = None
    for i in range(1, len(rows)):
        a = rows[i - 1]["call_gamma_oi"] - rows[i - 1]["put_gamma_oi"]
        b = rows[i]["call_gamma_oi"] - rows[i]["put_gamma_oi"]
        if (a < 0 <= b) or (a > 0 >= b) or a == 0:
            mid = (rows[i - 1]["strike"] + rows[i]["strike"]) / 2.0
            d = abs(mid - spot)
            if best_dist is None or d < best_dist:
                best_dist, flip = d, mid

    net_b = net / 1e9
    return {
        "net_gex_b": round(net_b, 3),
        "regime_band": _gex_regime_band(net_b),
        "gamma_flip_gex": round(flip, 2) if flip is not None else None,
        "gex_by_strike": [
            {"strike": g["strike"], "gex_b": round(g["gex"] / 1e9, 5)} for g in gex_by_strike
        ],
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


# Cache COT per simbolo: {symbol: {"value": ..., "fetched_at": ...}}
_COT_CACHE: Dict[str, Dict[str, Any]] = {}


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


# ============================================================================
# VIX — volatilità (domanda 3 del processo "Argo")
# ============================================================================
# Livello VIX (soglie del video: 20/25/30) + struttura a termine VIX/VIX3M
# (contango = calmo, backwardation = stress). Fonte: Yahoo quote (^VIX, ^VIX3M,
# ^VIX9D). Vedi memory/argo-roadmap.

_VIX_SNAPSHOT_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


def _vix_band(vix: float) -> str:
    """Banda di volatilità dalle soglie del video (20/25/30), + 'Calmo' <15."""
    if vix < 15:
        return "Calmo"
    if vix < 20:
        return "Normale"
    if vix < 25:
        return "Elevato"
    if vix < 30:
        return "Alto"
    return "Estremo"


def _vix_term_structure(vix: float, vix3m: float):
    """('Contango'|'Piatta'|'Backwardation', ratio) da VIX vs VIX3M.

    ratio = VIX/VIX3M: <0.98 contango (mercato calmo), >1.02 backwardation
    (stress, tipico dei giorni di gamma negativo/cascata).
    """
    if not vix or not vix3m or vix3m <= 0:
        return None, None
    ratio = vix / vix3m
    if ratio <= 0.98:
        term = "Contango"
    elif ratio <= 1.02:
        term = "Piatta"
    else:
        term = "Backwardation"
    return term, round(ratio, 3)


def _cboe_index_level(symbol_root: str) -> Optional[float]:
    """Livello di un indice CBOE (VIX, VIX3M, VIX9D) da /quotes/_SYMBOL.json."""
    payload = _fetch_cboe_json(
        f"https://cdn.cboe.com/api/global/delayed_quotes/quotes/_{symbol_root}.json"
    )
    data = (payload or {}).get("data") or {}
    v = data.get("current_price")
    if not v:
        v = data.get("close")
    try:
        v = float(v)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def get_vix_snapshot_cached(max_age_seconds: int = 3 * 60) -> Optional[Dict[str, Any]]:
    """Snapshot volatilità: VIX + banda + struttura a termine (VIX/VIX3M).

    Fonte: indici CBOE delayed (VIX, VIX3M, VIX9D). Ritorna {vix, vix9d, vix3m,
    band, term_structure, term_ratio, note} o l'ultimo valore valido/None se
    CBOE non risponde (silent failure). Yahoo v7/quote non è affidabile (richiede
    auth/crumb → 401), per questo usiamo CBOE come per il GEX.
    """
    now = time.time()
    cached = _VIX_SNAPSHOT_CACHE.get("value")
    if cached and (now - _VIX_SNAPSHOT_CACHE.get("fetched_at", 0.0)) < max_age_seconds:
        return cached

    vix = _cboe_index_level("VIX")
    vix3m = _cboe_index_level("VIX3M")
    vix9d = _cboe_index_level("VIX9D")

    # Fallback livello VIX via yfinance se CBOE non risponde.
    if vix is None and yf is not None:
        try:
            h = yf.Ticker("^VIX").history(period="5d", interval="1d")
            if h is not None and not h.empty:
                vix = float(h["Close"].iloc[-1])
        except Exception:
            vix = None

    if vix is None:
        return cached  # stale-if-error

    band = _vix_band(vix)
    term, ratio = _vix_term_structure(vix, vix3m)

    # Nota interpretativa: lega volatilità e regime gamma atteso.
    if band in {"Alto", "Estremo"} or term == "Backwardation":
        note = "Stress di volatilità: favorisce gamma negativo / cascata"
    elif band in {"Calmo", "Normale"} and term == "Contango":
        note = "Volatilità benigna: coerente con pinning / gamma positivo"
    else:
        note = "Volatilità di transizione"

    result = {
        "vix": round(vix, 2),
        "vix9d": round(vix9d, 2) if vix9d is not None else None,
        "vix3m": round(vix3m, 2) if vix3m is not None else None,
        "band": band,
        "term_structure": term,
        "term_ratio": ratio,
        "note": note,
        "source": "cboe_delayed",
    }
    _VIX_SNAPSHOT_CACHE["value"] = result
    _VIX_SNAPSHOT_CACHE["fetched_at"] = now
    return result


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


COT_API_BASE_URL = os.environ.get("COT_API_BASE_URL", "http://178.104.133.41:8080")

TRADINGSTER_COT_BASE_URL = "https://www.tradingster.com/cot/legacy-futures"

# Contratti COT esposti dall'app (CFTC Legacy Futures Only).
#   upstream="api"         → servizio esterno {COT_API_BASE_URL}/cot/{symbol}
#   upstream="tradingster" → scraping diretto di tradingster.com (contratti che
#                            il servizio esterno non espone: oro e bitcoin)
_COT_CONTRACTS: Dict[str, Dict[str, str]] = {
    "sp500":     {"contract_code": "13874+", "ticker": "ES",  "label": "S&P 500 (Consolidated)", "upstream": "api"},
    "nasdaq100": {"contract_code": "20974+", "ticker": "NQ",  "label": "NASDAQ-100 (Consolidated)", "upstream": "api"},
    "eurofx":    {"contract_code": "099741", "ticker": "6E",  "label": "EURO FX", "upstream": "api"},
    "gold":      {"contract_code": "088691", "ticker": "GC",  "label": "GOLD", "upstream": "tradingster"},
    "bitcoin":   {"contract_code": "133741", "ticker": "BTC", "label": "BITCOIN", "upstream": "tradingster"},
    # 067651 = WTI-PHYSICAL NYMEX, cioè il contratto CL (nel report Legacy la CFTC
    # ha rinominato "CRUDE OIL, LIGHT SWEET"); 067411 è l'ICE Europe, non il CL.
    "crudeoil":  {"contract_code": "067651", "ticker": "CL",  "label": "WTI CRUDE OIL", "upstream": "tradingster"},
}

# Simboli supportati (retrocompatibilità: era il set dei soli contratti del servizio esterno).
_COT_SYMBOLS = set(_COT_CONTRACTS)


def _tradingster_text(fragment: str) -> str:
    """Strip tag HTML e normalizza gli spazi di un frammento tradingster."""
    return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", fragment))).strip()


def _tradingster_number(fragment: str) -> Optional[float]:
    """Converte una cella tradingster ("+1,987", "13.0%", "&nbsp;") in numero."""
    text = _tradingster_text(fragment).replace(",", "").replace("%", "").replace("+", "")
    if not text or text == "-":
        return None
    try:
        return float(text) if "." in text else int(text)
    except Exception:
        return None


def _parse_tradingster_cot_history(raw: str, weeks: int = 12) -> List[Dict[str, Any]]:
    """Estrae lo storico settimanale dagli array JS `dataLong`/`dataShort`.

    La pagina tradingster incorpora le serie dei grafici (dal 2018 a oggi) come
    letterali JavaScript: sono la sola fonte di storico senza fare una request
    per ogni settimana. Le variazioni sono ricalcolate come differenza rispetto
    alla settimana precedente (stessa definizione della colonna "Changes" CFTC).
    """

    def _series(var_name: str) -> Dict[str, Dict[str, int]]:
        match = re.search(r"var\s+" + var_name + r"\s*=\s*\[(.*?)\];", raw, re.S)
        if not match:
            return {}
        out: Dict[str, Dict[str, int]] = {}
        pattern = (
            r"new Date\('(\d{4}-\d{2}-\d{2})'\),\s*Commercial:\s*(-?\d+),"
            r"\s*NonCommercial:\s*(-?\d+),\s*NonRept:\s*(-?\d+)"
        )
        for row in re.finditer(pattern, match.group(1)):
            out[row.group(1)] = {
                "commercial": int(row.group(2)),
                "non_commercial": int(row.group(3)),
                "non_reportable": int(row.group(4)),
            }
        return out

    longs = _series("dataLong")
    shorts = _series("dataShort")
    dates = sorted(set(longs) & set(shorts))
    if not dates:
        return []

    # +1 settimana per poter calcolare la variazione della più vecchia mostrata.
    window = dates[-(weeks + 1):]
    history: List[Dict[str, Any]] = []
    for idx in range(len(window) - 1, 0, -1):
        date = window[idx]
        prev = window[idx - 1]
        cur_l, cur_s = longs[date], shorts[date]
        prev_l, prev_s = longs[prev], shorts[prev]
        history.append({
            "report_date": date,
            "non_commercial": {
                "long": cur_l["non_commercial"],
                "short": cur_s["non_commercial"],
                "change_long": cur_l["non_commercial"] - prev_l["non_commercial"],
                "change_short": cur_s["non_commercial"] - prev_s["non_commercial"],
            },
            "commercial": {
                "long": cur_l["commercial"],
                "short": cur_s["commercial"],
                "change_long": cur_l["commercial"] - prev_l["commercial"],
                "change_short": cur_s["commercial"] - prev_s["commercial"],
            },
            "non_reportable": {
                "long": cur_l["non_reportable"],
                "short": cur_s["non_reportable"],
                "change_long": cur_l["non_reportable"] - prev_l["non_reportable"],
                "change_short": cur_s["non_reportable"] - prev_s["non_reportable"],
            },
        })
    return history


def _parse_tradingster_cot(raw: str, symbol: str, meta: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Traduce la pagina tradingster nello stesso payload del servizio COT esterno.

    La tabella ha quattro righe di dati (posizioni, variazioni, % di open
    interest, numero di traders), ognuna preceduta da una riga-etichetta:
    ci si aggancia alle etichette perché l'ultima riga (traders) ha due celle
    vuote e non avrebbe 9 celle numeriche come le altre.
    """

    body = re.search(r"<tbody>(.*?)</tbody>", raw, re.S)
    if not body:
        return None

    sections: Dict[str, List[Optional[float]]] = {}
    pending = "positions"  # la prima riga numerica sono le posizioni correnti
    for row in re.findall(r"<tr>(.*?)</tr>", body.group(1), re.S):
        cells = re.findall(r"<td[^>]*class=\"number\"[^>]*>(.*?)</td>", row, re.S)
        if len(cells) >= 7:
            if pending:
                sections[pending] = [_tradingster_number(c) for c in cells]
                pending = None
            continue
        label = _tradingster_text(row).lower()
        if "changes" in label:
            pending = "changes"
        elif "percent of open interest" in label:
            pending = "percent"
        elif "number of traders" in label:
            pending = "traders"

    positions = sections.get("positions")
    if not positions or len(positions) < 9:
        return None

    changes = sections.get("changes") or [None] * 9
    percent = sections.get("percent") or [None] * 9
    traders = sections.get("traders") or [None] * 9

    def _at(values: List[Optional[float]], idx: int) -> Optional[float]:
        return values[idx] if idx < len(values) else None

    report_date = None
    match = re.search(r"AS OF:\s*(\d{4}-\d{2}-\d{2})", raw)
    if match:
        report_date = match.group(1)

    market = meta.get("label")
    match = re.search(r"<strong>([^<]+?)</strong>\s*<br\s*/?>\s*<strong>\s*FUTURES ONLY POSITIONS", raw)
    if match:
        market = _tradingster_text(match.group(1))

    open_interest = None
    match = re.search(r"Open Interest:\s*<span class=\"number\">(.*?)</span>", raw, re.S)
    if match:
        open_interest = _tradingster_number(match.group(1))

    change_in_oi = None
    match = re.search(r"Change In Open Interest:\s*<span class=\"number\">(.*?)</span>\s*</span>", raw, re.S)
    if match:
        change_in_oi = _tradingster_number(match.group(1))

    nc_long, nc_short = _at(positions, 0), _at(positions, 1)
    nc_net = None
    if nc_long is not None and nc_short is not None:
        nc_net = int(nc_long - nc_short)

    nc_net_change = None
    ch_long, ch_short = _at(changes, 0), _at(changes, 1)
    if ch_long is not None and ch_short is not None:
        nc_net_change = int(ch_long - ch_short)

    latest = {
        "report_date": report_date,
        "market": market,
        "contract_code": meta.get("contract_code"),
        "open_interest": open_interest,
        "change_in_open_interest": change_in_oi,
        "non_commercial": {
            "long": nc_long,
            "short": nc_short,
            "spread": _at(positions, 2),
            "change_long": ch_long,
            "change_short": ch_short,
            "change_spread": _at(changes, 2),
            "pct_oi_long": _at(percent, 0),
            "pct_oi_short": _at(percent, 1),
            "traders_long": _at(traders, 0),
            "traders_short": _at(traders, 1),
        },
        "commercial": {
            "long": _at(positions, 3),
            "short": _at(positions, 4),
            "change_long": _at(changes, 3),
            "change_short": _at(changes, 4),
        },
        "non_reportable": {
            "long": _at(positions, 7),
            "short": _at(positions, 8),
            "change_long": _at(changes, 7),
            "change_short": _at(changes, 8),
        },
        "non_commercial_net": nc_net,
        "non_commercial_net_change": nc_net_change,
    }

    history = _parse_tradingster_cot_history(raw)
    # La settimana corrente arriva dalla tabella (dato ufficiale, non ricalcolato).
    if history and latest.get("report_date") and history[0].get("report_date") == latest.get("report_date"):
        history[0] = {k: v for k, v in latest.items() if k != "non_commercial_net" and k != "non_commercial_net_change"}

    return {
        "source": "CFTC Legacy Futures Only Report",
        "source_page": f"{TRADINGSTER_COT_BASE_URL}/{meta.get('contract_code')}",
        "symbol": symbol,
        "ticker": meta.get("ticker"),
        "label": meta.get("label"),
        "contract_code": meta.get("contract_code"),
        "market": market,
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "latest": latest,
        "history": history,
    }


def _fetch_tradingster_cot(symbol: str, meta: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Scarica e parsa la pagina COT tradingster per un contratto (oro, bitcoin)."""

    url = f"{TRADINGSTER_COT_BASE_URL}/{meta.get('contract_code')}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=6) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"error": f"COT fetch error: {e}"}

    try:
        parsed = _parse_tradingster_cot(raw, symbol, meta)
    except Exception as e:
        return {"error": f"COT parse error: {e}"}

    if not parsed:
        return {"error": "Tradingster: tabella COT non riconosciuta"}
    return parsed


def get_cot_cached(symbol: str, max_age_seconds: int = 60 * 60) -> Optional[Dict[str, Any]]:
    """Fetch COT (Commitment of Traders) data for a symbol with caching.

    Source: CFTC Legacy Futures Only Report, letto dal servizio esterno
    {COT_API_BASE_URL}/cot/{symbol} (sp500 / nasdaq100 / eurofx) oppure da
    tradingster.com per i contratti che quel servizio non espone (gold,
    bitcoin). Il report è settimanale, quindi 1h di cache è abbondante.
    """

    now = time.time()
    entry = _COT_CACHE.setdefault(symbol, {"value": None, "fetched_at": 0.0})
    cached = entry.get("value")
    fetched_at = float(entry.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    meta = _COT_CONTRACTS.get(symbol) or {}
    if meta.get("upstream") == "tradingster":
        data = _fetch_tradingster_cot(symbol, meta)
        if not isinstance(data, dict) or data.get("error"):
            # stale-while-error, come per il servizio esterno.
            return cached or data
        entry["value"] = data
        entry["fetched_at"] = now
        return data

    url = f"{COT_API_BASE_URL}/cot/{symbol}"
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

    entry["value"] = data
    entry["fetched_at"] = now
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


def _flip_distance_label(dist_atr: Optional[float]) -> Optional[str]:
    """Etichetta qualitativa della distanza prezzo↔gamma flip, in ATR.

    Soglie del processo "Argo": |d| <= 0.3 → sul flip, 0.3 < |d| <= 1 → vicino,
    > 1 → lontano. Ritorna None se la distanza in ATR non è calcolabile.
    """
    if dist_atr is None:
        return None
    side = 'sopra' if dist_atr > 0 else 'sotto'
    mag = abs(dist_atr)
    if mag <= 0.3:
        return 'sul flip'
    if mag <= 1.0:
        return f'vicino {side}'
    return f'lontano {side}'


def analyze_0dte(
    df: pd.DataFrame,
    current_price: float = None,
    levels_mode: str = "price",
    prefer_strike_multiple: Optional[float] = 25.0,
    atr: Optional[float] = None,
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

        # Distanza prezzo↔flip in ATR (domanda 2 del processo "Argo").
        # atr è opzionale: se assente i campi distanza restano None e il regime
        # torna a essere deciso dal semplice confronto prezzo vs flip.
        dist_points = None
        dist_atr = None
        on_flip = False
        if current_price is not None:
            dist_points = round(float(current_price) - gamma_flip, 2)
            if atr and atr > 0:
                dist_atr = round(dist_points / float(atr), 2)
                on_flip = abs(dist_atr) <= 0.3

        results['atr'] = round(float(atr), 2) if (atr and atr > 0) else None
        results['flip_distance_points'] = dist_points
        results['flip_distance_atr'] = dist_atr
        results['flip_distance_label'] = _flip_distance_label(dist_atr)

        # Regime: la banda ±0.3 ATR dà larghezza allo stato "sul flip", che con
        # il solo confronto d'uguaglianza non scattava mai.
        if on_flip:
            results['regime'] = 'At Gamma Flip'
            results['strategy'] = 'Cautela - punto di transizione (prezzo entro 0.3 ATR dal flip)'
        elif current_price is not None and current_price > gamma_flip:
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

    # Tabella strike completa: consente al client di ricalcolare i livelli (CP/GF)
    # quando l'utente inserisce un ES live manuale, senza dover ricaricare il PDF.
    try:
        results['strikes_table'] = [
            {
                'strike': float(row['Strike']),
                'call_oi': int(row['Call_OI']),
                'put_oi': int(row['Put_OI']),
                'gamma': int(row['Gamma_Exposure']),
            }
            for _, row in df_sorted.iterrows()
        ]
    except Exception:
        results['strikes_table'] = []

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
    # Serve the SVG favicon for legacy /favicon.ico requests.
    return redirect(url_for('static', filename='favicon.svg'), code=302)


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


def _cot_json_response(symbol: str):
    """Shared handler: COT report for a symbol as JSON (non-commercial focus)."""

    force = (request.args.get('force') or '').strip() == '1'
    try:
        data = get_cot_cached(symbol, max_age_seconds=0 if force else 60 * 60)
    except Exception as e:
        return jsonify({"error": f"Impossibile recuperare il COT {symbol}: {e}"})

    if not data:
        return jsonify({"error": f"Impossibile recuperare il COT {symbol} in questo momento"})

    try:
        fetched_at = float((_COT_CACHE.get(symbol) or {}).get('fetched_at') or 0.0)
        if fetched_at:
            data = dict(data)
            data['cache_age_seconds'] = int(max(0.0, time.time() - fetched_at))
    except Exception:
        pass

    return jsonify(data)


@app.route('/api/cot/<symbol>', methods=['GET'])
def api_cot(symbol):
    """Return weekly COT report for a supported symbol (vedi _COT_CONTRACTS)."""

    sym = (symbol or '').strip().lower()
    if sym not in _COT_SYMBOLS:
        return jsonify({
            "error": f"Simbolo COT non supportato: {symbol}",
            "available_symbols": sorted(_COT_SYMBOLS),
        }), 404
    return _cot_json_response(sym)


@app.route('/api/cot-sp500', methods=['GET'])
def api_cot_sp500():
    """Alias storico di /api/cot/sp500."""
    return _cot_json_response('sp500')


@app.route('/api/release-notes', methods=['GET'], endpoint='api_release_notes')
def api_release_notes():
    """Returns the full CHANGELOG.md rendered as HTML (single source of truth
    for the in-app "Cosa c'è di nuovo" modal). Public — no auth required."""
    text = _read_changelog_text()
    return jsonify({
        "version": _BUILD_INFO.get("version"),
        "date": _BUILD_INFO.get("date"),
        "html": _render_changelog_html(text),
    })


@app.route('/api/health', methods=['GET'], endpoint='api_health')
def api_health():
    mongo = _get_mongo_collection()
    google_oauth_configured = _ensure_google_oauth_registered()
    return jsonify({
        "status": "ok",
        "pymupdf_available": bool(_PYMUPDF_AVAILABLE),
        "app_build": _APP_BUILD,
        "version": _BUILD_INFO.get("version"),
        "python": _RUNTIME_PYTHON,
        "in_venv": bool(_IN_VENV),
        "virtual_env": os.getenv("VIRTUAL_ENV"),
        "mongo_configured": mongo is not None,
        "google_oauth_configured": google_oauth_configured,
        "google_oauth_missing": _google_oauth_missing_vars(),
        "authlib_available": OAuth is not None,
        # `cryptography` si importa solo dentro le funzioni OAuth di IBKR:
        # senza questo campo un pacchetto mancante sul deploy si manifesterebbe
        # come un generico "credenziali non configurate".
        "cryptography_available": _cryptography_available(),
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

def _build_spx_gamma_scope(rows: list, spot: float, atr: Optional[float]) -> Optional[dict]:
    """Compone Net GEX (profilo) + walls/livelli (analyze_0dte su OI) per uno scope."""
    if not rows:
        return None

    gex = _compute_gex_profile(rows, spot)

    df = pd.DataFrame([
        {
            "Strike": r["strike"],
            "Call_OI": r["call_oi"],
            "Put_OI": r["put_oi"],
            # Gamma_Exposure = GEX per-strike in $/punto (solo per display; la
            # logica dei livelli in analyze_0dte usa gli OI).
            "Gamma_Exposure": int((r["call_gamma_oi"] - r["put_gamma_oi"]) * 100 * spot),
        }
        for r in rows
    ])
    # SPX: strike a 5 punti → nessun bias ai multipli di 25.
    analysis = analyze_0dte(df, current_price=spot, atr=atr, prefer_strike_multiple=None)

    # Flip: preferisci il proxy da GEX, fallback all'euristica OI di analyze_0dte.
    flip = (gex or {}).get("gamma_flip_gex")
    if flip is None:
        flip = analysis.get("gamma_flip")

    dist_atr = None
    dist_label = None
    if flip and atr and atr > 0:
        dist_atr = round((spot - flip) / atr, 2)
        dist_label = _flip_distance_label(dist_atr)

    supports = analysis.get("supports") or []
    resistances = analysis.get("resistances") or []
    return {
        "net_gex_b": (gex or {}).get("net_gex_b"),
        "regime_band": (gex or {}).get("regime_band"),
        "gamma_flip": round(flip, 2) if flip else None,
        "put_wall": supports[0]["strike"] if supports else None,
        "call_wall": resistances[0]["strike"] if resistances else None,
        "supports": supports,
        "resistances": resistances,
        "regime": analysis.get("regime"),
        "strategy": analysis.get("strategy"),
        "flip_distance_atr": dist_atr,
        "flip_distance_label": dist_label,
        "gex_by_strike": (gex or {}).get("gex_by_strike") or [],
        "stats": analysis.get("stats"),
    }


@app.route('/api/spx-gamma', methods=['GET'])
@login_required
def api_spx_gamma():
    """GEX SPX live da CBOE (no-PDF), in due scope: 0DTE e aggregato.

    Query: ?scope=0dte|all|both (default both). Ritorna Net GEX in $B, regime,
    gamma flip, put/call wall, distanza in ATR e profilo gamma-per-strike.
    """
    snap = get_spx_gamma_cboe_cached()
    if not snap or not snap.get("spot"):
        return jsonify({"error": "Dati CBOE non disponibili", "source": None}), 503

    spot = float(snap["spot"])
    atr = _compute_atr_cached("^GSPC")

    scope_req = (request.args.get("scope") or "both").strip().lower()
    wanted = ["0dte", "all"] if scope_req in {"both", ""} else [scope_req]

    out = {
        "spot": round(spot, 2),
        "source": snap.get("source"),
        "nearest_expiry": snap.get("nearest_expiry"),
        "delayed_note": "CBOE delayed ~15 min · OI a chiusura precedente",
        "atr": round(atr, 2) if (atr and atr > 0) else None,
        "scopes": {},
    }
    for scope in wanted:
        rows = (snap.get("scopes") or {}).get(scope)
        built = _build_spx_gamma_scope(rows, spot, atr) if rows else None
        if built:
            out["scopes"][scope] = built

    if not out["scopes"]:
        return jsonify({"error": "Nessuno strike valido nei dati CBOE", "source": snap.get("source")}), 503

    return jsonify(out)


@app.route('/api/vix-regime', methods=['GET'])
@login_required
def api_vix_regime():
    """Volatilità (domanda 3 di "Argo"): livello VIX a bande + struttura a termine."""
    snap = get_vix_snapshot_cached()
    if not snap or snap.get("vix") is None:
        return jsonify({"error": "Dati VIX non disponibili"}), 503
    return jsonify(snap)


# ============================================================================
# WEB ROUTES - Main Application (PDF Analysis)
# ============================================================================


def _analyze_es_levels(df, current_price, levels_mode='price'):
    """Esegue analyze_0dte e allega le varianti dei livelli CP (current price) e GF (gamma flip).

    Condiviso tra l'endpoint /analyze (upload PDF) e /api/recalculate-levels
    (ricalcolo con ES live manuale), così la logica dei livelli resta unica.
    """
    # ATR su ES=F (stessa unità di prezzo/flip): alimenta la distanza in ATR
    # dal gamma flip. None se yfinance non è disponibile → degrada senza errori.
    es_atr = _compute_atr_cached("ES=F") if current_price else None

    results = analyze_0dte(df, current_price, levels_mode=levels_mode, atr=es_atr)
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

    return results


@app.route('/api/recalculate-levels', methods=['POST'])
@login_required
def api_recalculate_levels():
    """Ricalcola i livelli ES (gamma flip, supporti/resistenze CP e GF) usando un
    prezzo ES live fornito manualmente e la tabella strike dell'ultima analisi.

    Payload JSON:
        { "current_price": <float>, "strikes": [{strike, call_oi, put_oi, gamma}, ...] }
    Non richiede di ricaricare il PDF: usa la tabella strike già estratta.
    """
    payload = request.get_json(silent=True) or {}

    current_price = payload.get('current_price')
    try:
        current_price = float(current_price) if current_price is not None else None
    except (TypeError, ValueError):
        current_price = None
    if current_price is None or current_price <= 0:
        return jsonify({'error': 'Prezzo ES live mancante o non valido'}), 400

    raw_strikes = payload.get('strikes')
    if not isinstance(raw_strikes, list) or not raw_strikes:
        return jsonify({'error': 'Dati strike mancanti: ricarica il PDF per ricalcolare'}), 400

    rows = []
    for s in raw_strikes:
        if not isinstance(s, dict):
            continue
        try:
            rows.append({
                'Strike': float(s.get('strike')),
                'Call_OI': int(float(s.get('call_oi') or 0)),
                'Put_OI': int(float(s.get('put_oi') or 0)),
                'Gamma_Exposure': int(float(s.get('gamma') or 0)),
            })
        except (TypeError, ValueError):
            continue

    if not rows:
        return jsonify({'error': 'Nessuno strike valido nei dati forniti'}), 400

    df = pd.DataFrame(rows)
    levels_mode = (payload.get('levels_mode') or 'price').strip().lower()

    try:
        results = _analyze_es_levels(df, current_price, levels_mode=levels_mode)
    except Exception as e:
        return jsonify({'error': f'Errore durante il ricalcolo: {str(e)}'}), 500

    # Aggiorna la cache prezzo ES con l'input manuale (best-effort).
    try:
        _seed_es_price_manual(current_price, note="manual recalc")
    except Exception:
        pass

    # Persisti la nuova analisi ricalcolata come "ultima analisi" (best-effort).
    try:
        if isinstance(results, dict) and not results.get('error'):
            filename = (payload.get('filename') or 'ES live manuale').strip() or 'ES live manuale'
            _save_last_analysis(filename, results)
    except Exception:
        pass

    return jsonify(results)


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
        results = _analyze_es_levels(df, current_price, levels_mode=levels_mode)

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
# WEB ROUTES - Macro
# ============================================================================


@app.route('/macro')
@login_required
def macro_page():
    """Pagina Macro: dati macro/posizionamento (COT S&P 500)."""
    return render_template('macro.html')


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


def _checklist_num(value):
    """Numbers out of the checklist arrive as form strings: '' and 'null' mean 'not set'."""
    if value in (None, '', 'null'):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _checklist_day_pnl(checklist: dict) -> dict:
    """Reduce one day's checklist to the figures the calendar shows.

    The headline is the *net* result — the same number the "Net P&L" of the
    session shows — because that is the day's bottom line; gross and
    commissions stay available underneath so it can be taken apart.

    The day's P&L is the one written in the session when it's there: it may
    have been corrected by hand after the import. Only when it's missing is it
    summed back from the trades.
    """
    session = checklist.get('session') or {}
    trades = checklist.get('trades') or []

    executed, wins, closed, trades_pnl = 0, 0, 0, 0.0
    per_symbol: dict = {}
    for t in trades:
        if not isinstance(t, dict):
            continue
        result = t.get('result') or {}
        if result.get('executed') == 'skip':
            continue
        executed += 1
        pnl = _checklist_num(result.get('pnl'))
        if pnl is None:
            continue
        closed += 1
        trades_pnl += pnl
        if pnl > 0:
            wins += 1
        sym = (t.get('symbol') or '—').strip() or '—'
        bucket = per_symbol.setdefault(sym, {'symbol': sym, 'pnl': 0.0, 'trades': 0})
        bucket['pnl'] += pnl
        bucket['trades'] += 1

    gross = _checklist_num(session.get('daily_pnl'))
    if gross is None:
        gross = trades_pnl if closed else None

    commissions = _checklist_num(session.get('commissions')) or 0.0

    # Currency: whatever the imported accounts say. Mixed currencies would be
    # summed as if they were the same, so the day is flagged instead.
    currencies = {
        (v or {}).get('currency') for v in (session.get('account_netliq') or {}).values()
        if isinstance(v, dict) and (v or {}).get('currency')
    }

    if gross is None and not executed:
        return {}

    return {
        'pnl': round((gross or 0.0) - commissions, 2),
        'gross': round(gross, 2) if gross is not None else None,
        'commissions': round(commissions, 2),
        'trades': executed,
        'closed': closed,
        'wins': wins,
        'symbols': sorted(
            ({'symbol': s['symbol'], 'pnl': round(s['pnl'], 2), 'trades': s['trades']}
             for s in per_symbol.values()),
            key=lambda s: s['pnl'],
        ),
        'currencies': sorted(c for c in currencies if c),
        # Una giornata con trade eseguiti ma senza P&L è mezza compilata: il
        # totale è per difetto e la cella lo dice invece di darlo per zero.
        'partial': executed > closed,
    }


@app.route('/api/checklist/pnl-calendar', methods=['GET'])
@login_required
def api_checklist_pnl_calendar():
    """Daily P&L of the recorded checklist days, for the calendar tab.

    Reads the same documents the checklist writes: no separate archive to keep
    in sync, so a day corrected by hand shows corrected here too.
    """
    coll = _get_checklist_collection()
    if coll is None:
        return jsonify({'days': {}, 'currency': 'EUR',
                        'hint': 'MongoDB non configurato: lo storico non è disponibile.'})

    try:
        docs = list(coll.find({}, sort=[("date_key", -1)], limit=800))
    except TypeError:
        docs = list(coll.find({}).sort("date_key", -1).limit(800))
    except Exception as e:
        return jsonify({'days': {}, 'currency': 'EUR', 'hint': f'Storico non leggibile: {e}'})

    import re as _re
    days = {}
    currency_votes: dict = {}
    for doc in docs:
        date_key = (doc.get('date_key') or '').strip()
        if not _re.match(r'^\d{4}-\d{2}-\d{2}$', date_key):
            continue
        info = _checklist_day_pnl(doc.get('checklist') or {})
        if not info:
            continue
        for c in info.pop('currencies', []):
            currency_votes[c] = currency_votes.get(c, 0) + 1
        days[date_key] = info

    # Una sola valuta per il calendario: le giornate non si sommano fra valute
    currency = max(currency_votes, key=currency_votes.get) if currency_votes else 'EUR'
    hint = None
    if len(currency_votes) > 1:
        hint = ('Le giornate non sono tutte nella stessa valuta ('
                + ', '.join(sorted(currency_votes)) + '): i totali le sommano come se lo fossero.')

    return jsonify({'days': days, 'currency': currency, 'hint': hint})


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
# BROKER CSV IMPORT — parse order exports into round-trip trades
# ============================================================================
#
# Two sources are supported, each with its own row parser:
#   - "rithmic"   → Overcharts TSV export (Apex, AMP/Rithmic, ...), one file
#   - "tradovate" → Orders + Fills + Cash History, uploaded together
# Both boil down to the same list of "fills" and share `_group_round_trips()` /
# `_round_trip_to_trade()` for position tracking, P&L and trade-card shaping.
# What each Tradovate file adds: Orders the bracket levels (stop/target → R:R),
# Fills the per-fill commission and the broker's session date, Cash History the
# opening and closing balance with the movements that connect them.


def _futures_point_multiplier(symbol: str) -> float:
    """Dollar value of one point for a futures contract symbol.

    Micro contracts must be checked before their full-size counterparts
    (MNQ before NQ, MES before ES, ...).
    """
    _sym = (symbol or '').upper()
    if _sym.startswith('MNQ'):   return 2.0     # Micro E-mini NASDAQ
    elif _sym.startswith('MES'): return 5.0     # Micro E-mini S&P 500
    elif _sym.startswith('M2K'): return 5.0     # Micro Russell 2000
    elif _sym.startswith('MGC'): return 10.0    # Micro Gold
    elif _sym.startswith('MCL'): return 100.0   # Micro Crude Oil
    elif _sym.startswith('NQ'):  return 20.0    # E-mini NASDAQ
    elif _sym.startswith('RTY'): return 50.0    # E-mini Russell 2000
    elif _sym.startswith('GC'):  return 100.0   # Gold
    elif _sym.startswith('CL'):  return 1000.0  # Crude Oil
    return 50.0                                 # ES default $50


def _group_round_trips(fills: list) -> list:
    """Bucket fills by (account, symbol) and close them into round trips.

    Separate buckets keep e.g. MES and MNQ positions independent. Within a
    bucket fills are sorted chronologically (stable, so same-second fills keep
    file order) and a new round trip is closed every time the net position
    returns to flat. Each fill dict must carry:
    dt, time_str, side ('Buy'/'Sell'), qty, price, account, symbol.

    Returns a list of fill lists — the caller decides what to do with each.
    """
    from collections import defaultdict as _defaultdict

    if not fills:
        return []

    buckets: dict = _defaultdict(list)
    for f in fills:
        buckets[(f['account'], f['symbol'])].append(f)

    groups = []
    for _key, bucket_fills in buckets.items():
        bucket_fills.sort(key=lambda x: x['dt'])

        position = 0
        current: list = []
        for f in bucket_fills:
            current.append(f)
            position += f['qty'] if f['side'] == 'Buy' else -f['qty']
            if position == 0:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

    return groups


def _round_trip_to_trade(rt: list):
    """Shape one round trip into the trade object the checklist UI consumes.

    Returns None when the round trip has fills on one side only: that is an
    open position, not a trade with a result.
    """
    buys = [f for f in rt if f['side'] == 'Buy']
    sells = [f for f in rt if f['side'] == 'Sell']
    if not (buys and sells):
        return None

    is_long = rt[0]['side'] == 'Buy'

    total_buy_qty = sum(f['qty'] for f in buys)
    total_sell_qty = sum(f['qty'] for f in sells)
    closed_qty = min(total_buy_qty, total_sell_qty)
    is_open = total_buy_qty != total_sell_qty

    entry_fills, exit_fills = (buys, sells) if is_long else (sells, buys)
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

    _sym_label = rt[0].get('symbol', '')
    multiplier = _futures_point_multiplier(_sym_label)
    pnl_dollars = round(pnl_points * multiplier, 2)

    direction_label = 'Long' if is_long else 'Short'
    note_parts = [
        f"{direction_label} {closed_qty}x {_sym_label}".strip(),
        f"entry {avg_entry:.2f} → exit {avg_exit:.2f}",
    ]
    if is_open:
        note_parts.append("(posizione aperta)")

    # Clean symbol for display: strip exchange suffix (e.g. MESH6.CME -> MESH6)
    _sym_display = _sym_label.split('.')[0] if _sym_label else ''

    return {
        'time': rt[0]['time_str'],
        'account': rt[0]['account'],
        'symbol': _sym_display,
        'imported': True,
        'qty': closed_qty,
        'entry_price': round(avg_entry, 4),
        'exit_price': round(avg_exit, 4),
        'multiplier': multiplier,
        'open': is_open,
        'context': {'trend': 'long' if is_long else 'short'},
        'result': {
            'executed': 'true',
            'pnl': '' if is_open else str(pnl_dollars),
            'notes': ' | '.join(note_parts),
        },
    }


def _fills_to_round_trips(fills: list) -> list:
    """Group broker fills into round-trip trades ready for the checklist UI."""
    trades = [t for t in (_round_trip_to_trade(rt) for rt in _group_round_trips(fills)) if t]
    trades.sort(key=lambda x: x.get('time', ''))
    return trades


def _parse_apex_csv(content: str, date_filter: str = '') -> list:
    """Parse an Overcharts TSV order export and return a list of round-trip trades.

    Compatible with any broker connected to Overcharts (Apex, AMP/Rithmic, etc.)
    since they all share the same column layout. The function is named "apex"
    only because that was the first broker tested.

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

    return _fills_to_round_trips(fills)


# Timestamp layouts seen in Tradovate exports (Orders / Order History / Fills)
_TRADOVATE_DT_FORMATS = (
    '%m/%d/%Y %H:%M:%S',
    '%m/%d/%Y %H:%M',
    '%m/%d/%y %H:%M:%S',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%dT%H:%M:%S',
)


def _parse_tradovate_datetime(raw: str):
    """Parse a Tradovate timestamp, tolerating fractional seconds and 'Z'."""
    s = (raw or '').strip().replace('Z', '')
    if not s:
        return None
    if '.' in s:
        s = s.split('.')[0]
    for fmt in _TRADOVATE_DT_FORMATS:
        try:
            return _dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _parse_tradovate_money(raw: str):
    """Parse a Tradovate money cell: '"50,000.00"' / '-1.55' / '' → float|None."""
    s = (raw or '').strip().replace('"', '').replace(',', '').replace('$', '')
    if not s:
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _read_tradovate_rows(content: str):
    """Read a Tradovate CSV into (rows, getter).

    Columns are matched by header *name*, normalised (case/space/underscore
    insensitive), because Tradovate ships the same data with different column
    orders depending on the screen the export starts from. `getter(row, *names)`
    returns the first non-empty value among the candidate header names.
    """
    import csv as _csv
    import io as _io

    reader = _csv.DictReader(_io.StringIO(content))
    if not reader.fieldnames:
        return [], (lambda row, *names: '')

    def _norm(name: str) -> str:
        # Underscores are kept on purpose: Tradovate ships an internal `_price`
        # next to the display `Price`, and collapsing them would make which one
        # wins depend on column order.
        return (name or '').strip().lower().replace(' ', '')

    headers = {_norm(h): h for h in reader.fieldnames if h}

    def _get(row: dict, *names: str) -> str:
        for n in names:
            key = headers.get(_norm(n))
            if key is None:
                continue
            val = row.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
        return ''

    return list(reader), _get


def _classify_tradovate_file(content: str) -> str:
    """Guess which of the three Tradovate exports a file is, from its header.

    Filenames are unreliable (the user renames them, the browser appends "(1)"),
    the header row is not: each export has a column no other one carries.
    """
    first_line = (content or '').split('\n', 1)[0].lower()
    if 'cash change type' in first_line:
        return 'cash'
    if 'fill id' in first_line:
        return 'fills'
    if 'stop price' in first_line or 'limit price' in first_line:
        return 'orders'
    return ''


def _tradovate_side(raw: str) -> str:
    """' Sell' / 'B' / 'Buy' → 'Buy' | 'Sell' | ''."""
    s = (raw or '').strip().lower()
    if s.startswith('b'):
        return 'Buy'
    if s.startswith('s'):
        return 'Sell'
    return ''


def _parse_tradovate_orders(content: str) -> list:
    """Read every order row (filled or not) out of a Tradovate Orders export."""
    rows, _get = _read_tradovate_rows(content)

    orders = []
    for row in rows:
        try:
            order_id = int(_get(row, 'Order ID', 'orderId'))
        except (ValueError, TypeError):
            continue

        version_raw = _get(row, 'Version ID')
        try:
            version_id = int(version_raw) if version_raw else order_id
        except (ValueError, TypeError):
            version_id = order_id

        orders.append({
            'order_id': order_id,
            'version_id': version_id,
            'side': _tradovate_side(_get(row, 'B/S', 'Side', 'Action')),
            'type': _get(row, 'Type').strip().lower(),      # limit / stop / market
            'text': _get(row, 'Text').strip().lower(),      # multibracket / Exit / ...
            'status': _get(row, 'Status').strip().lower(),
            'limit_price': _parse_tradovate_money(_get(row, 'decimalLimit', 'Limit Price')),
            'stop_price': _parse_tradovate_money(_get(row, 'decimalStop', 'Stop Price')),
            'symbol': _get(row, 'Contract', 'Symbol', 'Product'),
            'account': _get(row, 'Account') or 'Unknown',
            'fill_dt': _parse_tradovate_datetime(_get(row, 'Fill Time')),
            'fill_qty': _parse_tradovate_money(_get(row, 'Filled Qty', 'filledQty')),
            'fill_price': _parse_tradovate_money(_get(row, 'Avg Fill Price', 'decimalFillAvg', 'avgPrice')),
        })

    orders.sort(key=lambda o: o['order_id'])
    return orders


def _tradovate_brackets(orders: list) -> dict:
    """Map each bracket's entry order id to its stop and target legs.

    A Tradovate bracket is three consecutive order ids: the entry, then two
    protective legs on the *opposite* side — one Limit (the target) and one Stop
    (the stop loss). Requiring exactly that shape is what keeps a standalone
    order (a manual market exit, say) from swallowing the next bracket's entry.

    Caveat worth knowing when reading the numbers: the export carries each
    order's *latest* version, so a stop that was trailed during the trade shows
    where it ended up, not where it started. `modified` flags that case.
    """
    brackets = {}
    i = 0
    n = len(orders)
    while i < n:
        entry = orders[i]
        legs = orders[i + 1:i + 3]

        is_bracket = (
            len(legs) == 2
            and entry['side'] in ('Buy', 'Sell')
            and all(leg['side'] and leg['side'] != entry['side'] for leg in legs)
            and {leg['type'] for leg in legs} == {'limit', 'stop'}
        )
        if not is_bracket:
            i += 1
            continue

        target_leg = next(leg for leg in legs if leg['type'] == 'limit')
        stop_leg = next(leg for leg in legs if leg['type'] == 'stop')

        brackets[entry['order_id']] = {
            'target': target_leg['limit_price'],
            'stop': stop_leg['stop_price'],
            'planned_entry': entry['limit_price'] if entry['type'] == 'limit' else entry['stop_price'],
            # An order whose latest version isn't the original was moved while live
            'modified': (
                target_leg['version_id'] != target_leg['order_id']
                or stop_leg['version_id'] != stop_leg['order_id']
            ),
        }
        i += 3

    return brackets


def _tradovate_fills_from_fills_csv(content: str, filter_date) -> list:
    """Fills from a Tradovate Fills export — the authoritative per-fill list.

    Preferred over the Orders export because it carries the commission actually
    charged on each fill and the broker's own trade date, which is not the
    calendar date of the timestamp for anything traded in the overnight session.
    """
    rows, _get = _read_tradovate_rows(content)

    fills = []
    for row in rows:
        side = _tradovate_side(_get(row, 'B/S', 'Side', 'Action'))
        if not side:
            continue

        qty = _parse_tradovate_money(_get(row, 'Quantity', '_qty'))
        price = _parse_tradovate_money(_get(row, 'Price', '_price'))
        if not qty or not price or qty <= 0 or price <= 0:
            continue

        dt = _parse_tradovate_datetime(_get(row, 'Timestamp', '_timestamp'))
        if dt is None:
            continue

        # `_tradeDate` is the session the fill belongs to; the timestamp's own
        # date is wrong for the overnight session (23:56 on the 26th is the 27th)
        trade_date = None
        raw_trade_date = _get(row, '_tradeDate', 'Trade Date')
        if raw_trade_date:
            parsed = _parse_tradovate_datetime(raw_trade_date + ' 00:00:00')
            if parsed is not None:
                trade_date = parsed.date()

        if filter_date is not None and (trade_date or dt.date()) != filter_date:
            continue

        order_id = None
        try:
            order_id = int(_get(row, '_orderId', 'Order ID'))
        except (ValueError, TypeError):
            pass

        fills.append({
            'dt': dt,
            'time_str': dt.strftime('%H:%M'),
            'side': side,
            'qty': int(qty),
            'price': price,
            'account': _get(row, 'Account', '_accountId') or 'Unknown',
            'symbol': _get(row, 'Contract', 'Symbol', 'Product'),
            'order_id': order_id,
            'commission': _parse_tradovate_money(_get(row, 'commission')) or 0.0,
        })

    return fills


def _tradovate_fills_from_orders(orders: list, filter_date) -> list:
    """Fills reconstructed from the Orders export, when Fills.csv isn't there.

    Only rows with status Filled are real: in a bracket export most rows are
    cancelled legs. Commissions aren't in this file, so they come out zero.
    """
    fills = []
    for o in orders:
        if o['status'] != 'filled' or o['fill_dt'] is None:
            continue
        if not o['fill_qty'] or not o['fill_price']:
            continue
        if not o['side']:
            continue
        if filter_date is not None and o['fill_dt'].date() != filter_date:
            continue

        fills.append({
            'dt': o['fill_dt'],
            'time_str': o['fill_dt'].strftime('%H:%M'),
            'side': o['side'],
            'qty': int(o['fill_qty']),
            'price': o['fill_price'],
            'account': o['account'],
            'symbol': o['symbol'],
            'order_id': o['order_id'],
            'commission': 0.0,
        })

    return fills


def _parse_tradovate_cash(content: str, date_filter: str = '') -> dict:
    """Per-account cash summary for a day, from a Tradovate Cash History export.

    Every row carries both the movement (`Delta`) and the running balance after
    it (`Amount`), so the opening balance is the first row's Amount minus its
    Delta — no need for a separate statement. `Date` is the session date, which
    is what we filter on.

    Movements are split into deposits, commissions and realized P&L; whatever
    doesn't fall in those three lands in `other` so the arithmetic
    start + deposits + realized − commissions + other = end always closes.
    """
    rows, _get = _read_tradovate_rows(content)

    per_account: dict = {}
    for row in rows:
        account = _get(row, 'Account') or 'Unknown'
        delta = _parse_tradovate_money(_get(row, 'Delta'))
        amount = _parse_tradovate_money(_get(row, 'Amount'))
        if delta is None or amount is None:
            continue

        day = _get(row, 'Date')
        if date_filter:
            parsed = _parse_tradovate_datetime((day or '') + ' 00:00:00')
            if parsed is None or parsed.strftime('%Y-%m-%d') != date_filter:
                continue

        try:
            seq = int(_get(row, 'Transaction ID'))
        except (ValueError, TypeError):
            seq = 0

        per_account.setdefault(account, []).append({
            'seq': seq,
            'dt': _parse_tradovate_datetime(_get(row, 'Timestamp')),
            'delta': delta,
            'amount': amount,
            'kind': _get(row, 'Cash Change Type').strip().lower(),
            'currency': _get(row, 'Currency') or 'USD',
        })

    summary = {}
    for account, entries in per_account.items():
        entries.sort(key=lambda e: (e['dt'] or _dt.datetime.min, e['seq']))

        start = entries[0]['amount'] - entries[0]['delta']
        end = entries[-1]['amount']

        deposits = sum(e['delta'] for e in entries if 'fund' in e['kind'])
        commissions = -sum(e['delta'] for e in entries if 'commission' in e['kind'] or 'fee' in e['kind'])
        realized = sum(e['delta'] for e in entries if 'trade' in e['kind'] or 'p&l' in e['kind'])

        summary[account] = {
            'currency': entries[0]['currency'],
            'start_cash': round(start, 2),
            'end_cash': round(end, 2),
            'deposits': round(deposits, 2),
            'commissions': round(commissions, 2),
            'realized': round(realized, 2),
            'other': round(end - start - deposits - realized + commissions, 2),
            'movements': len(entries),
        }

    return summary


def _import_tradovate(files: list, date_filter: str = '') -> dict:
    """Build the day's picture from any subset of the three Tradovate exports.

    Orders alone give the trades and the bracket levels; add Fills and the
    commissions and the broker's trade date come with them; add Cash History and
    the account balance is reconciled against the movements instead of inferred.
    Each file is recognised by its header, so they can be dropped in any order.
    """
    filter_date = None
    if date_filter:
        try:
            filter_date = _dt.datetime.strptime(date_filter, '%Y-%m-%d').date()
        except Exception:
            pass

    by_kind = {}
    unknown = []
    for name, content in files:
        kind = _classify_tradovate_file(content)
        if not kind:
            unknown.append(name)
            continue
        by_kind[kind] = (name, content)

    warnings = []
    if unknown:
        warnings.append('File non riconosciuti e ignorati: ' + ', '.join(unknown))

    orders = _parse_tradovate_orders(by_kind['orders'][1]) if 'orders' in by_kind else []
    brackets = _tradovate_brackets(orders)

    if 'fills' in by_kind:
        fills = _tradovate_fills_from_fills_csv(by_kind['fills'][1], filter_date)
    else:
        fills = _tradovate_fills_from_orders(orders, filter_date)
        if orders:
            warnings.append('Senza Fills.csv le commissioni per trade non sono disponibili.')

    if not orders and not fills and 'cash' not in by_kind:
        warnings.append('Nessun file Tradovate riconosciuto (Orders, Fills o Cash History).')

    trades = []
    for rt in _group_round_trips(fills):
        trade = _round_trip_to_trade(rt)
        if trade is None:
            continue

        trade['commission'] = round(sum(f.get('commission') or 0.0 for f in rt), 2)

        # The bracket belongs to the order that opened the position
        entry_side = rt[0]['side']
        bracket = None
        for f in rt:
            if f['side'] == entry_side:
                bracket = brackets.get(f.get('order_id'))
                if bracket:
                    break

        if bracket:
            # `:g` would round 29631.25 to 29631.2 — tick precision matters here
            def _price(v):
                return f"{v:.4f}".rstrip('0').rstrip('.') or '0'

            control = {}
            if bracket['stop'] is not None:
                control['stop_loss'] = _price(bracket['stop'])
                control['stop_defined'] = True
            if bracket['target'] is not None:
                control['target'] = _price(bracket['target'])
                control['target_defined'] = True

            # Spunta "R:R ≥ 1:1.5" solo se i due livelli lo dicono davvero. Uno
            # stop trailato finisce oltre l'entry e rende il rischio negativo:
            # in quel caso il rapporto non è calcolabile e la casella resta vuota.
            if bracket['stop'] is not None and bracket['target'] is not None:
                is_long = trade['context']['trend'] == 'long'
                entry = trade['entry_price']
                risk = (entry - bracket['stop']) if is_long else (bracket['stop'] - entry)
                reward = (bracket['target'] - entry) if is_long else (entry - bracket['target'])
                if risk > 0 and reward > 0 and reward / risk >= 1.5:
                    control['rr_ok'] = True

            if control:
                trade['control'] = control
            trade['bracket_modified'] = bool(bracket['modified'])

        trades.append(trade)

    trades.sort(key=lambda t: t.get('time', ''))

    accounts = _parse_tradovate_cash(by_kind['cash'][1], date_filter) if 'cash' in by_kind else {}
    for acct in accounts.values():
        acct['broker'] = 'Tradovate'

    # Commissions from the fills are a fallback for accounts the cash history
    # doesn't cover (e.g. Cash History.csv wasn't uploaded)
    for trade in trades:
        acct = accounts.setdefault(trade['account'], {'broker': 'Tradovate'})
        if 'commissions' not in acct:
            acct['commissions'] = 0.0
            acct['commissions_from_fills'] = True
        if acct.get('commissions_from_fills'):
            acct['commissions'] = round(acct['commissions'] + trade.get('commission', 0.0), 2)

    # Only claim a net liq equal to cash when nothing is left open
    open_accounts = {t['account'] for t in trades if t.get('open')}
    for name, acct in accounts.items():
        acct['flat'] = name not in open_accounts

    return {
        'trades': trades,
        'accounts': accounts,
        'sources': {kind: name for kind, (name, _c) in by_kind.items()},
        'warnings': warnings,
    }


def _import_rithmic(files: list, date_filter: str = '') -> dict:
    """Round-trip trades from an Overcharts TSV export (Apex, AMP/Rithmic, ...)."""
    name, content = files[0]
    return {
        'trades': _parse_apex_csv(content, date_filter),
        'accounts': {},
        'sources': {'overcharts': name},
        'warnings': (
            ['Solo il primo file è stato letto: la sorgente Rithmic accetta un export alla volta.']
            if len(files) > 1 else []
        ),
    }


# source key (as sent by the checklist UI) → importer
_TRADE_IMPORTERS = {
    'rithmic': _import_rithmic,
    'tradovate': _import_tradovate,
}


@app.route('/api/checklist/import-apex', methods=['POST'])
@login_required
def api_import_apex():
    """Parse a broker export (Rithmic/Overcharts or Tradovate) into trades.

    Accepts multipart form fields:
      - file: one or more CSV/TSV files (Tradovate takes Orders + Fills +
        Cash History together; each is recognised by its header)
      - date_key: YYYY-MM-DD — only fills matching this date are imported
      - source: 'rithmic' (default, Overcharts TSV) or 'tradovate'
    """
    uploaded = [f for f in request.files.getlist('file') if f and f.filename]
    if not uploaded:
        return jsonify({'error': 'Nessun file caricato'}), 400

    date_key = (request.form.get('date_key') or '').strip()
    import re as _re
    if date_key and not _re.match(r'^\d{4}-\d{2}-\d{2}$', date_key):
        return jsonify({'error': 'date_key non valido'}), 400

    # Default to 'rithmic' so older clients (no `source` field) keep working
    source = (request.form.get('source') or 'rithmic').strip().lower()
    importer = _TRADE_IMPORTERS.get(source)
    if importer is None:
        return jsonify({'error': f'Sorgente non supportata: {source}'}), 400

    # Accept only plain text / CSV — guard against large uploads
    MAX_SIZE = 2 * 1024 * 1024  # 2 MB per file
    files = []
    for f in uploaded:
        raw = f.read(MAX_SIZE + 1)
        if len(raw) > MAX_SIZE:
            return jsonify({'error': f'File troppo grande (max 2 MB): {f.filename}'}), 413
        try:
            files.append((f.filename, raw.decode('utf-8-sig', errors='replace')))
        except Exception:
            return jsonify({'error': f'Impossibile decodificare il file: {f.filename}'}), 400

    try:
        result = importer(files, date_filter=date_key)
    except Exception as e:
        return jsonify({'error': f'Errore parsing: {e}'}), 500

    return jsonify({
        'ok': True,
        'trades': result['trades'],
        'count': len(result['trades']),
        'accounts': result.get('accounts') or {},
        'sources': result.get('sources') or {},
        'warnings': result.get('warnings') or [],
        'date_filter': date_key,
        'source': source,
    })


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
    return (os.getenv("EDGAR_USER_AGENT") or "Polaris contact@bitsharp.it").strip()


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


def _compute_13f_period_info(today: Optional["_dt.date"] = None) -> dict:
    """Compute the current 13F reporting period and the next filing deadline.

    13F-HR are due 45 days after each calendar quarter end (SEC rule 13f-1).
    Returns:
      - current_report_date: latest quarter-end whose 45-day deadline has passed
      - current_quarter_label: e.g. "Q4 2025"
      - next_report_date: the upcoming quarter end (data being reported next)
      - next_deadline: the SEC deadline for that upcoming filing
      - days_until_next_deadline: int (can be negative if we're past it)
    """
    today = today or _dt.date.today()
    # Quarter ends: Mar 31, Jun 30, Sep 30, Dec 31 — plus 45 days = deadline.
    candidates = []
    for y in (today.year - 1, today.year, today.year + 1):
        for m, d in ((3, 31), (6, 30), (9, 30), (12, 31)):
            try:
                q_end = _dt.date(y, m, d)
            except ValueError:
                continue
            deadline = q_end + _dt.timedelta(days=45)
            candidates.append((q_end, deadline))
    candidates.sort(key=lambda t: t[0])

    passed = [c for c in candidates if c[1] <= today]
    upcoming = [c for c in candidates if c[1] > today]
    current = passed[-1] if passed else candidates[0]
    nxt = upcoming[0] if upcoming else candidates[-1]

    def _qlabel(d: "_dt.date") -> str:
        q = (d.month - 1) // 3 + 1
        return f"Q{q} {d.year}"

    return {
        "current_report_date": current[0].isoformat(),
        "current_deadline": current[1].isoformat(),
        "current_quarter": _qlabel(current[0]),
        "next_report_date": nxt[0].isoformat(),
        "next_deadline": nxt[1].isoformat(),
        "next_quarter": _qlabel(nxt[0]),
        "days_until_next_deadline": (nxt[1] - today).days,
        "today": today.isoformat(),
    }


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
        "period_info": _compute_13f_period_info(),
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

# Display metadata for each sector bucket (label IT, Bootstrap icon, accent color).
# Used by the Settori view to render the sector grid and the drill-down header.
_SCREENER_SECTOR_LABELS = {
    "Tech":          ("Tecnologia",          "bi-cpu",              "#60a5fa"),
    "Comms":         ("Comunicazioni",       "bi-broadcast",        "#a78bfa"),
    "Discretionary": ("Consumer Cyclical",   "bi-bag",              "#fb7185"),
    "Staples":       ("Consumer Defensive",  "bi-basket",           "#34d399"),
    "Financial":     ("Financial Services",  "bi-bank",             "#fbbf24"),
    "Healthcare":    ("Healthcare",          "bi-heart-pulse",      "#f472b6"),
    "Industrial":    ("Industrials",         "bi-gear",             "#94a3b8"),
    "Energy":        ("Energy",              "bi-fuel-pump",        "#fb923c"),
    "Materials":     ("Basic Materials",     "bi-bricks",           "#facc15"),
    "RealEstate":    ("Real Estate",         "bi-building",         "#22d3ee"),
    "Utilities":     ("Utilities",           "bi-lightning-charge", "#2dd4bf"),
    "Lusso":         ("Lusso",               "bi-gem",              "#e879f9"),
}

# Mapping from yfinance GICS sector strings to our internal bucket.
# Il lusso non è un settore GICS, quindi partendo dal solo `sector` il bucket
# "Lusso" (premio +5 sul P/E teorico) non veniva mai assegnato a nessuno e
# questi titoli finivano in "Discretionary" (sconto 0). Si recupera dal campo
# `industry`, che sia FMP sia yfinance espongono.
#
# 1) Industry che identificano il lusso senza ambiguità.
_SCREENER_LUXURY_INDUSTRIES = {
    "Luxury Goods",
}

# 2) Override per ticker: aziende che il modello tratta come lusso ma la cui
#    `industry` dice altro — Ferrari è "Auto - Manufacturers", Moncler e
#    Cucinelli sono "Apparel - Manufacturers", indistinguibili dall'abbigliamento
#    di massa. Lista curata a mano: allargarla o restringerla è una scelta di
#    metodo sulla strategia, non un dettaglio tecnico.
_SCREENER_LUXURY_TICKERS = {
    "RACE.MI",   # Ferrari
    "MONC.MI",   # Moncler
    "BC.MI",     # Brunello Cucinelli
    "TOD.MI",    # Tod's
    "CFR.SW",    # Richemont (Cartier)
    "MC.PA",     # LVMH
    "RMS.PA",    # Hermès
    "KER.PA",    # Kering
    "1913.HK",   # Prada
}

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


def _resolve_bucket(sector: Optional[str], industry: Optional[str] = None,
                    ticker: Optional[str] = None) -> str:
    """Bucket settoriale per lo sconto Damodaran.

    Il lusso viene prima del settore: un titolo lusso è sempre "Consumer
    Cyclical" per GICS, quindi controllando solo il settore il premio +5 non
    scatterebbe mai. Fallback su Tech come prima, per i settori sconosciuti."""
    if ticker and ticker.strip().upper() in _SCREENER_LUXURY_TICKERS:
        return "Lusso"
    if (industry or "").strip() in _SCREENER_LUXURY_INDUSTRIES:
        return "Lusso"
    return _SCREENER_GICS_TO_BUCKET.get(sector, "Tech")


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

# Euronext Amsterdam (.AS) — prime 30 per capitalizzazione sopra i 2B,
# dallo stock screener FMP (exchange=AMS). Quotano tutte in EUR, quindi i
# filtri della strategia valgono senza conversioni. Alcune sono domiciliate
# fuori dai Paesi Bassi (Shell e Unilever GB, ArcelorMittal LU): resta
# corretto trattarle come mercato NL, il bucket paese lo impone il mercato.
_SCREENER_NL_UNIVERSE = [
    "ASML.AS", "SHELL.AS", "UNA.AS", "INGA.AS", "PRX.AS", "REN.AS",
    "MT.AS", "CCEP.AS", "ASM.AS", "HEIA.AS", "FER.AS", "ADYEN.AS",
    "ABN.AS", "AD.AS", "UMG.AS", "EXO.AS", "PHIA.AS", "DSFIR.AS",
    "NN.AS", "HEIO.AS", "CSG.AS", "BESI.AS", "WKL.AS", "KPN.AS",
    "CVC.AS", "HAL.AS", "ASRNL.AS", "AGN.AS", "MICC.AS", "AKZA.AS",
]

# Borsa di Madrid (.MC) — stesso criterio (exchange=BME, mcap > 2B).
# Esclusi i cross-listing latinoamericani col prefisso X (XVALO.MC = Vale,
# XBBDC.MC = Bradesco): sono titoli brasiliani, prenderebbero lo sconto
# paese europeo che non gli compete.
_SCREENER_ES_UNIVERSE = [
    "SAN.MC", "ITX.MC", "BBVA.MC", "IBE.MC", "CABK.MC", "MTS.MC",
    "ELE.MC", "AENA.MC", "FER.MC", "REP.MC", "ACS.MC", "NTGY.MC",
    "AMS.MC", "IAG.MC", "TEF.MC", "SAB.MC", "CLNX.MC", "BKT.MC",
    "MAP.MC", "ANA.MC", "IDR.MC", "PUIG.MC", "UNI.MC", "MRL.MC",
    "RED.MC", "ANE.MC", "GRF.MC", "FCC.MC", "LOG.MC", "ENG.MC",
]

# Maps a screener market code to the country bucket used by the Damodaran
# discount table (controls country_disc).
_SCREENER_MARKET_TO_COUNTRY = {
    "US": "US",
    "IT": "IT",
    "DE": "EU",
    "NL": "EU",
    "ES": "EU",
    "IN": "EM",
}

_SCREENER_VALID_MARKETS = ("US", "IT", "DE", "NL", "ES", "IN")

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


# FMP restituisce il codice ISO-2 in profile["country"] ("IT", "DE"), non il
# nome esteso. Col piano Ultimate FMP risponde anche sui ticker non-US, quindi
# entrambe le grafie devono risolvere allo stesso bucket: senza questa mappa
# un ticker .MI finirebbe nel bucket "US" (sconto_paese 0 invece di -5).
_SCREENER_ISO_TO_BUCKET = {
    "US": "US",
    "IT": "IT",
    "DE": "EU", "FR": "EU", "ES": "EU", "NL": "EU", "CH": "EU", "GB": "EU",
    "UK": "EU", "IE": "EU", "SE": "EU", "DK": "EU", "FI": "EU", "BE": "EU",
    "AT": "EU", "NO": "EU", "PT": "EU", "LU": "EU",
    "JP": "JP",
    "CN": "CN", "HK": "CN", "TW": "CN",
    "IN": "EM", "BR": "EM", "MX": "EM", "ZA": "EM", "RU": "EM", "TR": "EM",
    "ID": "EM",
}


def _map_country_to_bucket(country_name: Optional[str]) -> str:
    """Bucket paese per lo sconto Damodaran. Accetta sia il nome esteso
    (yfinance: "Italy") sia il codice ISO-2 (FMP: "IT"). Default: US."""
    name = (country_name or "").strip()
    if not name:
        return "US"
    bucket = _SCREENER_COUNTRY_TO_BUCKET.get(name)
    if bucket:
        return bucket
    return _SCREENER_ISO_TO_BUCKET.get(name.upper(), "US")


def _screener_universe_for(market: str) -> list:
    """Universe for a given market code. US always uses the full ~95-ticker
    universe — Vercel can handle it within the 60s budget thanks to FMP
    (nessun fallback per rate-limit) + 8 worker paralleli. Dal piano Ultimate
    anche IT/DE/IN passano da FMP, non più solo da yfinance."""
    if market == "US":
        return _SCREENER_US_UNIVERSE
    if market == "IT":
        return _SCREENER_IT_UNIVERSE
    if market == "DE":
        return _SCREENER_DE_UNIVERSE
    if market == "NL":
        return _SCREENER_NL_UNIVERSE
    if market == "ES":
        return _SCREENER_ES_UNIVERSE
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


# Zone classification — single source of truth for the
# Affare/Sconto/Equa/Cara buckets used by the screener.
# Compares peNow (price / forward_eps) against pe_theoretical: lower ratio
# means the market is paying less of what the model considers fair, leaving
# more room for multiple expansion (Serafini's preferred buy setup).
_ZONE_THRESHOLDS = (0.35, 0.55, 0.85)  # ratio peNow/peTheo
_ZONE_RANK_NA = 4  # used when inputs missing — ranked last


_ZONE_LABELS = ("Affare", "Sconto", "Equa", "Cara", "N/D")


def _compute_zone_rank(forward_eps, current_price, pe_theoretical) -> int:
    """0=Affare, 1=Sconto, 2=Equa, 3=Cara, 4=N/D. Lower = more attractive
    for purchase per the Serafini methodology."""
    if not forward_eps or forward_eps <= 0:
        return _ZONE_RANK_NA
    if not current_price or current_price <= 0:
        return _ZONE_RANK_NA
    if not pe_theoretical or pe_theoretical <= 0:
        return _ZONE_RANK_NA
    ratio = (current_price / forward_eps) / pe_theoretical
    t1, t2, t3 = _ZONE_THRESHOLDS
    if ratio <= t1:
        return 0
    if ratio <= t2:
        return 1
    if ratio <= t3:
        return 2
    return 3


def _zone_label_for(rank: int) -> str:
    if 0 <= rank < len(_ZONE_LABELS):
        return _ZONE_LABELS[rank]
    return "N/D"


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
        "zone_rank": _compute_zone_rank(forward_eps, current_price, pe_theo),
    }


_FMP_BASE_URL = "https://financialmodelingprep.com/stable"
_FMP_TIMEOUT_SECONDS = 8


def _fmp_get(path: str, _timeout: Optional[float] = None, **params) -> Optional[Any]:
    """GET wrapper for FMP `stable` API. Returns parsed JSON, or None on any
    failure (missing key, network error, non-200, error payload). Never raises.

    `_timeout` ha l'underscore per non collidere con gli eventuali parametri
    di query di FMP, che arrivano tutti da **params."""
    api_key = (os.getenv("FMP_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        params["apikey"] = api_key
        qs = urllib.parse.urlencode(params)
        url = f"{_FMP_BASE_URL}/{path}?{qs}"
        req = urllib.request.Request(url, headers={"User-Agent": "polaris/1.0"})
        with urllib.request.urlopen(req, timeout=_timeout or _FMP_TIMEOUT_SECONDS) as resp:
            if resp.status != 200:
                return None
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict) and (data.get("Error Message") or data.get("error")):
            return None
        return data
    except Exception:
        return None


def _fmp_probe(path: str, **params) -> dict:
    """Come _fmp_get, ma restituisce l'esito invece di ingoiarlo.

    Usato solo dalla rotta di diagnostica: il percorso normale resta
    silenzioso per non far mai fallire una request per colpa di FMP. Serve
    perché un titolo che compare con badge YF può dipendere da tre cause
    diverse (chiave assente, endpoint negato dal piano, risposta vuota) che
    dal risultato finale sono indistinguibili."""
    api_key = (os.getenv("FMP_API_KEY") or "").strip()
    out = {"endpoint": path, "ok": False}
    if not api_key:
        out["error"] = "FMP_API_KEY non configurata"
        return out
    try:
        params["apikey"] = api_key
        url = f"{_FMP_BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "polaris/1.0"})
        with urllib.request.urlopen(req, timeout=_FMP_TIMEOUT_SECONDS) as resp:
            out["http_status"] = resp.status
            data = json.loads(resp.read().decode("utf-8"))
        if isinstance(data, dict) and (data.get("Error Message") or data.get("error")):
            out["error"] = str(data.get("Error Message") or data.get("error"))[:300]
            return out
        rows = len(data) if isinstance(data, list) else 1
        out["rows"] = rows
        if not data:
            out["error"] = ("risposta vuota — il piano non copre questo mercato "
                            "oppure il simbolo non esiste su FMP")
            return out
        out["ok"] = True
        return out
    except urllib.error.HTTPError as e:
        out["http_status"] = e.code
        try:
            # 402/403 arrivano col messaggio "requires higher plan" nel body.
            out["error"] = e.read().decode("utf-8")[:300]
        except Exception:
            out["error"] = str(e)
        return out
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out


def _fetch_ticker_fundamentals_fmp(ticker: str) -> Optional[dict]:
    """Fetch fundamentals for a single ticker via FinancialModelingPrep
    `stable` API. Unica fonte dei fondamentali dello screener.
    Returns None when required fields are missing or FMP is unavailable.
    """
    profile_data = _fmp_get("profile", symbol=ticker)
    if not profile_data or not isinstance(profile_data, list) or not profile_data:
        return None
    p = profile_data[0]

    current_price = p.get("price")
    market_cap = p.get("marketCap")
    beta = p.get("beta")
    sector = p.get("sector") or ""
    industry = p.get("industry") or ""
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
    # `from` esplicito: col piano Ultimate lo storico arriva a 30+ anni e senza
    # filtro scaricheremmo migliaia di righe per usarne 260 (banda + latenza).
    dev_st_pct = None
    try:
        vol_from = (_dt.date.today() - _dt.timedelta(days=550)).isoformat()
        hist = _fmp_get("historical-price-eod/light", symbol=ticker,
                        **{"from": vol_from})
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

    bucket = _resolve_bucket(sector, industry, ticker)

    return {
        "ticker": ticker,
        "name": long_name,
        "yf_sector": sector,
        "industry": industry,
        "bucket": bucket,
        "country_iso": country_iso,
        "country": _map_country_to_bucket(country_iso),
        "market_cap": float(market_cap) if market_cap else None,
        "beta": float(beta) if beta is not None else None,
        "current_price": float(current_price),
        "forward_eps": float(forward_eps),
        "growth_5y": float(growth),
        "dev_st_pct": dev_st_pct,
        "_source": "fmp",
    }


def _fetch_ticker_fundamentals(ticker: str) -> Optional[dict]:
    """Fondamentali di un ticker. Fonte unica: FMP.

    Restituisce None quando FMP non copre il simbolo o non ha i campi minimi
    (EPS forward futuro positivo, prezzo, growth). Il fallback su yfinance e'
    stato rimosso: col piano Ultimate FMP copre i mercati che ci servono, e il
    fallback introduceva due dati eterogenei nella stessa lista — la growth
    yfinance nasce da LTG/+1y, quella FMP dal CAGR del consenso sugli anni
    fiscali — oltre a far comparire titoli che FMP non sa analizzare.
    Un ticker senza dati ora sparisce dalla lista, invece di comparire con
    numeri costruiti in un altro modo."""
    return _fetch_ticker_fundamentals_fmp(ticker)


# ============================================================================
# HISTORICAL FORWARD P/E (Serafini "zona di riacquisto storica")
# ----------------------------------------------------------------------------
# Builds a 5-year time series of forward P/E using the "hindsight NTM" method
# (same approach Tikr shows on its forward-P/E history chart): for each past
# date D, NTM_EPS(D) = sum of the next 4 quarterly EPS reported after D.
# A stock is flagged as being in a historical buy zone when its current
# forward P/E sits in the bottom quartile of this distribution.
# ============================================================================

_PE_HISTORY_YEARS = 5
_PE_HISTORY_SAMPLE_EVERY_N_DAYS = 7
_PE_HISTORY_CACHE_TTL_SECONDS = 7 * 24 * 3600
_MONGO_PE_HISTORY_COLLECTION = None

_INSIDER_CACHE_TTL_SECONDS = 24 * 3600
_MONGO_INSIDER_COLLECTION = None
_EDGAR_CIK_CACHE: dict = {}  # ticker -> CIK int; 0 = confirmed not found


def _get_mongo_pe_history_collection():
    """Lazy getter for the forward P/E history cache collection."""
    global _MONGO_CLIENT, _MONGO_PE_HISTORY_COLLECTION
    if _MONGO_PE_HISTORY_COLLECTION is not None:
        return _MONGO_PE_HISTORY_COLLECTION
    if MongoClient is None:
        return None
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_PE_HISTORY_COLLECTION") or "pe_history_cache").strip()
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]
        try:
            coll.create_index("ticker", unique=True)
            coll.create_index("computed_at", expireAfterSeconds=_PE_HISTORY_CACHE_TTL_SECONDS)
        except Exception:
            pass
        _MONGO_PE_HISTORY_COLLECTION = coll
        return coll
    except Exception:
        return None


def _get_mongo_insider_collection():
    """Lazy getter for insider transactions cache collection."""
    global _MONGO_CLIENT, _MONGO_INSIDER_COLLECTION
    if _MONGO_INSIDER_COLLECTION is not None:
        return _MONGO_INSIDER_COLLECTION
    if MongoClient is None:
        return None
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_INSIDER_COLLECTION") or "insider_cache").strip()
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        db = _MONGO_CLIENT[db_name]
        coll = db[coll_name]
        try:
            coll.create_index("ticker", unique=True)
            coll.create_index("computed_at", expireAfterSeconds=_INSIDER_CACHE_TTL_SECONDS)
        except Exception:
            pass
        _MONGO_INSIDER_COLLECTION = coll
        return coll
    except Exception:
        return None


def _fetch_insider_transactions(ticker: str) -> Optional[list]:
    """Fetch insider transactions (last 90 days): SEC EDGAR first, FMP fallback.
    Returns list (possibly empty) or None if all sources fail."""
    cutoff = (_dt.date.today() - _dt.timedelta(days=90)).isoformat()

    # SEC EDGAR: authoritative for all US public companies
    edgar = _fetch_insider_edgar(ticker, cutoff)
    if edgar is not None:
        return edgar

    # FMP fallback (may cover non-US stocks)
    data = _fmp_get("insider-trading", symbol=ticker, limit=50)
    if not isinstance(data, list):
        return None
    txns = []
    for item in data:
        date_str = (item.get("transactionDate") or item.get("date") or "")[:10]
        if not date_str or date_str < cutoff:
            continue
        tx_type = item.get("transactionType") or item.get("type") or ""
        is_buy = tx_type.startswith("P-")
        is_sell = tx_type.startswith("S-")
        if not is_buy and not is_sell:
            continue
        name = (item.get("reportingName") or item.get("acquirorName")
                or item.get("insiderName") or item.get("name") or "")
        title = item.get("title") or item.get("typeOfOwner") or ""
        price = item.get("price")
        qty = (item.get("securitiesTransacted") or item.get("sharesTransacted")
               or item.get("shares"))
        try:
            price = float(price) if price is not None else None
        except (TypeError, ValueError):
            price = None
        try:
            qty = float(qty) if qty is not None else None
        except (TypeError, ValueError):
            qty = None
        txns.append({
            "date": date_str,
            "type": "buy" if is_buy else "sell",
            "name": name,
            "title": title,
            "price": price,
            "qty": qty,
        })
    return txns


def _get_edgar_cik(ticker: str) -> Optional[int]:
    """Return SEC EDGAR CIK for a US ticker. Memory-cached per process.
    Downloads /files/company_tickers.json once and populates the full map."""
    if ticker in _EDGAR_CIK_CACHE:
        return _EDGAR_CIK_CACHE[ticker] or None
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Polaris luca.taurisano@bitsharp.it"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        for entry in raw.values():
            t = (entry.get("ticker") or "").upper()
            c = entry.get("cik_str")
            if t and c is not None:
                _EDGAR_CIK_CACHE[t] = int(c)
    except Exception:
        _EDGAR_CIK_CACHE[ticker] = 0
        return None
    cik = _EDGAR_CIK_CACHE.get(ticker)
    if not cik:
        _EDGAR_CIK_CACHE[ticker] = 0
    return cik or None


def _parse_form4_xml(xml_text: str, fallback_date: str) -> list:
    """Parse a Form 4 XML; return one aggregated row per (date, type) pair.
    Multiple same-day tranches by the same insider are merged into one row
    with summed quantity and weighted-average price."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)
    except Exception:
        return []

    name, title = "", ""
    try:
        n = root.find(".//rptOwnerName")
        if n is not None and n.text:
            name = n.text.strip().title()
        t = root.find(".//officerTitle")
        if t is not None and t.text:
            title = t.text.strip()
        if not title:
            if root.findtext(".//isDirector") == "1":
                title = "Director"
            elif root.findtext(".//isOfficer") == "1":
                title = "Officer"
    except Exception:
        pass

    # Aggregate tranches: key = (date, type) -> {qty_sum, weighted_price_sum}
    groups: dict = {}
    for txn in root.findall(".//nonDerivativeTransaction"):
        try:
            code_el = txn.find("transactionCoding/transactionCode")
            code = (code_el.text or "").strip() if code_el is not None else ""
            if code == "P":
                tx_type = "buy"
            elif code == "S":
                tx_type = "sell"
            else:
                continue

            date_el = txn.find("transactionDate/value")
            date_str = (date_el.text or fallback_date)[:10] if date_el is not None else fallback_date

            shares_el = txn.find("transactionAmounts/transactionShares/value")
            price_el = txn.find("transactionAmounts/transactionPricePerShare/value")
            qty = float(shares_el.text) if (shares_el is not None and shares_el.text) else 0.0
            price = float(price_el.text) if (price_el is not None and price_el.text) else 0.0

            key = (date_str, tx_type)
            if key not in groups:
                groups[key] = {"qty": 0.0, "val": 0.0, "name": name, "title": title}
            groups[key]["qty"] += qty
            groups[key]["val"] += qty * price
        except Exception:
            continue

    txns = []
    for (date_str, tx_type), g in sorted(groups.items(), reverse=True):
        total_qty = g["qty"]
        avg_price = g["val"] / total_qty if total_qty > 0 else None
        txns.append({
            "date": date_str,
            "type": tx_type,
            "name": g["name"],
            "title": g["title"],
            "price": round(avg_price, 4) if avg_price else None,
            "qty": total_qty,
        })
    return txns


def _fetch_insider_edgar(ticker: str, cutoff: str) -> Optional[list]:
    """Fetch Form 4 insider transactions from SEC EDGAR.
    Returns list (possibly empty) or None if CIK lookup fails."""
    from concurrent.futures import ThreadPoolExecutor

    cik = _get_edgar_cik(ticker)
    if not cik:
        return None

    subs_url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    try:
        req = urllib.request.Request(
            subs_url, headers={"User-Agent": "Polaris luca.taurisano@bitsharp.it"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            subs = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None

    recent = subs.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])

    form4s = []
    for i, form in enumerate(forms):
        if form not in ("4", "4/A"):
            continue
        d = dates[i] if i < len(dates) else ""
        if d < cutoff:
            break  # reverse-chronological
        acc = (accessions[i] if i < len(accessions) else "").replace("-", "")
        if acc:
            form4s.append((d, acc))

    if not form4s:
        return []

    form4s = form4s[:20]
    ua = "Polaris luca.taurisano@bitsharp.it"

    def fetch_one(args):
        filing_date, acc_nd = args
        # Form 4 raw XML is always form4.xml regardless of the styled primaryDocument
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nd}/form4.xml"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": ua})
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status != 200:
                    return []
                xml_text = resp.read().decode("utf-8", errors="replace")
            return _parse_form4_xml(xml_text, filing_date)
        except Exception:
            return []

    all_txns = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for result in ex.map(fetch_one, form4s):
            all_txns.extend(result)

    # Deduplicate (same filing may appear in 4 and 4/A)
    seen: set = set()
    out = []
    for t in sorted(all_txns, key=lambda x: x["date"], reverse=True):
        if t["date"] < cutoff:
            continue
        key = (t["date"], t["type"], t["name"], round(t["qty"] or 0))
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _percentile(sorted_vals, p):
    """Linear-interpolated percentile for a pre-sorted list. p in [0,100]."""
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _compute_forward_pe_history_fmp(ticker: str) -> Optional[dict]:
    """Compute a ~5y weekly series of forward P/E using hindsight NTM EPS.

    Returns dict {ticker, computed_at, series: [{date, pe}], stats: {q1, median, q3,
    current_pe, current_percentile, count, in_buy_zone}} or None on failure.
    """
    today = _dt.date.today()
    from_date = (today - _dt.timedelta(days=_PE_HISTORY_YEARS * 365 + 30)).isoformat()

    # Quarterly EPS (10y horizon to ensure we have NTM windows for early dates).
    quarters = _fmp_get(
        "income-statement", symbol=ticker, period="quarter", limit=44
    )
    if not quarters or not isinstance(quarters, list):
        return None
    eps_quarters = []
    for q in quarters:
        d = (q.get("date") or "")[:10]
        # L'API `stable` espone epsDiluted (camelCase); "epsdiluted" era la
        # grafia della v3. Fallback finale su eps (basic).
        eps = q.get("epsDiluted")
        if eps is None:
            eps = q.get("epsdiluted")
        if eps is None:
            eps = q.get("eps")
        if d and eps is not None:
            try:
                eps_quarters.append((d, float(eps)))
            except (TypeError, ValueError):
                continue
    if len(eps_quarters) < 8:
        return None
    eps_quarters.sort()  # ascending by date

    # Historical daily prices.
    hist = _fmp_get("historical-price-eod/light", symbol=ticker, **{"from": from_date})
    if not hist or not isinstance(hist, list):
        return None
    prices = []
    for h in hist:
        d = (h.get("date") or "")[:10]
        p = h.get("price")
        if d and p:
            try:
                prices.append((d, float(p)))
            except (TypeError, ValueError):
                continue
    if len(prices) < 60:
        return None
    prices.sort()  # ascending by date

    # For each historical price date, find the next 4 quarters and sum.
    # Walk both lists in order (O(n)).
    series = []
    qi = 0
    for idx, (pdate, price) in enumerate(prices):
        if idx % _PE_HISTORY_SAMPLE_EVERY_N_DAYS != 0:
            continue
        # Advance qi until eps_quarters[qi].date > pdate
        while qi < len(eps_quarters) and eps_quarters[qi][0] <= pdate:
            qi += 1
        # Need 4 quarters strictly after pdate.
        if qi + 4 > len(eps_quarters):
            continue
        ntm_eps = sum(eps_quarters[qi + j][1] for j in range(4))
        if ntm_eps <= 0:
            continue
        pe = price / ntm_eps
        if pe <= 0 or pe > 500:  # sanity cap
            continue
        series.append({"date": pdate, "pe": round(pe, 2), "price": round(price, 2)})

    if len(series) < 20:
        return None

    pe_values = sorted(s["pe"] for s in series)
    q1 = _percentile(pe_values, 25)
    median = _percentile(pe_values, 50)
    q3 = _percentile(pe_values, 75)

    # Identify the historical max/min P/E points (with their dates and prices).
    max_pe_point = max(series, key=lambda s: s["pe"])
    min_pe_point = min(series, key=lambda s: s["pe"])

    # Current point from current forward EPS estimate.
    current_pe = None
    current_percentile = None
    current_price = None
    in_buy_zone = False
    fund = _fetch_ticker_fundamentals_fmp(ticker)
    if fund and fund.get("forward_eps") and fund["forward_eps"] > 0 and fund.get("current_price"):
        current_pe = fund["current_price"] / fund["forward_eps"]
        current_price = fund["current_price"]
        below = sum(1 for v in pe_values if v <= current_pe)
        current_percentile = round(100.0 * below / len(pe_values), 1)
        in_buy_zone = current_pe <= q1 if q1 else False

    return {
        "ticker": ticker,
        # v3: EPS diluito letto dal campo corretto (epsDiluted) — le serie v2
        # erano costruite sull'EPS basic, quindi vanno ricalcolate.
        "schema_version": 3,
        "computed_at": _dt.datetime.utcnow(),
        "series": series,
        "stats": {
            "q1": round(q1, 2) if q1 else None,
            "median": round(median, 2) if median else None,
            "q3": round(q3, 2) if q3 else None,
            "min": round(pe_values[0], 2),
            "max": round(pe_values[-1], 2),
            "max_pe_value": max_pe_point["pe"],
            "max_pe_date": max_pe_point["date"],
            "max_pe_price": max_pe_point["price"],
            "min_pe_value": min_pe_point["pe"],
            "min_pe_date": min_pe_point["date"],
            "min_pe_price": min_pe_point["price"],
            "current_pe": round(current_pe, 2) if current_pe else None,
            "current_price": round(current_price, 2) if current_price else None,
            "current_percentile": current_percentile,
            "in_buy_zone": in_buy_zone,
            "count": len(series),
        },
    }


def _get_forward_pe_history(ticker: str) -> Optional[dict]:
    """Cached fetch of forward P/E history. Mongo-backed with 7-day TTL.
    Returns None when FMP can't deliver enough data."""
    ticker = (ticker or "").strip().upper()
    if not ticker:
        return None
    coll = _get_mongo_pe_history_collection()
    if coll is not None:
        try:
            doc = coll.find_one({"ticker": ticker})
            if doc and doc.get("schema_version") == 3:
                doc.pop("_id", None)
                return doc
        except Exception:
            pass
    fresh = _compute_forward_pe_history_fmp(ticker)
    if fresh is None:
        return None
    if coll is not None:
        try:
            coll.replace_one({"ticker": ticker}, fresh, upsert=True)
        except Exception:
            pass
    return fresh


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
                        # Pre-zone-rank rows lack `zone_rank` — compute on read
                        # so sorting and the UI badge work uniformly.
                        if "zone_rank" not in row:
                            row["zone_rank"] = _compute_zone_rank(
                                row.get("forward_eps"),
                                row.get("current_price"),
                                row.get("pe_theoretical"),
                            )
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
    limit = max(1, min(1000, limit))

    market = (request.args.get('market') or 'US').strip().upper()
    if market not in _SCREENER_VALID_MARKETS:
        return jsonify({"error": f"invalid market '{market}'", "valid": list(_SCREENER_VALID_MARKETS)}), 400

    fresh = _ensure_screener_cache_fresh(market=market)
    cache = _get_market_cache(market)
    results = list(cache.get("results") or [])
    qualified = [r for r in results if _stock_passes_strategy(r)]

    # Sort: zone bucket ascending (Affare first), then discount_pct descending.
    qualified.sort(key=lambda r: (
        r.get("zone_rank", _ZONE_RANK_NA),
        -(r.get("discount_pct") or -999),
    ))
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
    """Typeahead sui simboli FMP. Fino a 10 match: [{symbol, name, exchange, type}].

    Interroga in parallelo search-symbol (match sul ticker) e search-name (match
    sulla ragione sociale), i match per simbolo in testa. Fonte unica FMP: prima
    passava da Yahoo, che proponeva simboli su cui FMP non ha nulla — le linee
    regionali tedesche tipo AG1.F — cioe' risultati che portavano dritti a un
    "dati non disponibili" quando li sceglievi.

    Risponde sempre 200, anche se FMP e' irraggiungibile, per non rompere la UI.
    """
    q = (request.args.get('q') or '').strip()
    if not q or len(q) > 50:
        return jsonify({"results": []})

    # Timeout stretto: e' una typeahead, meglio nessun risultato che un campo
    # di ricerca che si pianta per otto secondi.
    def _search(path):
        return _fmp_get(path, _timeout=3, query=q, limit=10)

    paths = ("search-symbol", "search-name")
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as ex:
            responses = list(ex.map(_search, paths))
    except Exception:
        responses = []

    seen = set()
    out = []
    for data in responses:
        if not isinstance(data, list):
            continue
        for item in data:
            sym = (item.get("symbol") or "").strip()
            if not sym or sym in seen:
                continue
            seen.add(sym)
            out.append({
                "symbol": sym,
                "name": item.get("name") or "",
                "exchange": item.get("exchangeFullName") or item.get("exchange") or "",
                # FMP non espone il tipo di strumento in ricerca: al suo posto
                # mostriamo la valuta, che sulle quotazioni estere serve di piu'.
                "type": item.get("currency") or "",
            })
            if len(out) >= 10:
                break
        if len(out) >= 10:
            break
    return jsonify({"results": out})


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


@app.route('/api/screener/pe-history/<ticker>', methods=['GET'])
@login_required
def api_screener_pe_history(ticker):
    """Historical forward P/E (5y, weekly) + Q1/median/Q3 + current percentile.
    Used by the screener/portfolio cards to surface Serafini's "zona di
    riacquisto storica" — current forward P/E in the bottom quartile of
    its multi-year distribution. Mongo-cached for 7 days per ticker.
    """
    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm or not all(c.isalnum() or c in ".-" for c in ticker_norm) or len(ticker_norm) > 12:
        return jsonify({"error": "invalid ticker", "ticker": ticker_norm}), 400
    res = _get_forward_pe_history(ticker_norm)
    if res is None:
        return jsonify({"error": "history not available", "ticker": ticker_norm}), 404
    computed_at = res.get("computed_at")
    if hasattr(computed_at, "isoformat"):
        computed_at = computed_at.isoformat() + "Z"
    return jsonify({
        "ticker": res.get("ticker", ticker_norm),
        "computed_at": computed_at,
        "series": res.get("series", []),
        "stats": res.get("stats", {}),
    })


@app.route('/api/insider/<ticker>', methods=['GET'])
@login_required
def api_insider_transactions(ticker):
    """Insider transactions for <ticker> in the last 90 days, from FMP.
    Mongo-cached for 24 h per ticker."""
    ticker_norm = (ticker or "").strip().upper()
    if not ticker_norm or not all(c.isalnum() or c in ".-" for c in ticker_norm) or len(ticker_norm) > 12:
        return jsonify({"error": "invalid ticker", "ticker": ticker_norm}), 400

    coll = _get_mongo_insider_collection()
    if coll is not None:
        try:
            doc = coll.find_one({"ticker": ticker_norm})
            if doc:
                doc.pop("_id", None)
                computed_at = doc.get("computed_at")
                if hasattr(computed_at, "isoformat"):
                    doc["computed_at"] = computed_at.isoformat() + "Z"
                return jsonify(doc)
        except Exception:
            pass

    txns = _fetch_insider_transactions(ticker_norm)
    should_cache = txns is not None
    limited = txns is None  # True = no source could deliver data (plan/coverage limit)
    if txns is None:
        txns = []

    now = _dt.datetime.utcnow()
    result = {"ticker": ticker_norm, "transactions": txns, "limited": limited, "computed_at": now}
    if should_cache and coll is not None:
        try:
            coll.replace_one({"ticker": ticker_norm}, result, upsert=True)
        except Exception:
            pass
    result["computed_at"] = now.isoformat() + "Z"
    return jsonify(result)


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


@app.route('/api/screener/fmp-status', methods=['GET'])
@login_required
def api_screener_fmp_status():
    """Diagnostica FMP (admin). Dice se la chiave è configurata in questo
    ambiente e cosa risponde davvero l'API sui tre endpoint che alimentano lo
    screener, per un ticker a scelta (default ENI.MI, cioè un non-US).

    Serve a distinguere le tre cause di un badge YF: cache vecchia, chiave
    mancante/di un altro account, oppure mercato non coperto dal piano.
    """
    if not _is_admin():
        return jsonify({"error": "admin only"}), 403

    ticker = (request.args.get('ticker') or 'ENI.MI').strip().upper()
    api_key = (os.getenv("FMP_API_KEY") or "").strip()
    out = {
        "ticker": ticker,
        "key_configured": bool(api_key),
        # Ultimi 4 caratteri: bastano a capire SE la chiave deployata è quella
        # dell'account aggiornato, senza esporre il segreto.
        "key_suffix": api_key[-4:] if api_key else None,
    }
    if not api_key:
        out["checks"] = []
        out["verdict"] = ("FMP_API_KEY non è configurata in questo ambiente: senza "
                          "chiave lo screener non ha più alcuna fonte dati, il "
                          "fallback yfinance è stato rimosso.")
        return jsonify(out)

    vol_from = (_dt.date.today() - _dt.timedelta(days=550)).isoformat()
    out["checks"] = [
        _fmp_probe("profile", symbol=ticker),
        _fmp_probe("analyst-estimates", symbol=ticker, period="annual", limit=10),
        _fmp_probe("historical-price-eod/light", symbol=ticker, **{"from": vol_from}),
    ]

    fund = _fetch_ticker_fundamentals_fmp(ticker)
    out["fundamentals_ok"] = fund is not None
    out["fundamentals"] = fund

    failed = [c for c in out["checks"] if not c.get("ok")]
    if not failed and fund is not None:
        out["verdict"] = (f"FMP risponde correttamente su {ticker}. Se la lista mostra "
                          "ancora YF è cache: usa 'Forza ricalcolo' sul mercato aperto.")
    elif failed:
        out["verdict"] = ("FMP non risponde su questi endpoint: "
                          + ", ".join(f"{c['endpoint']} ({c.get('error') or 'errore'})"
                                      for c in failed))
    else:
        out["verdict"] = (f"Gli endpoint rispondono ma {ticker} non ha i campi minimi "
                          "(EPS forward futuro positivo + growth): il titolo non compare in lista.")
    return jsonify(out)


# ============================================================================
# SECTOR DRILL-DOWN (live FMP/yfinance data, no mock)
# ----------------------------------------------------------------------------
# Reuses the same screener cache as Damodaran view. The "criteria per sector"
# = the same Serafini strategy filters PLUS the per-bucket sector_disc applied
# in _calculate_damodaran_target. No new fundamentals fetched beyond what the
# main refresh already pulled (FMP profile + analyst-estimates + history,
# yfinance fallback).
# ============================================================================


@app.route('/api/screener/sectors', methods=['GET'])
@login_required
def api_screener_sectors():
    """List of sectors with per-bucket counts for a market.
    Counts come from the live cache (FMP/yfinance results). No mock data.
    """
    market = (request.args.get('market') or 'US').strip().upper()
    if market not in _SCREENER_VALID_MARKETS:
        return jsonify({"error": f"invalid market '{market}'"}), 400

    fresh = _ensure_screener_cache_fresh(market=market)
    cache = _get_market_cache(market)
    results = list(cache.get("results") or [])

    counts = {}
    qualified_counts = {}
    for r in results:
        b = r.get("bucket") or "Tech"
        counts[b] = counts.get(b, 0) + 1
        if _stock_passes_strategy(r):
            qualified_counts[b] = qualified_counts.get(b, 0) + 1

    sectors = []
    for bucket, (label, icon, color) in _SCREENER_SECTOR_LABELS.items():
        sectors.append({
            "bucket": bucket,
            "label": label,
            "icon": icon,
            "color": color,
            "sector_disc": _SCREENER_SECTOR_DISCOUNTS.get(bucket, 0),
            "count": counts.get(bucket, 0),
            "qualified_count": qualified_counts.get(bucket, 0),
        })

    return jsonify({
        "market": market,
        "computed_at": cache.get("computed_at"),
        "is_fresh": fresh,
        "in_progress": cache.get("in_progress", False),
        "universe_size": len(_screener_universe_for(market)),
        "evaluated_count": len(results),
        "qualified_count": sum(qualified_counts.values()),
        "sectors": sectors,
    })


@app.route('/api/screener/sectors/<bucket>', methods=['GET'])
@login_required
def api_screener_sector_top(bucket):
    """Top N qualified stocks for a single sector bucket in a market.
    Same ranking as the Damodaran top: zone_rank ASC, then discount_pct DESC.
    """
    bucket = (bucket or "").strip()
    if bucket not in _SCREENER_SECTOR_LABELS:
        return jsonify({
            "error": f"invalid sector '{bucket}'",
            "valid": list(_SCREENER_SECTOR_LABELS.keys()),
        }), 400

    market = (request.args.get('market') or 'US').strip().upper()
    if market not in _SCREENER_VALID_MARKETS:
        return jsonify({"error": f"invalid market '{market}'"}), 400

    try:
        limit = int(request.args.get('limit') or 5)
    except (TypeError, ValueError):
        limit = 5
    limit = max(1, min(20, limit))

    fresh = _ensure_screener_cache_fresh(market=market)
    cache = _get_market_cache(market)
    results = list(cache.get("results") or [])
    sector_rows = [r for r in results if (r.get("bucket") or "Tech") == bucket]
    qualified = [r for r in sector_rows if _stock_passes_strategy(r)]
    qualified.sort(key=lambda r: (
        r.get("zone_rank", _ZONE_RANK_NA),
        -(r.get("discount_pct") or -999),
    ))
    top = qualified[:limit]

    label, icon, color = _SCREENER_SECTOR_LABELS[bucket]
    return jsonify({
        "market": market,
        "bucket": bucket,
        "label": label,
        "icon": icon,
        "color": color,
        "sector_disc": _SCREENER_SECTOR_DISCOUNTS.get(bucket, 0),
        "computed_at": cache.get("computed_at"),
        "is_fresh": fresh,
        "in_progress": cache.get("in_progress", False),
        "evaluated_count": len(sector_rows),
        "qualified_count": len(qualified),
        "top": top,
    })


@app.route('/api/screener/sectors/<bucket>/lookup/<ticker>', methods=['GET'])
@login_required
def api_screener_sector_lookup(bucket, ticker):
    """On-demand analysis scoped to a sector. Uses the ticker's REAL sector
    discount (so the math stays honest), but flags `sector_mismatch=True`
    when the ticker's bucket differs from the user's chosen sector — the UI
    surfaces a warning instead of silently mis-classifying.
    """
    bucket = (bucket or "").strip()
    if bucket not in _SCREENER_SECTOR_LABELS:
        return jsonify({
            "error": f"invalid sector '{bucket}'",
            "valid": list(_SCREENER_SECTOR_LABELS.keys()),
        }), 400

    ticker_norm = (ticker or "").strip().upper()
    if (not ticker_norm
        or not all(c.isalnum() or c in ".-" for c in ticker_norm)
        or len(ticker_norm) > 12):
        return jsonify({"error": "invalid ticker", "ticker": ticker_norm}), 400

    fund = _fetch_ticker_fundamentals(ticker_norm)
    if not fund:
        return jsonify({
            "error": "ticker not found or missing data (forward EPS / growth)",
            "ticker": ticker_norm,
        }), 404

    detected_country = _map_country_to_bucket(fund.get("country_iso"))
    fund["country"] = detected_country
    actual_bucket = fund.get("bucket") or "Tech"

    calc = _calculate_damodaran_target(
        avg_growth=fund["growth_5y"],
        forward_eps=fund["forward_eps"],
        current_price=fund["current_price"],
        country=fund["country"],
        bucket=actual_bucket,
        dev_st_pct=fund["dev_st_pct"],
    )
    row = {**fund, **calc, "market": detected_country}
    check = _stock_strategy_check(row)
    row["passes_strategy"] = check["passes"]
    row["strategy_reasons"] = check["reasons"]
    row["requested_bucket"] = bucket
    row["sector_mismatch"] = (actual_bucket != bucket)
    if actual_bucket in _SCREENER_SECTOR_LABELS:
        row["actual_sector_label"] = _SCREENER_SECTOR_LABELS[actual_bucket][0]
    return jsonify(row)


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
    by_zone = {label: 0 for label in _ZONE_LABELS}
    valid = 0
    for h in holdings:
        if h.get("error"):
            continue
        valid += 1
        sector = h.get("bucket") or "N/D"
        country = h.get("country") or "N/D"
        by_sector[sector] = by_sector.get(sector, 0) + 1
        by_country[country] = by_country.get(country, 0) + 1
        rank = h.get("zone_rank")
        if rank is None:
            rank = _compute_zone_rank(
                h.get("forward_eps"), h.get("current_price"), h.get("pe_theoretical"),
            )
        by_zone[_zone_label_for(rank)] += 1
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
# INTERACTIVE BROKERS — snapshot posizioni/ordini + calendario earnings
# ============================================================================
#
# IBKR non è raggiungibile da qui: la Client Portal API vuole un gateway locale
# con login giornaliero, che su Vercel non esiste. Lo snapshot arriva quindi da
# fuori — il job schedulato legge posizioni e ordini dal connettore IBKR e li
# posta su /api/ibkr/sync — e l'app lo conserva su Mongo, lo arricchisce con le
# date earnings di FMP e lo serve alla pagina portafoglio.

_MONGO_IBKR_COLLECTION = None

# Un ordine "vivo" è ancora eseguibile. REPLACED resta nello snapshot ma non
# conta: è la versione superata di un ordine che IBKR ha già rimpiazzato, e
# contarla significherebbe elencare due volte lo stesso ordine.
_IBKR_LIVE_ORDER_STATUSES = {
    "NEW", "SUBMITTED", "PRESUBMITTED", "PENDINGSUBMIT",
    "PENDINGCHANGE", "PARTIALLYFILLED", "QUEUED",
}

# Tetto sulle righe accettate da una sync: lo snapshot finisce in un singolo
# documento Mongo (limite 16 MB) e un account normale sta ampiamente sotto.
_IBKR_MAX_ROWS = 500

# "Buy 100 BAC" / "Sell 84 CNC" — IBKR non espone il simbolo come campo a sé
# negli ordini, va letto dalla descrizione.
_IBKR_ORDER_DESC_RE = re.compile(r"^\s*(?:buy|sell)\s+[\d.,]+\s+(\S+)", re.IGNORECASE)


def _get_mongo_ibkr_collection():
    """Lazy getter per lo snapshot IBKR: un documento per proprietario.

    La chiave è l'email e non lo `user_key` di sessione: un conto IBKR
    appartiene a una persona, non alla particolare identità Google con cui
    quella persona ha fatto login, e il job che scrive lo snapshot gira
    headless, senza sessione da cui derivare un `google:<sub>`.
    """
    global _MONGO_CLIENT, _MONGO_IBKR_COLLECTION
    if _MONGO_IBKR_COLLECTION is not None:
        return _MONGO_IBKR_COLLECTION
    if MongoClient is None:
        return None
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_IBKR_COLLECTION") or "ibkr_snapshot").strip()
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        coll = _MONGO_CLIENT[db_name][coll_name]
        try:
            coll.create_index("owner_email", unique=True)
        except Exception:
            pass
        _MONGO_IBKR_COLLECTION = coll
        return coll
    except Exception:
        return None


def _current_user_email() -> Optional[str]:
    """Email dell'utente in sessione, normalizzata. Usata solo dallo snapshot
    IBKR: tutto il resto della persistenza per-utente passa da
    `_current_user_key()`."""
    user = session.get("user")
    if not isinstance(user, dict):
        return None
    email = (user.get("email") or "").strip().lower()
    return email or None


def _ibkr_default_owner_email() -> str:
    """Proprietario dello snapshot quando la richiesta non ne indica uno.
    IBKR_SYNC_USER_EMAIL se c'è, altrimenti il primo indirizzo di
    ADMIN_EMAILS."""
    explicit = (os.getenv("IBKR_SYNC_USER_EMAIL") or "").strip().lower()
    if explicit:
        return explicit
    admins = (os.getenv("ADMIN_EMAILS") or "").strip()
    if not admins:
        return ""
    return admins.split(",")[0].strip().lower()


def _ibkr_sync_authorized() -> bool:
    """Il job schedulato gira senza cookie di sessione, quindi si autentica con
    un bearer token condiviso. Se IBKR_SYNC_TOKEN non è configurato l'endpoint
    resta chiuso: meglio una sync che non parte di un endpoint di scrittura
    aperto a chiunque."""
    expected = (os.getenv("IBKR_SYNC_TOKEN") or "").strip()
    if not expected:
        return False
    header = (request.headers.get("Authorization") or "").strip()
    token = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not token:
        token = (request.headers.get("X-Polaris-Token") or "").strip()
    if not token:
        return False
    return hmac.compare_digest(token, expected)


# ---------------------------------------------------------------------------
# Traduzione simbolo IBKR → simbolo FMP
# ---------------------------------------------------------------------------
# IBKR nomina gli strumenti per borsa di quotazione, FMP per suffisso Yahoo:
# Grifols è `GRF` su BM e `GRF.MC` su FMP, CSG NV è `CSG1` su AEB e `CSG1.AS`
# su FMP. Il simbolo nudo non basta e nemmeno è univoco — `GRF` da solo su IBKR
# risolve a dieci strumenti diversi in cinque paesi.

_IBKR_EXCHANGE_SUFFIX = {
    # Stati Uniti: FMP usa il simbolo nudo
    "NASDAQ": "", "NYSE": "", "AMEX": "", "ARCA": "", "BATS": "", "IEX": "", "PINK": "",
    # Europa
    "AEB": ".AS", "FTA": ".AS",
    "BVME": ".MI", "BVME.ETF": ".MI",
    "BM": ".MC", "MEFFRV": ".MC",
    "IBIS": ".DE", "IBIS2": ".DE", "XETRA": ".DE", "FWB": ".DE", "SWB": ".DE", "GETTEX": ".DE",
    "SBF": ".PA", "ENEXT.BE": ".BR",
    "EBS": ".SW", "SWX": ".SW", "VSE": ".VI",
    "LSE": ".L", "LSEETF": ".L", "LSEIOB1": ".L",
    "SFB": ".ST", "OMXNO": ".OL", "CPH": ".CO", "HEX": ".HE",
    # Resto del mondo
    "NSE": ".NS", "BSE": ".BO", "TSEJ": ".T", "SEHK": ".HK", "ASX": ".AX", "TSE": ".TO",
}

_IBKR_COUNTRY_SUFFIXES = {
    "US": [""], "NL": [".AS"], "IT": [".MI"], "ES": [".MC"], "DE": [".DE"],
    "FR": [".PA"], "BE": [".BR"], "CH": [".SW"], "AT": [".VI"], "GB": [".L"],
    "IN": [".NS", ".BO"], "CA": [".TO"], "JP": [".T"], "HK": [".HK"], "AU": [".AX"],
    "SE": [".ST"], "NO": [".OL"], "DK": [".CO"], "FI": [".HE"],
}

# Ultima spiaggia quando non conosciamo né borsa né paese. L'euro è ambiguo per
# definizione, quindi elenca i mercati su cui l'app già lavora, in ordine di
# probabilità.
_IBKR_CURRENCY_SUFFIXES = {
    "USD": [""], "EUR": [".MI", ".DE", ".AS", ".PA", ".MC", ".BR", ".VI"],
    "GBP": [".L"], "CHF": [".SW"], "INR": [".NS"], "CAD": [".TO"],
    "JPY": [".T"], "HKD": [".HK"], "AUD": [".AX"],
    "SEK": [".ST"], "NOK": [".OL"], "DKK": [".CO"],
}

_IBKR_KNOWN_SUFFIXES = {s for s in _IBKR_EXCHANGE_SUFFIX.values() if s}

_IBKR_SYMBOL_OVERRIDES_CACHE: Optional[Dict[str, str]] = None


def _ibkr_symbol_overrides() -> Dict[str, str]:
    """Mappa esplicita simbolo IBKR → simbolo FMP, per i casi che nessuna
    regola per suffisso può indovinare: su Borsa Italiana IBKR numera parte dei
    ticker (Amplifon è `AMP2`) mentre FMP la chiama `AMP.MI`.

    Si estende senza toccare il codice con
    IBKR_SYMBOL_MAP="AMP2=AMP.MI,XYZ1=XYZ.MI".
    """
    global _IBKR_SYMBOL_OVERRIDES_CACHE
    if _IBKR_SYMBOL_OVERRIDES_CACHE is not None:
        return _IBKR_SYMBOL_OVERRIDES_CACHE
    overrides = {"AMP2": "AMP.MI"}
    raw = (os.getenv("IBKR_SYMBOL_MAP") or "").strip()
    for chunk in raw.split(","):
        if "=" not in chunk:
            continue
        src, _, dst = chunk.partition("=")
        src, dst = src.strip().upper(), dst.strip().upper()
        if src and dst:
            overrides[src] = dst
    _IBKR_SYMBOL_OVERRIDES_CACHE = overrides
    return overrides


def _ibkr_fmp_candidates(symbol: str, currency=None, exchange=None, country=None) -> List[str]:
    """Simboli FMP da provare per uno strumento IBKR, dal più al meno probabile.

    L'ordine conta: la borsa di quotazione è un'informazione certa, il paese
    quasi, la valuta è solo un indizio. Chi chiama si ferma al primo candidato
    su cui FMP risponde.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return []
    override = _ibkr_symbol_overrides().get(sym)
    if override:
        return [override]
    # Già in forma FMP (arriva così se la sync lo ha risolto a monte).
    if "." in sym and ("." + sym.rsplit(".", 1)[1]) in _IBKR_KNOWN_SUFFIXES:
        return [sym]

    suffixes: List[str] = []
    ex = (exchange or "").strip().upper()
    # IBKR qualifica le borse con un sotto-codice di comparto — `NASDAQ.NMS`,
    # `BVME.ETF`, `LSEIOB1` — che non cambia il mercato di quotazione. Si prova
    # il codice intero e poi la radice, così non serve enumerarli tutti.
    for key in (ex, ex.split(".", 1)[0]):
        if key in _IBKR_EXCHANGE_SUFFIX:
            suffixes.append(_IBKR_EXCHANGE_SUFFIX[key])
            break
    for source in (_IBKR_COUNTRY_SUFFIXES.get((country or "").strip().upper()),
                   _IBKR_CURRENCY_SUFFIXES.get((currency or "").strip().upper())):
        for suffix in (source or []):
            if suffix not in suffixes:
                suffixes.append(suffix)
    if not suffixes:
        suffixes = [""]

    candidates: List[str] = []
    for suffix in suffixes:
        for base in (sym, sym.rstrip("0123456789")):
            # La cifra finale è una convenzione IBKR di Borsa Italiana: non
            # provare a sfilarla altrove, spezzerebbe simboli legittimi.
            if base != sym and suffix != ".MI":
                continue
            if not base:
                continue
            candidate = base + suffix
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates[:6]


# ---------------------------------------------------------------------------
# Calendario earnings (FMP)
# ---------------------------------------------------------------------------

_EARNINGS_CACHE: Dict[str, dict] = {}
_EARNINGS_CACHE_TTL_SECONDS = int((os.getenv("EARNINGS_CACHE_TTL") or "21600").strip() or 21600)


def _fetch_next_earnings(symbol: str, currency=None, exchange=None, country=None) -> Optional[dict]:
    """Prossima trimestrale di uno strumento IBKR, o None se FMP non lo copre.

    Cache in memoria a 6h: il calendario si muove di rado e la pagina
    portafoglio richiede gli stessi ~20 simboli a ogni caricamento.
    """
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    cache_key = f"{sym}|{currency or ''}|{exchange or ''}|{country or ''}"
    cached = _EARNINGS_CACHE.get(cache_key)
    now = time.time()
    if cached and (now - cached.get("ts", 0)) < _EARNINGS_CACHE_TTL_SECONDS:
        return cached.get("value")

    today_iso = _ibkr_local_today().isoformat()
    value = None
    for candidate in _ibkr_fmp_candidates(sym, currency, exchange, country):
        rows = _fmp_get("earnings", symbol=candidate, limit=16)
        if not isinstance(rows, list) or not rows:
            continue
        upcoming = sorted(
            (r for r in rows
             if isinstance(r, dict) and isinstance(r.get("date"), str) and r["date"] >= today_iso),
            key=lambda r: r["date"],
        )
        if not upcoming:
            # FMP conosce il titolo ma non ha ancora una data futura: è un esito
            # legittimo (tipico degli ETC, che earnings non ne hanno), non un
            # fallimento di lookup — quindi ci si ferma qui invece di provare
            # altri suffissi e rischiare di agganciare un omonimo.
            value = {"date": None, "fmp_symbol": candidate,
                     "eps_estimated": None, "revenue_estimated": None}
            break
        nxt = upcoming[0]
        value = {
            "date": nxt.get("date"),
            "fmp_symbol": candidate,
            "eps_estimated": nxt.get("epsEstimated"),
            "revenue_estimated": nxt.get("revenueEstimated"),
        }
        break

    _EARNINGS_CACHE[cache_key] = {"ts": now, "value": value}
    return value


def _ibkr_local_today() -> _dt.date:
    """Oggi nel fuso di chi guarda la dashboard. La notifica parte alle 20:00
    italiane e parla del "giorno dopo": va calcolato su Europe/Rome, non su
    UTC né sull'ora della macchina che serve la richiesta."""
    if ZoneInfo is not None:
        try:
            return _dt.datetime.now(tz=ZoneInfo("Europe/Rome")).date()
        except Exception:
            pass
    return _dt.date.today()


def _ibkr_alert_target_date(reference: Optional[_dt.date] = None) -> _dt.date:
    """Il giorno da controllare: domani, o il primo giorno feriale successivo se
    domani cade nel weekend. Le trimestrali di sabato non esistono, e una
    notifica del venerdì sera che dice "nessun earning domani" non informa di
    nulla — meglio che guardi al lunedì."""
    target = (reference or _ibkr_local_today()) + _dt.timedelta(days=1)
    while target.weekday() >= 5:  # 5=sabato, 6=domenica
        target += _dt.timedelta(days=1)
    return target


# ---------------------------------------------------------------------------
# Normalizzazione del payload di sync
# ---------------------------------------------------------------------------

# Borse che quotano in sottomultipli pur dichiarando la valuta principale: il
# London Stock Exchange prezza in penny ma riporta GBP, e lo stesso fanno
# Johannesburg in centesimi e Tel Aviv in agorot. Il prezzo va diviso per cento,
# i controvalori no — IBKR quelli li dà già nella valuta piena.
# Senza questa correzione un ordine di 36 SSLN a 46,53 sterline risultava da
# 167.508 sterline, cioè otto volte il conto.
_IBKR_SUBUNIT_VENUES = (
    ("LSE", "GBP", 0.01),
    ("JSE", "ZAR", 0.01),
    ("TASE", "ILS", 0.01),
)


def _ibkr_price_scale(exchange: Optional[str], currency: Optional[str]) -> float:
    """Fattore per portare un prezzo IBKR nella valuta dichiarata."""
    ex = (exchange or "").strip().upper()
    cur = (currency or "").strip().upper()
    for prefix, expected_currency, scale in _IBKR_SUBUNIT_VENUES:
        if cur == expected_currency and ex.startswith(prefix):
            return scale
    return 1.0


def _ibkr_num(value) -> Optional[float]:
    """float() tollerante: IBKR manda le quantità come stringhe e i prezzi
    assenti come stringa vuota."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ibkr_str(value, limit: int = 120) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text[:limit] if text else None


def _ibkr_order_symbol(row: dict) -> Optional[str]:
    explicit = _ibkr_str(row.get("symbol"), 24)
    if explicit:
        return explicit.upper()
    for field in ("description", "primary_description"):
        match = _IBKR_ORDER_DESC_RE.match(str(row.get(field) or ""))
        if match:
            return match.group(1).upper()
    return None


def _ibkr_normalize_position(row: dict) -> Optional[dict]:
    symbol = _ibkr_str(row.get("symbol") or row.get("contract_description"), 24)
    if not symbol:
        return None
    currency = (_ibkr_str(row.get("currency"), 8) or "").upper() or None
    exchange = (_ibkr_str(row.get("exchange"), 16) or "").upper() or None
    scale = _ibkr_price_scale(exchange, currency)

    def price(value):
        number = _ibkr_num(value)
        return number * scale if number is not None else None

    return {
        "symbol": symbol.upper(),
        "name": _ibkr_str(row.get("name")),
        "quantity": _ibkr_num(row.get("quantity") if row.get("quantity") is not None
                              else row.get("position")),
        "avg_price": price(row.get("avg_price") or row.get("average_price")),
        "market_price": price(row.get("market_price")),
        # Il controvalore IBKR lo dà già in valuta piena: non va riscalato.
        "market_value": _ibkr_num(row.get("market_value")),
        "unrealized_pnl": _ibkr_num(row.get("unrealized_pnl")),
        "daily_pnl": _ibkr_num(row.get("daily_pnl")),
        "currency": (_ibkr_str(row.get("currency"), 8) or "").upper() or None,
        "exchange": (_ibkr_str(row.get("exchange"), 16) or "").upper() or None,
        "country": (_ibkr_str(row.get("country") or row.get("country_code"), 4) or "").upper() or None,
        "asset_class": (_ibkr_str(row.get("asset_class"), 8) or "").upper() or None,
    }


def _ibkr_normalize_order(row: dict) -> Optional[dict]:
    symbol = _ibkr_order_symbol(row)
    if not symbol:
        return None
    status = (_ibkr_str(row.get("status") or row.get("order_status"), 24) or "").upper()
    currency = (_ibkr_str(row.get("currency"), 8) or "").upper() or None
    exchange = (_ibkr_str(row.get("exchange"), 16) or "").upper() or None
    scale = _ibkr_price_scale(exchange, currency)

    def price(value):
        number = _ibkr_num(value)
        return number * scale if number is not None else None

    return {
        "order_id": _ibkr_str(row.get("order_id"), 32),
        "symbol": symbol,
        "name": _ibkr_str(row.get("name")),
        "side": (_ibkr_str(row.get("side"), 8) or "").upper() or None,
        "order_type": (_ibkr_str(row.get("order_type"), 16) or "").upper() or None,
        "status": status or None,
        "is_live": status.replace("_", "").replace(" ", "") in _IBKR_LIVE_ORDER_STATUSES,
        "quantity": _ibkr_num(row.get("quantity") or row.get("total_shares_qty")),
        "remaining": _ibkr_num(row.get("remaining") or row.get("remaining_shares_qty")),
        "limit_price": price(row.get("limit_price")),
        "stop_price": price(row.get("stop_price")),
        "tif": (_ibkr_str(row.get("tif"), 8) or "").upper() or None,
        "detail": _ibkr_str(row.get("detail") or row.get("secondary_description")),
        "description": _ibkr_str(row.get("description") or row.get("primary_description")),
        "order_time": _ibkr_str(row.get("order_time"), 32),
        "currency": currency,
        "exchange": exchange,
        "country": (_ibkr_str(row.get("country") or row.get("country_code"), 4) or "").upper() or None,
    }


def _ibkr_normalize_payload(data: dict) -> dict:
    positions, orders = [], []
    for raw in (data.get("positions") or [])[:_IBKR_MAX_ROWS]:
        if isinstance(raw, dict):
            row = _ibkr_normalize_position(raw)
            # Una posizione chiusa oggi resta nell'elenco di IBKR con quantità
            # zero: non è una partecipazione e non va mostrata. Lo zero è
            # l'unico valore da escludere — le quantità negative sono short.
            if row and row.get("quantity"):
                positions.append(row)
    for raw in (data.get("orders") or [])[:_IBKR_MAX_ROWS]:
        if isinstance(raw, dict):
            row = _ibkr_normalize_order(raw)
            if row:
                orders.append(row)
    return {"positions": positions, "orders": orders}


# ---------------------------------------------------------------------------
# Arricchimento: strumenti distinti + date earnings
# ---------------------------------------------------------------------------

def _ibkr_instruments(snapshot: dict) -> List[dict]:
    """Strumenti distinti nello snapshot, con l'origine (posizione, ordine o
    entrambe) e i metadati che servono a risolverli su FMP.

    Gli ordini contano solo se ancora vivi: un earning su un ordine già
    rimpiazzato o cancellato non è un rischio aperto.
    """
    by_symbol: Dict[str, dict] = {}
    rows_by_kind = (
        ("position", snapshot.get("positions") or []),
        ("order", [o for o in (snapshot.get("orders") or [])
                   if isinstance(o, dict) and o.get("is_live")]),
    )
    for kind, rows in rows_by_kind:
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            entry = by_symbol.setdefault(symbol, {
                "symbol": symbol, "sources": [],
                "name": None, "currency": None, "exchange": None, "country": None,
            })
            if kind not in entry["sources"]:
                entry["sources"].append(kind)
            # Il primo valore non vuoto vince: le posizioni portano valuta e
            # borsa, gli ordini spesso no.
            for field in ("name", "currency", "exchange", "country"):
                if not entry.get(field) and row.get(field):
                    entry[field] = row[field]
    return sorted(by_symbol.values(), key=lambda e: e["symbol"])


def _ibkr_earnings_map(snapshot: dict) -> Dict[str, dict]:
    """{simbolo IBKR: prossima trimestrale}. Le lookup FMP vanno in parallelo,
    stesso schema di /api/portfolio: una ventina di simboli su 8 thread."""
    instruments = _ibkr_instruments(snapshot)
    if not instruments:
        return {}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(
            lambda i: _fetch_next_earnings(
                i["symbol"], i.get("currency"), i.get("exchange"), i.get("country")),
            instruments,
        ))
    return {i["symbol"]: r for i, r in zip(instruments, results) if r}


def _ibkr_enriched_snapshot(snapshot: dict, earnings: Dict[str, dict]) -> dict:
    """Snapshot pronto per la pagina: ogni riga porta la sua prossima data
    earnings e i giorni che mancano."""
    today = _ibkr_local_today()

    def decorate(row: dict) -> dict:
        info = earnings.get((row.get("symbol") or "").upper()) or {}
        date_iso = info.get("date")
        days = None
        if date_iso:
            try:
                days = (_dt.date.fromisoformat(date_iso) - today).days
            except ValueError:
                days = None
        return {
            **row,
            "earnings_date": date_iso,
            "earnings_in_days": days,
            "earnings_eps_estimated": info.get("eps_estimated"),
            "fmp_symbol": info.get("fmp_symbol"),
        }

    return {
        "positions": [decorate(r) for r in (snapshot.get("positions") or [])],
        "orders": [decorate(r) for r in (snapshot.get("orders") or [])],
    }


# ---------------------------------------------------------------------------
# Cambi: servono per ordinare per importo investito
# ---------------------------------------------------------------------------
# Le posizioni sono in valute diverse, quindi confrontare i controvalori nudi
# metterebbe 4.760 EUR sotto 4.900 USD. Si converte tutto nella valuta base del
# conto prima di ordinare.

_FX_CACHE: Dict[str, dict] = {}
_FX_CACHE_TTL_SECONDS = 21600  # 6h: per un ordinamento non serve di meglio


def _ibkr_base_currency() -> str:
    return (_ibkr_api_env("IBKR_BASE_CURRENCY", "EUR") or "EUR").upper()


def _fx_to_base(currency: Optional[str]) -> Optional[float]:
    """Quante unità di valuta base vale una unità di `currency`.

    None quando il cambio non si riesce a stabilire: chi chiama deve poterlo
    distinguere da 1.0, altrimenti una posizione in dollari finirebbe ordinata
    come se fosse già in euro.
    """
    cur = (currency or "").strip().upper()
    base = _ibkr_base_currency()
    if not cur:
        return None
    if cur == base:
        return 1.0
    cached = _FX_CACHE.get(cur)
    now = time.time()
    if cached and (now - cached["ts"]) < _FX_CACHE_TTL_SECONDS:
        return cached["rate"]
    rate = None
    data = _fmp_get("quote", symbol=f"{cur}{base}")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        rate = data[0].get("price")
    if not rate:
        # Prova il verso opposto: FMP non quota tutte le coppie in entrambi i sensi.
        data = _fmp_get("quote", symbol=f"{base}{cur}")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            inverse = data[0].get("price")
            rate = (1.0 / inverse) if inverse else None
    _FX_CACHE[cur] = {"ts": now, "rate": rate}
    return rate


def _ibkr_order_capital_base(order: dict, fallback_currency: Optional[str] = None) -> Optional[float]:
    """Capitale che un ordine di acquisto impegnerebbe, in valuta base.

    Solo gli ordini in acquisto: una vendita su un titolo che già si possiede è
    un'uscita, non un impiego di capitale. Il prezzo è il limite, o lo stop
    quando limite non c'è.
    """
    if (order.get("side") or "").upper() != "BUY":
        return None
    quantity = order.get("remaining")
    if quantity is None:
        quantity = order.get("quantity")
    price = order.get("limit_price")
    if price is None:
        price = order.get("stop_price")
    if not quantity or price is None:
        return None
    rate = _fx_to_base(order.get("currency") or fallback_currency)
    return abs(quantity) * price * rate if rate else None


def _ibkr_market_value_base(row: dict) -> Optional[float]:
    """Controvalore nella valuta base. Preferisce il cambio che IBKR allega
    alla posizione (il Flex lo espone come `fxRateToBase`), altrimenti lo
    chiede a FMP."""
    value = row.get("market_value")
    if value is None:
        return None
    rate = row.get("fx_rate_to_base") or _fx_to_base(row.get("currency"))
    return value * rate if rate else None


def _ibkr_earnings_alert(snapshot: dict, target: Optional[_dt.date] = None,
                         earnings: Optional[Dict[str, dict]] = None) -> dict:
    """Strumenti dello snapshot che riportano il giorno indicato (di default
    domani). Ogni voce porta con sé posizione e ordini vivi, perché la
    notifica dica non solo *cosa* riporta ma *quanto* ci si è esposti."""
    target_date = target or _ibkr_alert_target_date()
    target_iso = target_date.isoformat()
    if earnings is None:
        earnings = _ibkr_earnings_map(snapshot)

    positions_by_symbol = {(p.get("symbol") or "").upper(): p
                           for p in (snapshot.get("positions") or [])}
    orders_by_symbol: Dict[str, List[dict]] = {}
    for order in (snapshot.get("orders") or []):
        if not order.get("is_live"):
            continue
        orders_by_symbol.setdefault((order.get("symbol") or "").upper(), []).append(order)

    items, unresolved = [], []
    for instrument in _ibkr_instruments(snapshot):
        symbol = instrument["symbol"]
        info = earnings.get(symbol)
        if not info:
            unresolved.append(symbol)
            continue
        if info.get("date") != target_iso:
            continue
        items.append({
            "symbol": symbol,
            "name": instrument.get("name"),
            "fmp_symbol": info.get("fmp_symbol"),
            "date": info.get("date"),
            "eps_estimated": info.get("eps_estimated"),
            "revenue_estimated": info.get("revenue_estimated"),
            "sources": instrument.get("sources") or [],
            "position": positions_by_symbol.get(symbol),
            "orders": orders_by_symbol.get(symbol, []),
        })

    return {
        "target_date": target_iso,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "count": len(items),
        "items": items,
        # Simboli che FMP non copre: dichiararli evita che un "nessun earning
        # domani" nasconda un titolo semplicemente non risolto.
        "unresolved": unresolved,
    }


# ---------------------------------------------------------------------------
# Resa del messaggio (Telegram + email)
# ---------------------------------------------------------------------------

_IBKR_WEEKDAYS_IT = ("lunedì", "martedì", "mercoledì", "giovedì",
                     "venerdì", "sabato", "domenica")


def _ibkr_format_date_it(date_iso: Optional[str]) -> str:
    if not date_iso:
        return "—"
    try:
        day = _dt.date.fromisoformat(date_iso)
    except (TypeError, ValueError):
        return date_iso
    return f"{_IBKR_WEEKDAYS_IT[day.weekday()]} {day.day:02d}/{day.month:02d}/{day.year}"


def _ibkr_order_line(order: dict) -> str:
    price = order.get("limit_price")
    if price is None:
        price = order.get("stop_price")
    qty = order.get("remaining") or order.get("quantity")
    parts = [
        (order.get("side") or "?"),
        f"{qty:g}" if isinstance(qty, float) else str(qty or "?"),
        (order.get("order_type") or "").title() or "?",
    ]
    if price is not None:
        parts.append(f"@ {price:g}")
    if order.get("tif"):
        parts.append(order["tif"])
    return " ".join(p for p in parts if p)


def _ibkr_alert_telegram_text(alert: dict) -> str:
    """Messaggio Telegram in HTML (parse_mode=HTML): niente Markdown, che
    inciampa sui punti e sugli underscore dei ticker.

    Si escapano solo & < >, cioè i tre caratteri che Telegram documenta: con
    `quote=True` gli apostrofi diventerebbero `&#x27;` e Telegram le entità
    numeriche non le converte, quindi si leggerebbero tali e quali.
    """
    def esc(value):
        return _html.escape(str(value), quote=False)
    header = f"📅 <b>Earnings {_ibkr_format_date_it(alert.get('target_date'))}</b>"
    if not alert.get("count"):
        lines = [header, "", "Nessun titolo in portafoglio o con ordini pendenti riporta."]
    else:
        lines = [header, ""]
        for item in alert["items"]:
            label = esc(item["symbol"])
            if item.get("name"):
                label += f" — {esc(item['name'])}"
            lines.append(f"🔸 <b>{label}</b>")
            position = item.get("position")
            if position and position.get("quantity"):
                bits = [f"posizione {position['quantity']:g}"]
                if position.get("market_value") is not None:
                    bits.append(f"{position['market_value']:,.0f} {position.get('currency') or ''}".strip())
                if position.get("unrealized_pnl") is not None:
                    bits.append(f"P&L {position['unrealized_pnl']:+,.0f}")
                lines.append("   " + esc(" · ".join(bits)))
            for order in item.get("orders") or []:
                lines.append(f"   ⏳ {esc(_ibkr_order_line(order))}")
            eps = item.get("eps_estimated")
            if eps is not None:
                lines.append("   EPS atteso " + esc(f"{eps:g}"))
        lines.append("")
        lines.append("<i>Polaris — controlla stop e size prima della chiusura.</i>")
    if alert.get("unresolved"):
        lines.append("")
        lines.append("⚠️ Senza calendario earnings: " + esc(", ".join(alert["unresolved"])))
    stale = _ibkr_stale_orders_note(alert)
    if stale:
        lines.append("")
        lines.append("🕒 " + esc(stale))
    return "\n".join(lines)


def _ibkr_alert_email_html(alert: dict) -> str:
    """Corpo HTML della mail. Stili inline: i client di posta ignorano <style>
    e la palette è quella della dashboard."""
    esc = _html.escape
    target_label = _ibkr_format_date_it(alert.get("target_date"))
    rows = []
    for item in alert.get("items") or []:
        position = item.get("position") or {}
        exposure = "—"
        if position.get("quantity"):
            exposure = f"{position['quantity']:g} pz"
            if position.get("market_value") is not None:
                exposure += f" · {position['market_value']:,.0f} {position.get('currency') or ''}".rstrip()
        orders = item.get("orders") or []
        orders_html = "<br>".join(esc(_ibkr_order_line(o)) for o in orders) or "—"
        eps = item.get("eps_estimated")
        rows.append(
            "<tr>"
            f'<td style="padding:10px 12px;border-bottom:1px solid #2b3d55;'
            f'font-weight:700;color:#fff;">{esc(item["symbol"])}'
            + (f'<div style="font-weight:400;font-size:12px;color:#adb5bd;">'
               f'{esc(item.get("name") or "")}</div>' if item.get("name") else "")
            + "</td>"
            f'<td style="padding:10px 12px;border-bottom:1px solid #2b3d55;color:#e2e8f0;">{esc(exposure)}</td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid #2b3d55;color:#e2e8f0;">{orders_html}</td>'
            f'<td style="padding:10px 12px;border-bottom:1px solid #2b3d55;color:#e2e8f0;">'
            f'{esc(f"{eps:g}") if eps is not None else "—"}</td>'
            "</tr>"
        )

    if rows:
        body = (
            '<table role="presentation" cellpadding="0" cellspacing="0" '
            'style="width:100%;border-collapse:collapse;font-size:14px;">'
            '<tr style="text-align:left;color:#778da9;font-size:11px;'
            'text-transform:uppercase;letter-spacing:0.05em;">'
            '<th style="padding:8px 12px;">Titolo</th>'
            '<th style="padding:8px 12px;">Posizione</th>'
            '<th style="padding:8px 12px;">Ordini pendenti</th>'
            '<th style="padding:8px 12px;">EPS atteso</th></tr>'
            + "".join(rows) + "</table>"
        )
    else:
        body = ('<p style="color:#adb5bd;font-size:14px;">Nessun titolo in portafoglio '
                'o con ordini pendenti riporta in questa data.</p>')

    warning = ""
    if alert.get("unresolved"):
        warning = (
            '<p style="color:#fbbf24;font-size:12px;margin-top:16px;">⚠️ Senza calendario '
            f'earnings su FMP: {esc(", ".join(alert["unresolved"]))}</p>'
        )
    stale = _ibkr_stale_orders_note(alert)
    if stale:
        warning += (
            '<p style="color:#fbbf24;font-size:12px;margin-top:10px;">🕒 '
            f'{esc(stale)}</p>'
        )

    return (
        '<div style="background:#0d1b2a;padding:24px;font-family:-apple-system,'
        'BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">'
        '<div style="max-width:680px;margin:0 auto;background:#1b263b;'
        'border:1px solid #415a77;border-radius:14px;padding:20px 22px;">'
        '<div style="font-size:13px;color:#778da9;letter-spacing:0.08em;'
        'text-transform:uppercase;">Polaris · Interactive Brokers</div>'
        f'<h2 style="color:#fff;margin:6px 0 18px;font-size:20px;">Earnings {esc(target_label)}</h2>'
        + body + warning +
        '<p style="color:#6b7e92;font-size:11px;margin-top:20px;">Generata dal job '
        'delle 20:00 su posizioni aperte e ordini ancora eseguibili.</p>'
        '</div></div>'
    )


def _ibkr_alert_subject(alert: dict) -> str:
    label = _ibkr_format_date_it(alert.get("target_date"))
    count = alert.get("count") or 0
    if not count:
        return f"Polaris · nessun earning {label}"
    symbols = ", ".join(i["symbol"] for i in alert["items"][:5])
    if count > 5:
        symbols += f" +{count - 5}"
    return f"Polaris · earnings {label}: {symbols}"


def _ibkr_alert_with_rendering(alert: dict, orders_freshness: Optional[dict] = None) -> dict:
    """Aggiunge all'alert i tre formati pronti da spedire, così chi lo riceve
    non deve reimpaginare niente.

    `orders_freshness` viene dall'ibrido Flex + gateway: se gli ordini sono
    vecchi il messaggio deve dirlo, perché un alert costruito su ordini
    stantii ha esattamente lo stesso aspetto di uno costruito su ordini veri.
    """
    enriched = {**alert, "orders_freshness": orders_freshness}
    return {
        **enriched,
        "subject": _ibkr_alert_subject(enriched),
        "telegram_text": _ibkr_alert_telegram_text(enriched),
        "email_html": _ibkr_alert_email_html(enriched),
    }


def _ibkr_stale_orders_note(alert: dict) -> Optional[str]:
    """Avviso da mostrare quando la lista ordini non è aggiornata."""
    freshness = alert.get("orders_freshness") or {}
    if not freshness.get("stale"):
        return None
    reason = freshness.get("reason") or "aggiornamento sconosciuto"
    return f"Ordini pendenti non aggiornati ({reason}): l'alert copre le sole posizioni."


def _ibkr_maybe_notify(alert: dict, data: dict) -> dict:
    """Manda l'alert su Telegram se c'è qualcosa da dire.

    Un "nessun earning domani" spedito ogni sera è una notifica che si impara a
    ignorare, e la sera in cui ce n'è uno vero non la si legge. Con
    `notify_always` si forza l'invio comunque — utile per verificare che il
    canale funzioni.
    """
    if not data.get("notify") and not data.get("notify_always"):
        return {"sent": False, "error": "notify non richiesto"}
    if not alert.get("count") and not data.get("notify_always"):
        return {"sent": False, "error": "nessun earning da segnalare"}
    return _telegram_send(alert["telegram_text"])


def _telegram_send(text: str) -> dict:
    """Manda un messaggio al bot Telegram configurato. Non solleva mai: la
    notifica è un canale accessorio, un token scaduto non deve far fallire la
    sync che l'ha innescata."""
    token = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
    chat_id = (os.getenv("TELEGRAM_CHAT_ID") or "").strip()
    if not token or not chat_id:
        return {"sent": False, "error": "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID non configurati"}
    try:
        payload = urllib.parse.urlencode({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = urllib.request.Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=payload,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "User-Agent": "polaris/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("ok"):
            return {"sent": True, "error": None}
        return {"sent": False, "error": _ibkr_str(body.get("description"), 200)}
    except Exception as e:
        return {"sent": False, "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Rotte
# ---------------------------------------------------------------------------

def _ibkr_load_snapshot(owner_email: str) -> Optional[dict]:
    coll = _get_mongo_ibkr_collection()
    if coll is None or not owner_email:
        return None
    try:
        return coll.find_one({"owner_email": owner_email}, {"_id": 0})
    except Exception:
        return None


# Oltre questa soglia gli ordini vengono dichiarati vecchi. Il gateway locale
# gira solo quando il PC è acceso, quindi capita che la lista non si aggiorni
# per un giorno: va detto, perché un alert calcolato su ordini stantii sembra
# aggiornato quanto uno vero.
_IBKR_ORDERS_STALE_AFTER_SECONDS = int(
    (os.getenv("IBKR_ORDERS_STALE_AFTER") or "129600").strip() or 129600)  # 36h

# Le posizioni invecchiano molto prima degli ordini: un ordine di ieri
# probabilmente è ancora lì, una posizione di ieri può essere stata aperta o
# chiusa stamattina all'apertura. Sei ore coprono una mattinata di borsa.
_IBKR_POSITIONS_STALE_AFTER_SECONDS = int(
    (os.getenv("IBKR_POSITIONS_STALE_AFTER") or "21600").strip() or 21600)  # 6h

# Il P&L giornaliero di IBKR invecchia più in fretta di tutto il resto: è un
# numero che si muove di continuo, e mostrarlo vecchio di ore come "oggi" è
# peggio che stimarlo.
_IBKR_DAILY_PNL_MAX_AGE_SECONDS = int(
    (os.getenv("IBKR_DAILY_PNL_MAX_AGE") or "7200").strip() or 7200)  # 2h


def _ibkr_positions_staleness(doc: Optional[dict]) -> dict:
    """Da quanto non si aggiornano le posizioni.

    Serve quanto quella degli ordini, e per lo stesso motivo: uno snapshot di
    ieri sera ha lo stesso aspetto di uno di adesso, ma se nel frattempo sono
    scattati dei limit il portafoglio che si sta guardando non è quello che si
    ha. Sotto le sei ore si tace, sopra si dichiara.
    """
    if not doc or not doc.get("positions"):
        return {"synced_at": None, "age_seconds": None, "stale": False, "source": None}
    synced_at = doc.get("positions_synced_at") or doc.get("synced_at")
    if not synced_at:
        return {"synced_at": None, "age_seconds": None, "stale": True,
                "source": doc.get("positions_source"), "reason": "data ignota"}
    age = max(0.0, time.time() - float(synced_at))
    stale = age > _IBKR_POSITIONS_STALE_AFTER_SECONDS
    return {"synced_at": synced_at, "age_seconds": age, "stale": stale,
            "source": doc.get("positions_source"),
            "reason": (f"ultimo aggiornamento {age / 3600:.0f}h fa" if stale else None)}


def _ibkr_orders_staleness(doc: Optional[dict]) -> dict:
    """Da quanto non arrivano gli ordini, e se è troppo."""
    if not doc:
        return {"synced_at": None, "age_seconds": None, "stale": True,
                "source": None, "reason": "nessuno snapshot"}
    synced_at = doc.get("orders_synced_at") or doc.get("synced_at")
    if not doc.get("orders"):
        return {"synced_at": synced_at, "age_seconds": None, "stale": True,
                "source": doc.get("orders_source"),
                "reason": "nessun ordine mai ricevuto"}
    if not synced_at:
        return {"synced_at": None, "age_seconds": None, "stale": True,
                "source": doc.get("orders_source"), "reason": "data ignota"}
    age = max(0.0, time.time() - float(synced_at))
    stale = age > _IBKR_ORDERS_STALE_AFTER_SECONDS
    return {"synced_at": synced_at, "age_seconds": age, "stale": stale,
            "source": doc.get("orders_source"),
            "reason": f"ultimo aggiornamento {age / 3600:.0f}h fa" if stale else None}


def _ibkr_report_date_to_epoch(report_date: Optional[str]) -> Optional[float]:
    """Data di un report Flex (`YYYYMMDD` o `YYYY-MM-DD`) → epoch di fine
    giornata. None se assente o illeggibile, così chi chiama può ripiegare
    sull'istante corrente."""
    raw = (report_date or "").strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        return None
    try:
        day = _dt.date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
    except ValueError:
        return None
    return _dt.datetime.combine(day, _dt.time(23, 59), _dt.timezone.utc).timestamp()


def _ibkr_store_snapshot(owner_email: str, positions: Optional[List[dict]] = None,
                         orders: Optional[List[dict]] = None,
                         source: str = "sync",
                         positions_as_of: Optional[float] = None,
                         account: Optional[dict] = None) -> Optional[dict]:
    """Salva lo snapshot fondendo, non sostituendo.

    Posizioni e ordini arrivano da due sorgenti diverse e con ritmi diversi —
    il Flex Web Service copre solo le posizioni e gira sempre, il gateway
    locale porta anche gli ordini ma solo a PC acceso. Se ogni scrittura
    sostituisse il documento intero, il giro notturno del Flex cancellerebbe
    ogni sera gli ordini raccolti di giorno.

    `positions_as_of` è la data *del dato*, non della scrittura: il Flex
    fotografa la chiusura precedente, quindi una sua scrittura serale è più
    recente ma meno aggiornata di una del gateway fatta durante il giorno. Vince
    il dato più recente, non l'ultimo arrivato — altrimenti il cron delle 20:00
    riporterebbe indietro il portafoglio ogni sera.

    Ritorna lo snapshot risultante, o None se Mongo non è disponibile.
    """
    coll = _get_mongo_ibkr_collection()
    if coll is None or not owner_email:
        return None
    existing = _ibkr_load_snapshot(owner_email) or {}
    now = time.time()

    merged = {
        "positions": existing.get("positions") or [],
        "orders": existing.get("orders") or [],
    }
    update = {"owner_email": owner_email, "synced_at": now}
    skipped = None
    if positions is not None:
        incoming_as_of = positions_as_of if positions_as_of is not None else now
        existing_as_of = existing.get("positions_as_of")
        if existing_as_of is not None and incoming_as_of < existing_as_of:
            skipped = {
                "reason": "posizioni ignorate: il dato in archivio è più recente",
                "incoming_as_of": incoming_as_of,
                "existing_as_of": existing_as_of,
                "existing_source": existing.get("positions_source"),
            }
            positions = None
    if positions is not None:
        merged["positions"] = positions
        update.update({"positions": positions, "positions_synced_at": now,
                       "positions_source": source,
                       "positions_as_of": positions_as_of if positions_as_of is not None else now})
    if orders is not None:
        merged["orders"] = orders
        update.update({"orders": orders, "orders_synced_at": now,
                       "orders_source": source})

    if isinstance(account, dict):
        # Il net liquidation è il denominatore dell'esposizione: senza, la
        # percentuale non è calcolabile e la pagina lo dichiara invece di
        # inventarsi un totale.
        previous = existing.get("account") if isinstance(existing.get("account"), dict) else {}
        daily = _ibkr_num(account.get("daily_pnl"))
        merged_account = {
            "net_liquidation": _ibkr_num(account.get("net_liquidation")),
            "cash": _ibkr_num(account.get("cash")) or previous.get("cash"),
            "currency": ((_ibkr_str(account.get("currency"), 8) or "").upper()
                         or previous.get("currency")),
            # Il giornaliero lo porta solo il gateway: il Flex è di fine
            # giornata e non ce l'ha. Una scrittura dal Flex non deve
            # cancellarlo — è lo stesso principio delle posizioni, una fonte
            # meno informata non sovrascrive una più informata.
            "daily_pnl": daily if daily is not None else previous.get("daily_pnl"),
            "unrealized_pnl": (_ibkr_num(account.get("unrealized_pnl"))
                               or previous.get("unrealized_pnl")),
        }
        update["account"] = merged_account
        update["account_synced_at"] = now
        # Timestamp separato per il giornaliero: quello del conto lo rinfresca
        # anche il Flex, e farebbe sembrare fresco un P&L di ore prima.
        if daily is not None:
            update["daily_pnl_at"] = now

    # Le date earnings si ricalcolano sull'unione: un titolo può entrare nello
    # snapshot da una sorgente e uscirne dall'altra.
    earnings = _ibkr_earnings_map(merged)
    update["earnings"] = earnings
    try:
        coll.update_one({"owner_email": owner_email}, {"$set": update}, upsert=True)
    except Exception:
        return None
    return {**existing, **update, **merged, "skipped": skipped,
            # Cosa è stato scritto davvero, non cosa era stato proposto: le
            # posizioni possono essere state scartate perché più vecchie.
            "applied": [k for k in ("positions", "orders") if k in update]}


# ---------------------------------------------------------------------------
# Flex Web Service — solo posizioni, ma senza niente da tenere acceso
# ---------------------------------------------------------------------------
# È il ripiego all'OAuth: un token annuale e due GET, nessun gateway e nessun
# login. In cambio il Flex non espone gli ordini di lavoro e i dati sono di
# fine giornata — per gli ordini serve il gateway locale.

_FLEX_BASE = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

# IBKR riusa lo stesso testo generico per cause molto diverse, quindi conviene
# tradurre il codice in una diagnosi invece di rimandare il messaggio nudo.
_FLEX_ERROR_HINTS = {
    "1003": "statement non disponibile per il periodo scelto",
    "1012": "token scaduto: rigenerane uno dal Flex Web Service",
    "1014": "query id inesistente: controlla di aver copiato il numero della "
            "query e non il suo nome",
    "1015": "token non valido",
    "1016": "account non valido per questa query",
    "1017": "reference code non valido",
    "1019": "statement ancora in generazione",
    "1020": "richiesta non validata: le cause tipiche sono il Flex Web Service "
            "non abilitato (il token da solo non basta), un token appena creato "
            "e non ancora propagato, o una restrizione per indirizzo IP sul "
            "token — Vercel esce da IP variabili, quindi va tolta",
    "1021": "statement non recuperabile",
}


def _flex_get(url: str) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "polaris/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


def _flex_xml_text(xml: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.S | re.I)
    return match.group(1).strip() if match else None


_FLEX_ACTIVITY_CACHE: Dict[str, Any] = {}
_FLEX_ACTIVITY_TTL_SECONDS = int(
    (os.getenv("FLEX_ACTIVITY_TTL") or "14400").strip() or 14400)  # 4h


def _flex_attrs(xml: str, *element_names: str) -> List[dict]:
    """Attributi di ogni elemento con uno dei nomi indicati.

    Si accettano più nomi perché IBKR non è coerente tra tipi di query — gli
    eseguiti stanno in `<TradeConfirm>` nella Trade Confirmation e in `<Trade>`
    nell'Activity — e indovinare il nome sbagliato darebbe una lista vuota
    indistinguibile da "non c'è niente".
    """
    out = []
    for name in element_names:
        for match in re.finditer(rf"<{name}\b([^>]*?)/?>", xml, re.S):
            out.append(dict(re.findall(r'(\w+)="([^"]*)"', match.group(1))))
    return out


def _flex_fetch_statement(query_id: str) -> dict:
    """Scarica un report Flex. Ritorna {"xml"} oppure {"error"}.

    Il report non è pronto subito: si chiede la generazione, si riceve un
    reference code e si ripassa a ritirarlo.
    """
    token = _ibkr_api_env("IBKR_FLEX_TOKEN")
    if not token or not query_id:
        return {"error": "IBKR_FLEX_TOKEN o query id non configurati"}

    sent = _flex_get(f"{_FLEX_BASE}/SendRequest?"
                     + urllib.parse.urlencode({"t": token, "q": query_id, "v": "3"}))
    if not sent:
        return {"error": "Flex SendRequest irraggiungibile"}
    if (_flex_xml_text(sent, "Status") or "").lower() != "success":
        # Il codice conta più del messaggio: IBKR usa lo stesso testo generico
        # per token sbagliato, token non ancora attivo e query inesistente.
        code = _flex_xml_text(sent, "ErrorCode")
        message = _flex_xml_text(sent, "ErrorMessage") or sent[:200]
        return {"error": f"Flex SendRequest [{code or '?'}]: {message}",
                "error_code": code,
                "hint": _FLEX_ERROR_HINTS.get(code or "")}
    reference = _flex_xml_text(sent, "ReferenceCode")
    # L'elemento <Url> della risposta punta al vecchio servlet
    # /Universal/servlet/FlexStatementService, che IBKR documenta come legacy e
    # dice di ignorare: seguirlo è l'errore che si propaga da mezzo internet.
    base_url = f"{_FLEX_BASE}/GetStatement"
    if not reference:
        return {"error": "Flex: nessun ReferenceCode nella risposta"}

    statement = None
    for attempt in range(6):
        # Prima attesa più lunga: chiedere subito il report è quasi sempre
        # inutile e IBKR limita le richieste ravvicinate.
        time.sleep(4 if attempt == 0 else 6)
        body = _flex_get(base_url + "?"
                         + urllib.parse.urlencode({"t": token, "q": reference, "v": "3"}))
        if not body:
            continue
        if "<FlexQueryResponse" in body:
            statement = body
            break
        # Ancora in generazione: la risposta è di nuovo un FlexStatementResponse.
        if "<FlexStatementResponse" not in body:
            return {"error": "Flex GetStatement: risposta inattesa " + body[:200]}
    if statement is None:
        return {"error": "Flex: report non pronto dopo 6 tentativi"}
    return {"xml": statement}


def _flex_parse_activity(xml: str) -> dict:
    """Posizioni, capitale e cambi da un Activity Flex.

    Capitale e cambi ci sono solo se le rispettive sezioni sono state aggiunte
    alla query nel portale: la loro assenza non è un errore, è una query
    configurata più stretta.
    """
    positions = []
    for attrs in _flex_attrs(xml, "OpenPosition"):
        symbol = (attrs.get("symbol") or "").strip()
        if not symbol:
            continue
        currency = (attrs.get("currency") or "").upper() or None
        exchange = (attrs.get("listingExchange") or "").upper() or None
        scale = _ibkr_price_scale(exchange, currency)

        def price(value, _scale=scale):
            number = _ibkr_num(value)
            return number * _scale if number is not None else None

        positions.append({
            "symbol": symbol,
            "name": attrs.get("description") or None,
            "quantity": _ibkr_num(attrs.get("position")),
            "avg_price": price(attrs.get("costBasisPrice") or attrs.get("openPrice")),
            "market_price": price(attrs.get("markPrice")),
            "market_value": _ibkr_num(attrs.get("positionValue")),
            "unrealized_pnl": _ibkr_num(attrs.get("fifoPnlUnrealized")),
            "currency": currency,
            "asset_class": (attrs.get("assetCategory") or "").upper() or None,
            "exchange": exchange,
        })

    # Sezione "Net Asset Value (NAV) Summary in Base": l'ultima riga per data
    # di report è il capitale a quella chiusura.
    account = None
    nav_rows = _flex_attrs(xml, "EquitySummaryByReportDateInBase", "EquitySummaryInBase")
    nav_rows = [r for r in nav_rows if r.get("total") or r.get("cash")]
    if nav_rows:
        latest = max(nav_rows, key=lambda r: r.get("reportDate") or "")
        account = {
            "net_liquidation": _ibkr_num(latest.get("total")),
            "cash": _ibkr_num(latest.get("cash")),
            "currency": (latest.get("currency") or "").upper() or _ibkr_base_currency(),
        }

    # Sezione "Currency Conversion Rate": cambi ufficiali IBKR verso la valuta
    # base, preferibili a quelli chiesti a FMP.
    rates = {}
    for attrs in _flex_attrs(xml, "ConversionRate"):
        source = (attrs.get("fromCurrency") or "").upper()
        rate = _ibkr_num(attrs.get("rate"))
        if source and rate:
            rates[source] = rate

    report_date = re.search(r'<FlexStatement\b[^>]*\btoDate="([^"]*)"', xml)
    return {"positions": positions, "account": account, "rates": rates,
            "report_date": report_date.group(1) if report_date else None}


def _flex_trade_rows(xml: str) -> List[dict]:
    """Le righe degli eseguiti, una per esecuzione e non di più.

    Una Activity Flex a cui siano stati spuntati più livelli di dettaglio
    ripete lo stesso fill una volta per livello — `EXECUTION`, `ORDER`,
    `CLOSED_LOT` — e sommarli conterebbe due o tre volte lo stesso realizzato.
    Si tiene il livello più fine fra quelli presenti; i lotti chiusi si
    scartano sempre, perché ripartiscono un realizzato già contato altrove.
    """
    rows = _flex_attrs(xml, "TradeConfirm", "Trade")
    levels = {(row.get("levelOfDetail") or "").upper() for row in rows}
    for preferred in ("EXECUTION", "ORDER"):
        if preferred in levels:
            return [r for r in rows
                    if (r.get("levelOfDetail") or "").upper() == preferred]
    # Nessun livello dichiarato: è il caso della Trade Confirmation, dove le
    # righe sono già una per eseguito.
    return [r for r in rows
            if (r.get("levelOfDetail") or "").upper() not in ("CLOSED_LOT", "LOT")]


def _flex_trade_clock(attrs: dict) -> Optional[str]:
    """Ora dell'eseguito, `HH:MM`. IBKR la scrive `YYYYMMDD;HHMMSS` oppure
    `YYYY-MM-DD HH:MM:SS` a seconda del formato scelto nel portale."""
    raw = (attrs.get("dateTime") or attrs.get("orderTime") or "").strip()
    if not raw:
        return None
    match = re.search(r"(\d{2}):(\d{2})", raw)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    match = re.search(r"[;\s](\d{2})(\d{2})\d{2}\s*$", raw)
    return f"{match.group(1)}:{match.group(2)}" if match else None


def _flex_trade_from_attrs(attrs: dict) -> Optional[dict]:
    """Un eseguito dagli attributi della sua riga. None se la riga non lo è."""
    symbol = (attrs.get("symbol") or "").strip()
    quantity = _ibkr_num(attrs.get("quantity"))
    if not symbol or not quantity:
        return None
    currency = (attrs.get("currency") or "").upper() or None
    exchange = (attrs.get("listingExchange") or attrs.get("exchange") or "").upper() or None
    price = _ibkr_num(attrs.get("price") or attrs.get("tradePrice"))
    if price is not None:
        price *= _ibkr_price_scale(exchange, currency)
    # `quantity` è già firmata sugli eseguiti IBKR, ma non su tutte le
    # varianti: se c'è buySell lo si usa come verità.
    side = (attrs.get("buySell") or "").upper()
    if side.startswith("SELL") and quantity > 0:
        quantity = -quantity
    elif side.startswith("BUY") and quantity < 0:
        quantity = abs(quantity)
    # Le commissioni entrano nel carico, come fa IBKR: senza, il prezzo di
    # carico risulta più basso del vero e il gain/loss di conseguenza più
    # generoso. Arrivano negative perché sono un costo.
    commission = _ibkr_num(attrs.get("ibCommission") or attrs.get("commission"))
    # Il realizzato c'è solo se la query lo porta, e vale zero sugli eseguiti
    # che aprono. Le due cose vanno tenute distinte — "campo assente" significa
    # query da correggere, "zero" significa che quell'ordine non ha chiuso
    # niente — quindi niente `or`, che confonderebbe lo zero con l'assenza.
    realized = attrs.get("fifoPnlRealized")
    if realized in (None, ""):
        realized = attrs.get("realizedPnl")
    # Controvalore dell'eseguito: il capitale che quell'operazione ha mosso.
    # Si ricava da prezzo e quantità invece che da `tradeMoney` perché il
    # prezzo qui è già riportato alla valuta piena — il London quota in penny
    # — e `tradeMoney` no di sicuro. `tradeMoney` resta il ripiego per quando
    # il prezzo manca del tutto.
    multiplier = _ibkr_num(attrs.get("multiplier")) or 1.0
    if price is not None:
        money = abs(quantity) * price * multiplier
    else:
        money = _ibkr_num(attrs.get("tradeMoney"))
        money = abs(money) if money is not None else None
    return {
        "symbol": symbol,
        "name": attrs.get("description") or None,
        "quantity": quantity,
        "price": price,
        "commission": abs(commission) if commission is not None else 0.0,
        "realized_pnl": _ibkr_num(realized),
        "trade_money": money,
        # Con che ordine è stata eseguita: LMT, MKT, STP, STPLMT. Su una
        # chiusura è l'unica traccia che il Flex conserva del fatto che sia
        # scattato uno stop invece di un target — gli ordini in sé il report
        # non li contiene affatto.
        "order_type": (attrs.get("orderType") or "").strip().upper() or None,
        # "O" apre, "C" chiude, "C;O" chiude e riapre girando di segno.
        "open_close": (attrs.get("openCloseIndicator") or "").strip().upper() or None,
        "clock": _flex_trade_clock(attrs),
        # Cambio verso la valuta base allegato da IBKR: è quello con cui il
        # conto è valorizzato, e vale più di uno chiesto a FMP mesi dopo.
        "fx_rate_to_base": _ibkr_num(attrs.get("fxRateToBase")),
        "currency": currency,
        "exchange": exchange,
        "asset_class": (attrs.get("assetCategory") or "").upper() or None,
        "trade_date": (attrs.get("tradeDate") or attrs.get("reportDate") or "").replace("-", ""),
        "order_id": attrs.get("orderID") or attrs.get("ibOrderID") or None,
    }


def _flex_parse_trades(xml: str) -> List[dict]:
    """Eseguiti da un report Flex — Trade Confirmation o Activity con la
    sezione Trades.

    Si ignorano le righe di annullo: quello che serve è come è cambiata la
    quantità in portafoglio e quanto si è realizzato chiudendo.
    """
    return [t for t in (_flex_trade_from_attrs(a) for a in _flex_trade_rows(xml)) if t]


def _flex_apply_trades(positions: List[dict], trades: List[dict],
                       after_date: Optional[str] = None) -> dict:
    """Applica gli eseguiti alle posizioni di chiusura per ottenere quelle correnti.

    È il pezzo che rende il Flex utilizzabile in giornata: l'Activity fotografa
    la chiusura precedente e da solo non vedrebbe un limit scattato stamattina.

    `after_date` scarta gli eseguiti già compresi nella fotografia — senza,
    un'operazione di ieri verrebbe contata due volte.
    """
    current = {}
    for position in positions:
        current[(position.get("symbol") or "").upper()] = {**position}

    applied = 0
    for trade in sorted(trades, key=lambda t: t.get("trade_date") or ""):
        date = trade.get("trade_date") or ""
        if after_date and date and date <= after_date:
            continue
        symbol = trade["symbol"].upper()
        delta = trade["quantity"]
        existing = current.get(symbol)
        old_qty = (existing or {}).get("quantity") or 0.0
        new_qty = old_qty + delta
        applied += 1

        if abs(new_qty) < 1e-9:
            current.pop(symbol, None)
            continue

        price = trade.get("price")
        fee = trade.get("commission") or 0.0
        if existing is None or old_qty == 0 or (old_qty > 0) != (new_qty > 0):
            # Posizione nuova, o girata di segno: il carico riparte da questo
            # eseguito, commissione compresa.
            avg = (price + fee / abs(new_qty)) if price is not None and new_qty else price
        elif abs(new_qty) > abs(old_qty):
            # Si sta aggiungendo: media ponderata sui costi, commissione inclusa.
            old_avg = (existing or {}).get("avg_price")
            avg = (((old_qty * old_avg) + (delta * price) + fee) / new_qty
                   if old_avg is not None and price is not None else old_avg)
        else:
            # Si sta riducendo: il carico non cambia. La commissione della
            # vendita è un costo realizzato, non un aumento del carico residuo.
            avg = (existing or {}).get("avg_price")

        # Il prezzo dell'ultimo eseguito è la quotazione più recente che si
        # abbia: meglio della chiusura di ieri, che è l'alternativa.
        market_price = trade.get("price") or (existing or {}).get("market_price")
        current[symbol] = {
            **(existing or {}),
            "symbol": symbol,
            "name": (existing or {}).get("name") or trade.get("name"),
            "quantity": new_qty,
            "avg_price": avg,
            "market_price": market_price,
            "market_value": (new_qty * market_price) if market_price is not None else None,
            "currency": (existing or {}).get("currency") or trade.get("currency"),
            "exchange": (existing or {}).get("exchange") or trade.get("exchange"),
            "asset_class": (existing or {}).get("asset_class") or trade.get("asset_class"),
            # A questo livello il P&L non è calcolabile: l'unico prezzo che si
            # ha è quello dell'eseguito, che darebbe zero per costruzione. Lo
            # riempie la rivalutazione col listino, in lettura.
            "unrealized_pnl": None,
            "derived_from_trades": True,
        }

    return {"positions": list(current.values()), "applied": applied}


def _flex_fetch_positions() -> dict:
    """Posizioni correnti dal Flex: chiusura precedente più eseguiti di oggi.

    Senza la query degli eseguiti si ottiene comunque la fotografia della
    chiusura, che è il comportamento di prima.
    """
    query_id = _ibkr_api_env("IBKR_FLEX_QUERY_ID")
    if not _ibkr_api_env("IBKR_FLEX_TOKEN") or not query_id:
        return {"error": "IBKR_FLEX_TOKEN / IBKR_FLEX_QUERY_ID non configurati"}

    # L'Activity fotografa una chiusura già avvenuta: entro la giornata non
    # cambia, e IBKR stessa dice che non c'è beneficio a rigenerarla più di una
    # volta al giorno. Durante il giorno si rilegge solo la Trade Confirmation,
    # che è la parte viva — e si dimezzano le richieste, che il Flex limita.
    cached = _FLEX_ACTIVITY_CACHE.get("value")
    if cached and (time.time() - _FLEX_ACTIVITY_CACHE.get("ts", 0)) < _FLEX_ACTIVITY_TTL_SECONDS:
        activity = cached
    else:
        fetched = _flex_fetch_statement(query_id)
        if fetched.get("error"):
            return fetched
        activity = _flex_parse_activity(fetched["xml"])
        _FLEX_ACTIVITY_CACHE.update({"ts": time.time(), "value": activity})

    result = {
        "positions": activity["positions"],
        "account": activity["account"],
        "rates": activity["rates"],
        "report_date": activity["report_date"],
        "trades_applied": 0,
    }

    trades_query = _ibkr_api_env("IBKR_FLEX_TRADES_QUERY_ID")
    # Va detto se la query non c'è: "zero eseguiti applicati" da solo non
    # distingue una giornata senza operazioni da una variabile mancante, ed è
    # la prima cosa che si vuole sapere dopo aver configurato il portale.
    result["trades_query_configured"] = bool(trades_query)
    if not trades_query:
        return result

    trades_fetched = _flex_fetch_statement(trades_query)
    if trades_fetched.get("error"):
        # Gli eseguiti sono un miglioramento, non un requisito: se la query non
        # risponde restano le posizioni di chiusura, dichiarando perché.
        result["trades_error"] = trades_fetched["error"]
        return result

    trades = _flex_parse_trades(trades_fetched["xml"])
    merged = _flex_apply_trades(activity["positions"], trades,
                                after_date=activity["report_date"])
    result["positions"] = merged["positions"]
    result["trades_applied"] = merged["applied"]
    result["trades_seen"] = len(trades)
    if merged["applied"]:
        # Le posizioni non sono più quelle di chiusura: la data del dato è
        # adesso, ed è ciò che impedisce al cron di essere scartato come vecchio.
        result["report_date"] = None
    return result


# ---------------------------------------------------------------------------
# Storico del P&L giornaliero — dagli eseguiti Flex
# ---------------------------------------------------------------------------
# Il calendario del portafoglio chiede una cosa che nessuna delle sorgenti già
# in uso porta: quanto si è guadagnato o perso *ogni* giorno, indietro nel
# tempo. Le posizioni dicono dove si è adesso, il P&L del gateway dice com'è
# andata oggi, e nessuno dei due si ricorda di ieri.
#
# La sorgente giusta è una Activity Flex con la sezione "Trades" su un periodo
# lungo: ogni eseguito porta `fifoPnlRealized`, cioè il realizzato di quella
# chiusura, già al netto delle commissioni. Sommandoli per giornata si ottiene
# la riga del calendario.

_FLEX_PNL_CACHE: Dict[str, Any] = {}
_FLEX_PNL_TTL_SECONDS = int((os.getenv("FLEX_PNL_TTL") or "3600").strip() or 3600)

# Le conversioni valutarie compaiono fra gli eseguiti ma non sono operazioni:
# contarle gonfierebbe il numero di trade della giornata senza portare P&L.
_FLEX_PNL_SKIP_CATEGORIES = {"CASH"}

# Quanti giorni di storico si tengono in archivio. Tre anni bastano a
# qualunque uso del calendario e il documento resta lontanissimo dai 16 MB.
_IBKR_PNL_MAX_DAYS = 1100

# Oltre quanti simboli si smette di dettagliare la giornata. Il dettaglio serve
# al passaggio del mouse, non a rileggere il registro: venti righe sono già più
# di quante se ne guardino.
_IBKR_PNL_MAX_SYMBOLS_PER_DAY = 20

# Quante operazioni si conservano per giornata. Il registro di un giorno serve
# a rileggere cosa si è fatto, e oltre una sessantina di righe non lo si legge
# più; il tetto tiene anche il documento lontano dai 16 MB di Mongo — un anno
# molto attivo sta sulle tremila operazioni, quindi il caso peggiore vero è
# ordini di grandezza sotto il limite.
_IBKR_PNL_MAX_OPS_PER_DAY = 60


def _flex_trade_day(trade: dict) -> Optional[str]:
    """Data di un eseguito, da `YYYYMMDD` a `YYYY-MM-DD`."""
    raw = (trade.get("trade_date") or "").strip().replace("-", "")
    if len(raw) != 8 or not raw.isdigit():
        return None
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _flex_pnl_days(trades: List[dict]) -> dict:
    """Eseguiti → una riga per giornata, in valuta base.

    Un giorno conta due numeri diversi e vanno tenuti separati: le *operazioni*
    sono tutti gli eseguiti, le *chiusure* solo quelle che hanno realizzato
    qualcosa. La percentuale di successo si calcola sulle seconde — un ordine
    di acquisto non è né vinto né perso, e metterlo al denominatore
    schiaccerebbe la percentuale verso il basso senza motivo.
    """
    days: Dict[str, dict] = {}
    fx_missing = set()
    realized_seen = False

    for trade in trades:
        if (trade.get("asset_class") or "").upper() in _FLEX_PNL_SKIP_CATEGORIES:
            continue
        day = _flex_trade_day(trade)
        if not day:
            continue
        entry = days.setdefault(day, {
            "date": day, "pnl": 0.0, "trades": 0, "closed": 0,
            "wins": 0, "losses": 0, "commissions": 0.0, "symbols": {},
            "capital_open": 0.0, "capital_close": 0.0,
            "exits_stop": 0, "exits_target": 0, "exits_market": 0,
            "ops": [],
        })
        entry["trades"] += 1

        rate = trade.get("fx_rate_to_base")
        if rate is None:
            rate = _fx_to_base(trade.get("currency"))
        commission = trade.get("commission") or 0.0
        if rate is not None:
            entry["commissions"] += commission * rate

        # Capitale mosso dall'operazione, in valuta base. Aperture e chiusure
        # si sommano a parte: metterle insieme conterebbe due volte lo stesso
        # capitale su un giro aperto e chiuso in giornata.
        money = trade.get("trade_money")
        money_base = money * rate if (money is not None and rate is not None) else None
        side = trade.get("open_close") or ""
        if money_base is not None:
            if "O" in side:
                entry["capital_open"] += money_base
            if "C" in side:
                entry["capital_close"] += money_base
        # Come si è usciti. Ha senso solo sulle chiusure: il tipo d'ordine di
        # un'apertura dice come si è entrati, non se è andata bene.
        if "C" in side:
            order_type = trade.get("order_type") or ""
            if order_type.startswith("STP"):
                entry["exits_stop"] += 1
            elif order_type.startswith("LMT") or order_type == "MIDPX":
                entry["exits_target"] += 1
            else:
                entry["exits_market"] += 1

        entry["ops"].append({
            "symbol": (trade.get("symbol") or "").upper(),
            "quantity": trade.get("quantity"),
            "price": trade.get("price"),
            "capital": round(money_base, 2) if money_base is not None else None,
            "currency": trade.get("currency"),
            "order_type": trade.get("order_type"),
            "open_close": trade.get("open_close"),
            "clock": trade.get("clock"),
            "realized": None,   # riempito sotto, quando il cambio c'è
        })

        realized = trade.get("realized_pnl")
        if realized is None:
            # La query non porta il realizzato: la giornata resta contata come
            # attività ma senza P&L. Chi chiama lo deve poter dire in pagina.
            continue
        realized_seen = True
        if rate is None:
            # Senza cambio il numero non è sommabile: si dichiara la giornata
            # incompleta invece di trattare i dollari come euro.
            fx_missing.add(day)
            continue
        value = realized * rate
        entry["pnl"] += value
        entry["ops"][-1]["realized"] = round(value, 2)
        symbol = (trade.get("symbol") or "").upper()
        bucket = entry["symbols"].setdefault(symbol, {"symbol": symbol, "pnl": 0.0, "trades": 0})
        bucket["pnl"] += value
        bucket["trades"] += 1
        if abs(value) < 1e-9:
            # Eseguito di apertura: non ha chiuso niente, quindi non è né una
            # vincita né una perdita.
            continue
        entry["closed"] += 1
        if value > 0:
            entry["wins"] += 1
        else:
            entry["losses"] += 1

    out = {}
    for day, entry in days.items():
        symbols = sorted(entry["symbols"].values(), key=lambda s: -abs(s["pnl"]))
        ops = sorted(entry["ops"], key=lambda o: (o.get("clock") or "99:99"))
        out[day] = {
            "date": day,
            "pnl": round(entry["pnl"], 2),
            "trades": entry["trades"],
            "closed": entry["closed"],
            "wins": entry["wins"],
            "losses": entry["losses"],
            "commissions": round(entry["commissions"], 2),
            "capital_open": round(entry["capital_open"], 2),
            "capital_close": round(entry["capital_close"], 2),
            "exits_stop": entry["exits_stop"],
            "exits_target": entry["exits_target"],
            "exits_market": entry["exits_market"],
            "symbols": [{"symbol": s["symbol"], "pnl": round(s["pnl"], 2),
                         "trades": s["trades"]}
                        for s in symbols[:_IBKR_PNL_MAX_SYMBOLS_PER_DAY]],
            "ops": ops[:_IBKR_PNL_MAX_OPS_PER_DAY],
            # Se il registro è stato tagliato va detto: una tabella troncata in
            # silenzio si legge come se fosse tutta la giornata.
            "ops_truncated": max(0, len(ops) - _IBKR_PNL_MAX_OPS_PER_DAY),
            "partial": day in fx_missing,
        }
    return {"days": out, "realized_available": realized_seen,
            "fx_missing": sorted(fx_missing)}


# Le sorelle si chiamano IBKR_FLEX_QUERY_ID e IBKR_FLEX_TRADES_QUERY_ID, quindi
# scrivere questa senza il suffisso è un errore naturale — ed è successo. Si
# accettano entrambe le forme: l'alternativa è che il calendario ripieghi in
# silenzio su una query che il realizzato non ce l'ha, cioè un sintomo che non
# assomiglia per niente alla causa.
_IBKR_PNL_QUERY_ENV_NAMES = ("IBKR_FLEX_PNL_QUERY_ID", "IBKR_FLEX_PNL_QUERY")


def _ibkr_pnl_query_id() -> str:
    for name in _IBKR_PNL_QUERY_ENV_NAMES:
        value = _ibkr_api_env(name)
        if value:
            return value
    return ""


def _flex_fetch_pnl_history(force: bool = False) -> dict:
    """Storico giornaliero dal Flex. Ritorna {"days", ...} oppure {"error"}.

    Vuole una query propria (`IBKR_FLEX_PNL_QUERY_ID`) perché quella degli
    eseguiti copre "Today" e per il calendario servirebbe a poco. Se non c'è si
    ripiega su quella, dichiarandolo: si otterrà la sola giornata di oggi, che
    accumulandosi giro dopo giro costruisce comunque lo storico da qui in poi.
    """
    query_id = _ibkr_pnl_query_id()
    fallback = False
    if not query_id:
        query_id = _ibkr_api_env("IBKR_FLEX_TRADES_QUERY_ID")
        fallback = bool(query_id)
    if not _ibkr_api_env("IBKR_FLEX_TOKEN") or not query_id:
        return {"error": "IBKR_FLEX_TOKEN / IBKR_FLEX_PNL_QUERY_ID non configurati"}

    cached = _FLEX_PNL_CACHE.get(query_id)
    if not force and cached and (time.time() - cached.get("ts", 0)) < _FLEX_PNL_TTL_SECONDS:
        return {**cached["value"], "cached": True}

    fetched = _flex_fetch_statement(query_id)
    if fetched.get("error"):
        return fetched
    trades = _flex_parse_trades(fetched["xml"])
    aggregated = _flex_pnl_days(trades)
    result = {
        "days": aggregated["days"],
        "realized_available": aggregated["realized_available"],
        "fx_missing": aggregated["fx_missing"],
        "trades_seen": len(trades),
        "currency": _ibkr_base_currency(),
        "fallback_query": fallback,
    }
    _FLEX_PNL_CACHE[query_id] = {"ts": time.time(), "value": result}
    return result


def _ibkr_store_pnl_days(owner_email: str, days: dict,
                         meta: Optional[dict] = None) -> Optional[dict]:
    """Fonde lo storico giornaliero con quello già in archivio.

    Fonde per data invece di sostituire perché la query Flex può coprire un
    periodo corto — al limite la sola giornata di oggi — e una sostituzione
    butterebbe via tutto il resto a ogni giro. Così lo storico si accumula
    anche partendo da una query stretta: ci mette solo il tempo che serve.

    Ritorna lo storico risultante, o None se Mongo non è disponibile.
    """
    coll = _get_mongo_ibkr_collection()
    if coll is None or not owner_email:
        return None
    existing = _ibkr_load_snapshot(owner_email) or {}
    stored = existing.get("pnl_days") if isinstance(existing.get("pnl_days"), dict) else {}
    merged = {**stored, **(days or {})}
    # Si tengono le giornate più recenti: il taglio è sulla data, non
    # sull'ordine di inserimento, perché una risincronizzazione può riportare
    # indietro giorni vecchi.
    if len(merged) > _IBKR_PNL_MAX_DAYS:
        keep = sorted(merged.keys())[-_IBKR_PNL_MAX_DAYS:]
        merged = {k: merged[k] for k in keep}
    update = {"pnl_days": merged, "pnl_synced_at": time.time()}
    if isinstance(meta, dict):
        update["pnl_meta"] = meta
    try:
        coll.update_one({"owner_email": owner_email}, {"$set": update}, upsert=True)
    except Exception:
        return None
    return merged


# Ogni quanto il giro schedulato rilegge lo storico. Non è un dato che si
# muove di continuo — cambia solo quando si chiude qualcosa — e ogni lettura è
# un report Flex da generare, su un servizio che le richieste le limita.
_IBKR_PNL_MAX_AGE_SECONDS = int(
    (os.getenv("IBKR_PNL_MAX_AGE") or "14400").strip() or 14400)  # 4h


def _ibkr_reset_pnl_days(owner_email: str) -> bool:
    """Butta via lo storico giornaliero.

    Serve dopo aver cambiato o corretto la query: le giornate raccolte con
    quella vecchia possono essere incomplete, e siccome la fusione è per data
    resterebbero lì per sempre — le sovrascrive solo un report che copra la
    stessa giornata.
    """
    coll = _get_mongo_ibkr_collection()
    if coll is None or not owner_email:
        return False
    try:
        coll.update_one({"owner_email": owner_email},
                        {"$unset": {"pnl_days": "", "pnl_synced_at": "", "pnl_meta": ""}})
        return True
    except Exception:
        return False


def _ibkr_refresh_pnl_history(owner_email: str, force: bool = False) -> dict:
    """Aggiorna lo storico giornaliero, se è ora.

    Best effort: il calendario è un di più rispetto alle posizioni, e un errore
    qui non deve far fallire il giro che le porta.
    """
    if not owner_email:
        return {"status": "skipped", "reason": "nessun proprietario"}
    doc = _ibkr_load_snapshot(owner_email) or {}
    synced_at = doc.get("pnl_synced_at")
    if not force and synced_at:
        age = time.time() - float(synced_at)
        if age < _IBKR_PNL_MAX_AGE_SECONDS:
            return {"status": "skipped",
                    "reason": f"aggiornato {age / 3600:.1f}h fa",
                    "days": len(doc.get("pnl_days") or {})}

    fetched = _flex_fetch_pnl_history(force=force)
    if fetched.get("error"):
        return {"status": "error", "error": fetched["error"],
                "hint": fetched.get("hint")}
    meta = {
        "currency": fetched.get("currency"),
        "realized_available": fetched.get("realized_available"),
        "fallback_query": fetched.get("fallback_query"),
        "trades_seen": fetched.get("trades_seen"),
        "fx_missing": fetched.get("fx_missing"),
    }
    # Un report senza realizzato produce giornate a zero: non "non ho
    # guadagnato niente", ma "non lo so". Salvarle vorrebbe dire scrivere in
    # calendario un numero sicuro di sé che non ha nessuna base, e la fusione
    # per data lo lascerebbe lì finché non passa un report sulla stessa
    # giornata. Si tiene solo il meta, che è ciò che fa comparire l'avviso.
    if not fetched.get("realized_available"):
        _ibkr_store_pnl_days(owner_email, {}, meta=meta)
        return {"status": "ok", "days_in_report": len(fetched["days"]),
                "days_stored": 0, "realized_available": False,
                "trades_seen": fetched.get("trades_seen"),
                "fallback_query": fetched.get("fallback_query"),
                "reason": "il report non porta il realizzato: giornate non salvate"}

    merged = _ibkr_store_pnl_days(owner_email, fetched["days"], meta=meta)
    if merged is None:
        return {"status": "error", "error": "mongo non disponibile: storico non salvato"}
    return {"status": "ok", "days": len(merged),
            "days_in_report": len(fetched["days"]),
            # Quante righe di registro sono finite in archivio: è il numero che
            # dice se il documento sta crescendo verso i 16 MB di Mongo, e
            # l'unico modo da fuori di distinguere un archivio col dettaglio
            # delle operazioni da uno col solo aggregato.
            "ops_stored": sum(len(d.get("ops") or []) for d in merged.values()
                              if isinstance(d, dict)),
            "trades_seen": fetched.get("trades_seen"),
            "realized_available": fetched.get("realized_available"),
            "fallback_query": fetched.get("fallback_query")}


@app.route('/api/ibkr/pnl-calendar', methods=['GET'])
@login_required
def api_ibkr_pnl_calendar():
    """Storico del P&L giornaliero per il calendario del portafoglio.

    Di default legge quello in archivio, che è istantaneo. `?refresh=1` va a
    rigenerare il report Flex: sono decine di secondi, quindi resta legato a un
    gesto esplicito e non al caricamento della pagina.
    """
    owner_email = _current_user_email()
    if not owner_email:
        return jsonify({"error": "no user"}), 401

    refreshed = None
    if request.args.get("refresh") == "1":
        refreshed = _ibkr_refresh_pnl_history(owner_email, force=True)

    doc = _ibkr_load_snapshot(owner_email) or {}
    days = doc.get("pnl_days") if isinstance(doc.get("pnl_days"), dict) else {}
    meta = doc.get("pnl_meta") if isinstance(doc.get("pnl_meta"), dict) else {}

    # Il registro di una singola giornata. Sta in una richiesta a sé perché un
    # anno di operazioni sono megabyte di JSON: spedirli a ogni caricamento
    # della pagina per mostrarne una manciata al clic non ha senso.
    requested_day = (request.args.get("day") or "").strip()
    if requested_day:
        entry = days.get(requested_day)
        if not entry:
            return jsonify({"day": requested_day, "found": False,
                            "currency": meta.get("currency") or _ibkr_base_currency()})
        return jsonify({"day": requested_day, "found": True, "detail": entry,
                        "currency": meta.get("currency") or _ibkr_base_currency()})

    # Nella vista mensile il registro non serve: si tiene l'aggregato e il
    # riepilogo per titolo, che è quello che riempie casella e tooltip.
    days = {key: {k: v for k, v in entry.items() if k != "ops"}
            for key, entry in days.items() if isinstance(entry, dict)}

    # Perché il calendario è vuoto è la prima domanda che ci si pone, e da fuori
    # le cause si somigliano tutte. Qui si dice quale.
    hint = None
    if not days:
        if not _ibkr_api_env("IBKR_FLEX_TOKEN"):
            hint = ("IBKR_FLEX_TOKEN non configurato: senza Flex non c'è da dove "
                    "leggere gli eseguiti.")
        elif not _ibkr_pnl_query_id() and not _ibkr_api_env("IBKR_FLEX_TRADES_QUERY_ID"):
            hint = ("Nessuna query eseguiti configurata: serve IBKR_FLEX_PNL_QUERY_ID, "
                    "una Activity Flex con la sezione Trades su un periodo lungo.")
        elif not doc.get("pnl_synced_at"):
            hint = "Storico mai letto: premi Aggiorna, oppure aspetta il prossimo giro schedulato."
        else:
            hint = "Nessun eseguito nel periodo coperto dalla query."
    # L'ordine conta: il ripiego è la causa a monte. Segnalare per primo il
    # realizzato mancante manderebbe a correggere la query sbagliata — quella
    # di ripiego il realizzato non ce l'ha per costruzione.
    elif meta.get("fallback_query"):
        hint = ("IBKR_FLEX_PNL_QUERY_ID non configurato: si sta usando la query degli "
                "eseguiti di oggi, che il realizzato non lo porta. Serve una Activity "
                "Flex con la sezione Trades su un periodo lungo.")
    elif meta.get("realized_available") is False:
        hint = ("La query porta gli eseguiti ma non il realizzato: aggiungi il campo "
                "fifoPnlRealized alla sezione Trades, altrimenti le giornate restano a zero.")

    return jsonify({
        "days": days,
        "currency": meta.get("currency") or _ibkr_base_currency(),
        "synced_at": doc.get("pnl_synced_at"),
        "meta": meta,
        "hint": hint,
        "refreshed": refreshed,
    })


@app.route('/api/ibkr/sync', methods=['POST'])
def api_ibkr_sync():
    """Ingest dello snapshot IBKR, anche parziale.

    Body: {"positions": [...], "orders": [...], "notify": bool,
           "owner_email": "...", "source": "gateway"}. L'autenticazione è col
    bearer token `IBKR_SYNC_TOKEN`, non con la sessione: chi chiama gira
    headless.

    Le due liste sono indipendenti: mandare solo `orders` aggiorna gli ordini e
    lascia intatte le posizioni, e viceversa. Serve all'ibrido Flex + gateway,
    dove le due metà arrivano da sorgenti diverse e con ritmi diversi.
    """
    if not _ibkr_sync_authorized():
        return jsonify({"error": "unauthorized"}), 401
    data = request.get_json(silent=True) or {}
    owner_email = (_ibkr_str(data.get("owner_email"), 120) or "").lower() or _ibkr_default_owner_email()
    if not owner_email:
        return jsonify({"error": "owner_email mancante e nessun default configurato "
                                 "(IBKR_SYNC_USER_EMAIL o ADMIN_EMAILS)"}), 400

    # Chiave assente = "non ho notizie", chiave presente ma vuota = "non c'è
    # più niente". Sono due cose diverse e vanno distinte, altrimenti un
    # gateway che non trova ordini non riuscirebbe mai a svuotare la lista.
    normalized = _ibkr_normalize_payload(data)
    positions = normalized["positions"] if "positions" in data else None
    orders = normalized["orders"] if "orders" in data else None
    if positions is None and orders is None:
        return jsonify({"error": "payload senza né 'positions' né 'orders'"}), 400

    source = _ibkr_str(data.get("source"), 24) or "sync"
    merged = _ibkr_store_snapshot(owner_email, positions, orders, source=source,
                                  positions_as_of=_ibkr_num(data.get("positions_as_of")),
                                  account=data.get("account"))
    if merged is None:
        return jsonify({"error": "mongo non disponibile: snapshot non salvato"}), 503

    snapshot = {"positions": merged["positions"], "orders": merged["orders"]}
    freshness = _ibkr_orders_staleness(merged)
    alert = _ibkr_alert_with_rendering(
        _ibkr_earnings_alert(snapshot, earnings=merged.get("earnings")),
        orders_freshness=freshness)
    telegram = _ibkr_maybe_notify(alert, data)

    return jsonify({
        "status": "ok",
        "owner_email": owner_email,
        # Il chiamante manda la mail solo se c'è qualcosa da mandare: stessa
        # soglia della notifica Telegram, così i due canali non divergono.
        "should_email": bool(alert["count"]) or bool(data.get("notify_always")),
        "stored": True,
        "updated": merged.get("applied") or [],
        "skipped": merged.get("skipped"),
        "positions": len(snapshot["positions"]),
        "orders": len(snapshot["orders"]),
        "live_orders": sum(1 for o in snapshot["orders"] if o.get("is_live")),
        "orders_freshness": freshness,
        "telegram": telegram,
        "alert": alert,
    })


@app.route('/api/ibkr/snapshot', methods=['GET'])
@login_required
def api_ibkr_snapshot():
    """Snapshot IBKR dell'utente in sessione, arricchito con le date earnings.

    Di default riusa le date già risolte dall'ultima sync: sono buone per
    ore e rifarle significherebbe una ventina di chiamate FMP a ogni
    caricamento di pagina. `?refresh=1` le ricalcola.
    """
    owner_email = _current_user_email()
    if not owner_email:
        return jsonify({"error": "no user"}), 401
    doc = _ibkr_load_snapshot(owner_email)
    if not doc:
        # Senza snapshot la pagina non ha niente da mostrare, ma "niente" ha
        # cause diverse — token non configurato, Mongo giù, email di
        # destinazione diversa da quella con cui si è loggati — e sono tutte
        # invisibili da fuori. Meglio dire quale.
        default_owner = _ibkr_default_owner_email()
        return jsonify({
            "positions": [], "orders": [], "synced_at": None, "alert": None,
            "hint": "Nessuno snapshot IBKR ancora salvato per " + owner_email,
            "diagnostics": {
                "logged_in_as": owner_email,
                "sync_writes_to": default_owner or None,
                "owner_matches": bool(default_owner) and default_owner == owner_email,
                "sync_token_configured": bool((os.getenv("IBKR_SYNC_TOKEN") or "").strip()),
                "telegram_configured": bool((os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
                                            and (os.getenv("TELEGRAM_CHAT_ID") or "").strip()),
                "fmp_configured": bool((os.getenv("FMP_API_KEY") or "").strip()),
                "mongo_available": _get_mongo_ibkr_collection() is not None,
            },
        })

    snapshot = {"positions": doc.get("positions") or [], "orders": doc.get("orders") or []}
    earnings = doc.get("earnings") if isinstance(doc.get("earnings"), dict) else None
    if request.args.get("refresh") == "1" or not earnings:
        earnings = _ibkr_earnings_map(snapshot)

    enriched = _ibkr_enriched_snapshot(snapshot, earnings)
    alert = _ibkr_earnings_alert(snapshot, earnings=earnings)
    return jsonify({
        "positions": enriched["positions"],
        "orders": enriched["orders"],
        "synced_at": doc.get("synced_at"),
        "positions_synced_at": doc.get("positions_synced_at"),
        "positions_source": doc.get("positions_source"),
        # La pagina deve poter dire da dove viene ogni metà e quanto è vecchia:
        # con Flex e gateway che aggiornano a ritmi diversi, un unico "aggiornato
        # alle 20:00" sarebbe fuorviante su una delle due.
        "orders_freshness": _ibkr_orders_staleness(doc),
        "positions_freshness": _ibkr_positions_staleness(doc),
        "alert": {k: alert[k] for k in ("target_date", "count", "unresolved")},
        "alert_symbols": [i["symbol"] for i in alert["items"]],
    })


def _ibkr_capital_summary(doc: dict, positions: List[dict], orders_only: List[dict]) -> dict:
    """Quanto capitale è investito, quanto lo sarebbe e su quale totale.

    Il totale è il net liquidation del conto, che arriva dal gateway: senza,
    l'esposizione non è calcolabile e va detto invece di inventare un
    denominatore. Il capitale degli ordini è la somma dei soli acquisti, cioè
    il caso in cui venissero eseguiti tutti — non una previsione, un tetto.
    """
    account = doc.get("account") if isinstance(doc.get("account"), dict) else {}
    total = account.get("net_liquidation")

    invested = [p["market_value_base"] for p in positions if p.get("market_value_base") is not None]
    invested_total = sum(invested) if invested else None
    missing_value = sum(1 for p in positions if p.get("market_value_base") is None)

    pending = [r["pending_buy_base"] for r in (positions + orders_only)
               if r.get("pending_buy_base") is not None]
    pending_total = sum(pending) if pending else None

    def pct(part):
        if part is None or not total:
            return None
        return part / total

    # Il giornaliero del conto viene dal gateway e comprende anche il
    # realizzato delle posizioni chiuse oggi. Senza gateway si ripiega sulla
    # somma delle variazioni di giornata dei titoli ancora aperti, che è una
    # cosa diversa e va detto: chi ha venduto in giornata non la vedrebbe.
    # Il numero di IBKR vale solo finché è fresco: il gateway gira una volta la
    # mattina, e un P&L delle 9:00 mostrato alle 17:00 come "oggi" sarebbe
    # sbagliato senza sembrarlo. Scaduto, si ripiega sulla stima.
    stamped_at = doc.get("daily_pnl_at")
    daily_age = (time.time() - float(stamped_at)) if stamped_at else None
    daily = account.get("daily_pnl")
    if daily is not None and (daily_age is None or daily_age > _IBKR_DAILY_PNL_MAX_AGE_SECONDS):
        daily = None
    daily_source = "ibkr"
    if daily is None:
        parts = []
        for position in positions:
            value = position.get("daily_pnl")
            if value is None:
                continue
            rate = _fx_to_base(position.get("currency"))
            if rate:
                parts.append(value * rate)
        daily = sum(parts) if parts else None
        daily_source = "stimato" if daily is not None else None

    return {
        "net_liquidation": total,
        "daily_pnl": daily,
        "daily_pnl_pct": pct(daily),
        "daily_pnl_source": daily_source,
        "cash": account.get("cash"),
        "currency": account.get("currency") or _ibkr_base_currency(),
        "as_of": doc.get("account_synced_at"),
        "invested": invested_total,
        "invested_pct": pct(invested_total),
        "pending_buy": pending_total,
        "pending_pct": pct(pending_total),
        # Somma dell'investito e di quanto sarebbe investito se tutti gli
        # acquisti pendenti andassero a segno.
        "committed_pct": pct((invested_total or 0) + (pending_total or 0)) if total else None,
        # Se qualche posizione non ha un controvalore convertibile, la
        # percentuale è per difetto e va dichiarato.
        "positions_without_value": missing_value,
        # Rete di sicurezza contro errori di unità di misura: un capitale
        # impegnato molte volte più grande del conto non è un dato, è un bug.
        # È già successo con i prezzi in penny del London Stock Exchange, e
        # senza questo controllo la pagina avrebbe mostrato "1024%" con
        # l'aplomb di un numero vero.
        "implausible": bool(total and pending_total and pending_total > total * 5),
    }


_QUOTE_CACHE: Dict[str, dict] = {}
_QUOTE_CACHE_TTL_SECONDS = int((os.getenv("QUOTE_CACHE_TTL") or "300").strip() or 300)  # 5 min


def _fetch_quote(fmp_symbol: str) -> dict:
    """Ultimo prezzo e variazione di giornata da FMP.

    Cache 5 minuti: la pagina si ricarica da sola ogni minuto e non serve
    interrogare il listino a ogni giro.
    """
    if not fmp_symbol:
        return {}
    cached = _QUOTE_CACHE.get(fmp_symbol)
    now = time.time()
    if cached and (now - cached["ts"]) < _QUOTE_CACHE_TTL_SECONDS:
        return cached["quote"]
    quote = {}
    data = _fmp_get("quote", symbol=fmp_symbol)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        quote = {"price": _ibkr_num(data[0].get("price")),
                 "change": _ibkr_num(data[0].get("change"))}
    _QUOTE_CACHE[fmp_symbol] = {"ts": now, "quote": quote}
    return quote


def _fetch_quote_price(fmp_symbol: str) -> Optional[float]:
    return _fetch_quote(fmp_symbol).get("price")


def _ibkr_revalue_position(row: dict) -> dict:
    """Rivaluta una posizione col prezzo corrente di FMP.

    Serve solo alle posizioni che arrivano dal Flex: quelle di chiusura sono
    valorizzate a ieri, e quelle ricostruite dagli eseguiti hanno come unico
    prezzo quello a cui si è comprato — con un P&L che sarebbe zero per
    costruzione. Le posizioni del gateway invece sono già live e non si toccano.
    """
    price = _fetch_quote_price(row.get("fmp_symbol"))
    if price is None or price <= 0:
        return row
    # FMP quota Londra in penny come IBKR: senza la stessa correzione il
    # controvalore risulterebbe centuplicato.
    price *= _ibkr_price_scale(row.get("exchange"), row.get("currency"))
    reference = row.get("avg_price") or row.get("market_price")
    if reference and not (0.02 < price / reference < 50):
        # Scarto implausibile: quasi certamente unità di misura diverse o
        # simbolo agganciato al titolo sbagliato. Meglio il dato vecchio di uno
        # inventato.
        return {**row, "revalue_rejected": True}
    quantity = row.get("quantity") or 0
    avg = row.get("avg_price")
    scale = _ibkr_price_scale(row.get("exchange"), row.get("currency"))
    change = _fetch_quote(row.get("fmp_symbol")).get("change")
    if change is not None:
        change *= scale
    return {
        **row,
        "market_price": price,
        "market_value": quantity * price,
        "unrealized_pnl": ((price - avg) * quantity) if avg is not None else row.get("unrealized_pnl"),
        # Variazione di giornata del titolo per la quantità detenuta. Su una
        # posizione aperta oggi sovrastima, perché parte dalla chiusura di ieri
        # e non dal prezzo d'ingresso: chi la somma deve dirlo.
        "daily_pnl": (change * quantity) if change is not None else None,
        "revalued": True,
    }


_IBKR_ANALYSIS_CACHE: Dict[str, dict] = {}
_IBKR_ANALYSIS_TTL_SECONDS = int(
    (os.getenv("IBKR_ANALYSIS_TTL") or "21600").strip() or 21600)  # 6h


def _analyze_portfolio_ticker_cached(fmp_symbol: str) -> dict:
    """`_analyze_portfolio_ticker` con cache in memoria.

    Ogni analisi sono tre chiamate a FMP, e la pagina ne chiede una ventina in
    un colpo: senza cache ogni ricaricamento pagherebbe l'intero costo, con il
    rischio concreto di sbattere contro il tetto di durata della funzione.
    I fondamentali si muovono per trimestri, quindi 6h di cache non costano
    nulla in accuratezza.
    """
    cached = _IBKR_ANALYSIS_CACHE.get(fmp_symbol)
    now = time.time()
    if cached and (now - cached["ts"]) < _IBKR_ANALYSIS_TTL_SECONDS:
        return cached["value"]
    value = _analyze_portfolio_ticker(fmp_symbol, None)
    # Gli errori si mettono in cache per un decimo del tempo: se FMP era solo
    # momentaneamente giù, non deve restare "non disponibile" per sei ore.
    _IBKR_ANALYSIS_CACHE[fmp_symbol] = {
        "ts": now if not value.get("error") else now - _IBKR_ANALYSIS_TTL_SECONDS * 0.9,
        "value": value,
    }
    return value


@app.route('/api/ibkr/pulse', methods=['GET'])
@login_required
def api_ibkr_pulse():
    """Solo le date di ultimo aggiornamento dello snapshot.

    La pagina la interroga ogni minuto per sapere se vale la pena ricaricare:
    ripescare posizioni, ordini e analisi a ogni giro sarebbe una ventina di
    chiamate a FMP per scoprire che non è cambiato niente. Qui si legge un
    documento solo, senza arricchimenti.
    """
    owner_email = _current_user_email()
    if not owner_email:
        return jsonify({"error": "no user"}), 401
    coll = _get_mongo_ibkr_collection()
    if coll is None:
        return jsonify({"synced_at": None})
    try:
        doc = coll.find_one({"owner_email": owner_email},
                            {"_id": 0, "synced_at": 1, "positions_synced_at": 1,
                             "orders_synced_at": 1, "positions_source": 1,
                             "pnl_synced_at": 1}) or {}
    except Exception:
        return jsonify({"synced_at": None})
    return jsonify({
        "synced_at": doc.get("synced_at"),
        "positions_synced_at": doc.get("positions_synced_at"),
        "orders_synced_at": doc.get("orders_synced_at"),
        "positions_source": doc.get("positions_source"),
        # Lo storico si aggiorna con un ritmo suo, molto più lento delle
        # posizioni: senza una data a sé il calendario non saprebbe mai quando
        # è cambiato davvero.
        "pnl_synced_at": doc.get("pnl_synced_at"),
    })


@app.route('/api/screener/earnings', methods=['GET'])
@login_required
def api_screener_earnings():
    """Prossima trimestrale per un elenco di ticker.

    Serve allo screener, che carica le date dopo aver disegnato le schede: sono
    una ventina di chiamate a FMP e non devono ritardare la lista. I ticker
    arrivano già in forma FMP, quindi non passano dalla traduzione IBKR.
    """
    raw = (request.args.get("tickers") or "").strip()
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()][:80]
    if not tickers:
        return jsonify({"results": {}})

    today = _ibkr_local_today()
    monday = today - _dt.timedelta(days=today.weekday())
    sunday = monday + _dt.timedelta(days=6)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        infos = list(executor.map(lambda t: _fetch_next_earnings(t), tickers))

    results = {}
    for ticker, info in zip(tickers, infos):
        date_iso = (info or {}).get("date")
        if not date_iso:
            results[ticker] = {"date": None, "days": None, "in_current_week": False}
            continue
        try:
            day = _dt.date.fromisoformat(date_iso)
        except ValueError:
            results[ticker] = {"date": None, "days": None, "in_current_week": False}
            continue
        results[ticker] = {
            "date": date_iso,
            "days": (day - today).days,
            # Settimana corrente in senso stretto: da lunedì a domenica di
            # questa settimana, non "entro sette giorni".
            "in_current_week": monday <= day <= sunday,
            "eps_estimated": (info or {}).get("eps_estimated"),
        }
    return jsonify({"results": results, "today": today.isoformat(),
                    "week": {"from": monday.isoformat(), "to": sunday.isoformat()}})


def _ibkr_holdings_payload(owner_email: str, analyze: bool = True) -> dict:
    """Posizioni IBKR presentate come le partecipazioni aggiunte a mano.

    Sta fuori dal route handler perché è anche la risposta esatta che riceve la
    pagina: la diagnostica deve poter guardare quella, non una ricostruzione
    che potrebbe divergere proprio dove serve.


    Ogni riga porta l'analisi Damodaran del titolo, la data della prossima
    trimestrale e gli ordini vivi che la riguardano, così la scheda in pagina
    è autosufficiente. I titoli su cui c'è solo un ordine pendente e nessuna
    posizione escono in una lista separata: sono un'esposizione potenziale,
    non ancora un investimento, e mescolarli falserebbe i pesi.

    `?analyze=0` salta i fondamentali quando serve solo la struttura.
    """
    doc = _ibkr_load_snapshot(owner_email)
    if not doc:
        return {"positions": [], "orders_only": [], "synced_at": None,
                "base_currency": _ibkr_base_currency()}

    snapshot = {"positions": doc.get("positions") or [], "orders": doc.get("orders") or []}
    earnings = doc.get("earnings") if isinstance(doc.get("earnings"), dict) else {}
    today = _ibkr_local_today()

    orders_by_symbol: Dict[str, List[dict]] = {}
    for order in snapshot["orders"]:
        if order.get("is_live"):
            orders_by_symbol.setdefault((order.get("symbol") or "").upper(), []).append(order)

    def earnings_bits(symbol: str) -> dict:
        info = earnings.get(symbol) or {}
        date_iso = info.get("date")
        days = None
        if date_iso:
            try:
                days = (_dt.date.fromisoformat(date_iso) - today).days
            except ValueError:
                days = None
        return {"earnings_date": date_iso, "earnings_in_days": days,
                "earnings_eps_estimated": info.get("eps_estimated"),
                "fmp_symbol": info.get("fmp_symbol")}

    def pending_buy(orders: List[dict], currency: Optional[str]) -> Optional[float]:
        values = [_ibkr_order_capital_base(o, currency) for o in orders]
        values = [v for v in values if v is not None]
        return sum(values) if values else None

    # Le posizioni del gateway sono già valorizzate in tempo reale da IBKR;
    # quelle del Flex no, e vanno rivalutate col listino.
    needs_revalue = (doc.get("positions_source") or "").startswith("flex")
    if needs_revalue:
        # Le quotazioni si scaldano in parallelo: in serie sarebbero tante
        # chiamate quante le posizioni, una dopo l'altra.
        from concurrent.futures import ThreadPoolExecutor
        wanted = {(earnings.get((p.get("symbol") or "").upper()) or {}).get("fmp_symbol")
                  for p in snapshot["positions"]}
        wanted.discard(None)
        if wanted:
            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(_fetch_quote_price, wanted))

    rows, positioned = [], set()
    for position in snapshot["positions"]:
        symbol = (position.get("symbol") or "").upper()
        if not symbol:
            continue
        positioned.add(symbol)
        orders = orders_by_symbol.get(symbol, [])
        row = {**position, **earnings_bits(symbol), "symbol": symbol}
        if needs_revalue:
            row = _ibkr_revalue_position(row)
        row.update({
            "market_value_base": _ibkr_market_value_base(row),
            "pending_buy_base": pending_buy(orders, position.get("currency")),
            "orders": orders,
        })
        rows.append(row)

    orders_only = []
    for symbol, orders in sorted(orders_by_symbol.items()):
        if symbol in positioned:
            continue
        first = orders[0]
        currency = first.get("currency")
        orders_only.append({
            "symbol": symbol, "name": first.get("name"),
            "currency": currency, "exchange": first.get("exchange"),
            **earnings_bits(symbol), "orders": orders,
            "pending_buy_base": pending_buy(orders, currency),
        })

    if analyze:
        # Un ticker IBKR non è un ticker FMP: si riusa il simbolo già risolto
        # per gli earnings invece di rifare la traduzione.
        from concurrent.futures import ThreadPoolExecutor
        targets = rows + orders_only
        with ThreadPoolExecutor(max_workers=8) as executor:
            analyses = list(executor.map(
                lambda r: (_analyze_portfolio_ticker_cached(r["fmp_symbol"])
                           if r.get("fmp_symbol") else
                           {"ticker": r["symbol"], "error": "simbolo non risolto su FMP"}),
                targets,
            ))
        for row, analysis in zip(targets, analyses):
            row["analysis"] = analysis

    return {
        "positions": rows,
        "orders_only": orders_only,
        "capital": _ibkr_capital_summary(doc, rows, orders_only),
        "base_currency": _ibkr_base_currency(),
        "synced_at": doc.get("synced_at"),
        "positions_synced_at": doc.get("positions_synced_at"),
        "positions_source": doc.get("positions_source"),
        "orders_freshness": _ibkr_orders_staleness(doc),
        "positions_freshness": _ibkr_positions_staleness(doc),
    }


@app.route('/api/ibkr/holdings', methods=['GET'])
@login_required
def api_ibkr_holdings():
    """Le posizioni per la pagina portafoglio. `?analyze=0` salta i
    fondamentali quando serve solo la struttura."""
    owner_email = _current_user_email()
    if not owner_email:
        return jsonify({"error": "no user"}), 401
    return jsonify(_ibkr_holdings_payload(
        owner_email, analyze=request.args.get("analyze") != "0"))


@app.route('/api/ibkr/earnings-alert', methods=['GET', 'POST'])
def api_ibkr_earnings_alert():
    """Alert earnings sull'ultimo snapshot salvato.

    GET richiede la sessione ed è di sola lettura (anteprima dalla pagina).
    POST richiede il bearer token e può notificare: serve al job per
    rimandare la notifica senza rifare la sync.
    """
    if request.method == 'POST':
        if not _ibkr_sync_authorized():
            return jsonify({"error": "unauthorized"}), 401
        data = request.get_json(silent=True) or {}
        owner_email = (_ibkr_str(data.get("owner_email"), 120) or "").lower() or _ibkr_default_owner_email()
        target_raw = _ibkr_str(data.get("target_date"), 10)
    else:
        if not _is_authenticated():
            return jsonify({"error": "unauthorized"}), 401
        data = {}
        owner_email = _current_user_email()
        target_raw = _ibkr_str(request.args.get("target_date"), 10)

    if not owner_email:
        return jsonify({"error": "owner_email non determinabile"}), 400

    target = None
    if target_raw:
        try:
            target = _dt.date.fromisoformat(target_raw)
        except ValueError:
            return jsonify({"error": "target_date non valida (attesa YYYY-MM-DD)"}), 400

    doc = _ibkr_load_snapshot(owner_email)
    if not doc:
        return jsonify({"error": "nessuno snapshot IBKR salvato per " + owner_email}), 404

    snapshot = {"positions": doc.get("positions") or [], "orders": doc.get("orders") or []}
    earnings = doc.get("earnings") if isinstance(doc.get("earnings"), dict) else None
    alert = _ibkr_alert_with_rendering(
        _ibkr_earnings_alert(snapshot, target=target, earnings=earnings),
        orders_freshness=_ibkr_orders_staleness(doc))

    telegram = _ibkr_maybe_notify(alert, data)

    return jsonify({"owner_email": owner_email, "synced_at": doc.get("synced_at"),
                    "should_email": bool(alert["count"]) or bool(data.get("notify_always")),
                    "telegram": telegram, "alert": alert})


# ============================================================================
# IBKR WEB API — OAuth 1.0a first party
# ============================================================================
#
# Con le credenziali OAuth generate dal Self-Service Portal l'app legge IBKR da
# sola: niente gateway, niente sessione Claude, niente PC acceso. Il flusso è
# quello documentato da IBKR e non è OAuth standard — c'è di mezzo uno scambio
# Diffie-Hellman:
#
#   1. si firma RSA-SHA256 una POST a /oauth/live_session_token, mettendo in
#      testa alla base string il token secret decifrato (il "prepend");
#   2. dalla risposta si ricava il segreto condiviso DH e da lì il live session
#      token, valido 24h;
#   3. tutte le chiamate successive si firmano HMAC-SHA256 con quel token;
#   4. gli endpoint /iserver vogliono in più una brokerage session aperta.
#
# Il passo di request/access token del protocollo OAuth NON va fatto: per il
# first party quei valori arrivano dal portale, e chiamarlo darebbe errore.

_IBKR_API_BASE = "https://api.ibkr.com/v1/api"
_IBKR_API_TIMEOUT = 20

_MONGO_IBKR_SESSION_COLLECTION = None

# Il live session token vale 24h: tenerlo in memoria basterebbe con un processo
# lungo, ma su Vercel ogni cold start ripartirebbe da capo rifacendo l'handshake
# (che è la parte lenta e con rate limit). Quindi memoria + Mongo.
_IBKR_LST_CACHE: Dict[str, Any] = {"token": None, "expires_ms": 0, "cookie": None}


def _get_mongo_ibkr_session_collection():
    """Lazy getter per il live session token OAuth.

    Il token è un segreto a 24h: sta su Mongo, cioè nello stesso perimetro di
    fiducia della connection string che ci arriva già dall'ambiente.
    """
    global _MONGO_CLIENT, _MONGO_IBKR_SESSION_COLLECTION
    if _MONGO_IBKR_SESSION_COLLECTION is not None:
        return _MONGO_IBKR_SESSION_COLLECTION
    if MongoClient is None:
        return None
    uri = (os.getenv("MONGODB_URI") or "").strip()
    if not uri:
        return None
    db_name = (os.getenv("MONGODB_DB") or "es_gamma_analyzer").strip()
    coll_name = (os.getenv("MONGODB_IBKR_SESSION_COLLECTION") or "ibkr_session").strip()
    try:
        if _MONGO_CLIENT is None:
            _MONGO_CLIENT = MongoClient(uri, serverSelectionTimeoutMS=2500, connectTimeoutMS=2500)
        coll = _MONGO_CLIENT[db_name][coll_name]
        try:
            coll.create_index("key", unique=True)
        except Exception:
            pass
        _MONGO_IBKR_SESSION_COLLECTION = coll
        return coll
    except Exception:
        return None


def _cryptography_available() -> bool:
    try:
        from cryptography.hazmat.primitives import serialization  # noqa: F401
        return True
    except Exception:
        return False


def _ibkr_api_env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def _ibkr_api_private_key(name: str):
    """Legge una chiave privata RSA da variabile d'ambiente.

    Accetta il PEM così com'è, con i newline resi come `\\n` (come capita
    incollandolo in un pannello web) oppure l'intero PEM in base64. Le tre forme
    circolano tutte, e sbagliare formato qui produce un errore di firma
    incomprensibile trecento righe più avanti.
    """
    raw = _ibkr_api_env(name)
    if not raw:
        return None
    if "-----BEGIN" not in raw:
        try:
            raw = base64.b64decode(raw).decode("utf-8")
        except Exception:
            return None
    pem = raw.replace("\\n", "\n").strip().encode("utf-8")
    try:
        from cryptography.hazmat.primitives import serialization
        return serialization.load_pem_private_key(pem, password=None)
    except Exception:
        return None


def _ibkr_api_configured() -> bool:
    return all([
        _ibkr_api_env("IBKR_CONSUMER_KEY"),
        _ibkr_api_env("IBKR_ACCESS_TOKEN"),
        _ibkr_api_env("IBKR_ACCESS_TOKEN_SECRET"),
        _ibkr_api_env("IBKR_DH_PRIME"),
        _ibkr_api_private_key("IBKR_SIGNATURE_KEY") is not None,
        _ibkr_api_private_key("IBKR_ENCRYPTION_KEY") is not None,
    ])


# ---------------------------------------------------------------------------
# Firma
# ---------------------------------------------------------------------------

def _ibkr_oauth_nonce() -> str:
    import secrets as _secrets
    import string as _string
    alphabet = _string.ascii_letters + _string.digits
    return "".join(_secrets.choice(alphabet) for _ in range(16))


def _ibkr_base_string(method: str, url: str, oauth_params: dict,
                      query_params: Optional[dict] = None, prepend: Optional[str] = None) -> str:
    """Base string della firma: parametri OAuth e di query ordinati
    lessicograficamente, uniti da '&' e percent-encodati, preceduti da metodo e
    URL. Il `prepend` — il token secret decifrato in esadecimale — va davanti a
    tutto, e vale solo per la richiesta del live session token."""
    params = {**oauth_params, **(query_params or {})}
    joined = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    base = "&".join([
        method.upper(),
        urllib.parse.quote_plus(url),
        urllib.parse.quote_plus(joined),
    ])
    return f"{prepend}{base}" if prepend else base


def _ibkr_rsa_sha256_signature(base_string: str, private_key) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    signature = private_key.sign(
        base_string.encode("utf-8"), padding.PKCS1v15(), hashes.SHA256())
    return urllib.parse.quote_plus(base64.b64encode(signature).decode("utf-8"))


def _ibkr_hmac_sha256_signature(base_string: str, live_session_token: str) -> str:
    digest = hmac.new(base64.b64decode(live_session_token),
                      base_string.encode("utf-8"), hashlib.sha256).digest()
    return urllib.parse.quote_plus(base64.b64encode(digest).decode("utf-8"))


def _ibkr_authorization_header(params: dict, realm: str) -> str:
    pairs = ", ".join(f'{k}="{v}"' for k, v in sorted(params.items()))
    return f'OAuth realm="{realm}", {pairs}'


def _ibkr_int_to_signed_bytes(value: int) -> bytes:
    """Intero → byte come lo serializza un BigInteger Java, che è quello che si
    aspetta IBKR: se il bit più alto è a 1 va aggiunto uno zero in testa,
    altrimenti il numero verrebbe letto come negativo."""
    hex_string = format(value, "x")
    if len(hex_string) % 2:
        hex_string = "0" + hex_string
    raw = bytes.fromhex(hex_string)
    if raw and raw[0] & 0x80:
        raw = b"\x00" + raw
    return raw


# ---------------------------------------------------------------------------
# Handshake
# ---------------------------------------------------------------------------

def _ibkr_api_call(method: str, path: str, oauth_headers: dict, realm: str,
                   query: Optional[dict] = None, body: Optional[dict] = None,
                   cookie: Optional[str] = None) -> tuple:
    """Chiamata HTTP firmata. Ritorna (status, payload|testo)."""
    url = f"{_IBKR_API_BASE}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {
        "Authorization": _ibkr_authorization_header(oauth_headers, realm),
        "Accept": "*/*",
        "User-Agent": "polaris/1.0",
        "Host": "api.ibkr.com",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    if cookie:
        headers["Cookie"] = cookie
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=_IBKR_API_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(raw)
            except ValueError:
                return resp.status, raw
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace") if hasattr(e, "read") else str(e)
        return e.code, raw
    except Exception as e:
        return 0, str(e)


def _ibkr_request_live_session_token() -> dict:
    """Esegue l'handshake DH e calcola il live session token.

    Ritorna {"token", "expires_ms", "valid"} oppure {"error"}.
    """
    import secrets as _secrets

    signature_key = _ibkr_api_private_key("IBKR_SIGNATURE_KEY")
    encryption_key = _ibkr_api_private_key("IBKR_ENCRYPTION_KEY")
    if signature_key is None or encryption_key is None:
        return {"error": "chiavi RSA mancanti o non leggibili "
                         "(IBKR_SIGNATURE_KEY / IBKR_ENCRYPTION_KEY)"}

    consumer_key = _ibkr_api_env("IBKR_CONSUMER_KEY")
    access_token = _ibkr_api_env("IBKR_ACCESS_TOKEN")
    access_token_secret = _ibkr_api_env("IBKR_ACCESS_TOKEN_SECRET")
    dh_prime_hex = _ibkr_api_env("IBKR_DH_PRIME")
    dh_generator = int(_ibkr_api_env("IBKR_DH_GENERATOR", "2") or 2)
    realm = _ibkr_api_env("IBKR_REALM", "limited_poa")

    # Il prepend è il token secret del portale, decifrato con la propria chiave
    # di encryption e reso esadecimale.
    try:
        from cryptography.hazmat.primitives.asymmetric import padding
        decrypted = encryption_key.decrypt(
            base64.b64decode(access_token_secret), padding.PKCS1v15())
        prepend = decrypted.hex()
    except Exception as e:
        return {"error": f"decifratura del token secret fallita: {e}"}

    dh_random = _secrets.randbits(256)
    dh_prime = int(dh_prime_hex, 16)
    dh_challenge = format(pow(dh_generator, dh_random, dh_prime), "x")

    oauth_params = {
        "diffie_hellman_challenge": dh_challenge,
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": _ibkr_oauth_nonce(),
        "oauth_signature_method": "RSA-SHA256",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": access_token,
    }
    base_string = _ibkr_base_string(
        "POST", f"{_IBKR_API_BASE}/oauth/live_session_token", oauth_params, prepend=prepend)
    oauth_params["oauth_signature"] = _ibkr_rsa_sha256_signature(base_string, signature_key)

    status, payload = _ibkr_api_call("POST", "/oauth/live_session_token", oauth_params, realm)
    if status != 200 or not isinstance(payload, dict):
        return {"error": f"live_session_token HTTP {status}: {str(payload)[:300]}"}

    try:
        shared_secret = pow(int(payload["diffie_hellman_response"], 16), dh_random, dh_prime)
        token = base64.b64encode(hmac.new(
            _ibkr_int_to_signed_bytes(shared_secret),
            bytes.fromhex(prepend),
            hashlib.sha1,
        ).digest()).decode("utf-8")
    except Exception as e:
        return {"error": f"calcolo del live session token fallito: {e}"}

    # IBKR rimanda la firma del token: se non combacia, la colpa è quasi sempre
    # di una chiave sbagliata, e scoprirlo ora è molto meglio che vedere 401
    # opachi su ogni chiamata successiva.
    expected = hmac.new(base64.b64decode(token), consumer_key.encode("utf-8"),
                        hashlib.sha1).hexdigest()
    valid = hmac.compare_digest(expected, payload.get("live_session_token_signature") or "")
    if not valid:
        return {"error": "firma del live session token non valida: controlla "
                         "consumer key e chiavi RSA"}

    return {"token": token, "expires_ms": int(payload.get("live_session_token_expiration") or 0),
            "valid": True}


def _ibkr_live_session_token(force: bool = False) -> Optional[str]:
    """Live session token valido, dalla cache o rinnovato.

    Si rinnova con 5 minuti di margine: un token che scade a metà della sequenza
    di chiamate produrrebbe un fallimento parziale, il caso più fastidioso da
    diagnosticare.
    """
    now_ms = int(time.time() * 1000)
    if not force and _IBKR_LST_CACHE.get("token") and _IBKR_LST_CACHE["expires_ms"] > now_ms + 300_000:
        return _IBKR_LST_CACHE["token"]

    coll = _get_mongo_ibkr_session_collection()
    if not force and coll is not None:
        try:
            doc = coll.find_one({"key": "live_session_token"})
            if doc and int(doc.get("expires_ms") or 0) > now_ms + 300_000:
                _IBKR_LST_CACHE.update({"token": doc["token"], "expires_ms": doc["expires_ms"]})
                return doc["token"]
        except Exception:
            pass

    result = _ibkr_request_live_session_token()
    if result.get("error"):
        _IBKR_LST_CACHE["last_error"] = result["error"]
        return None
    _IBKR_LST_CACHE.update({"token": result["token"], "expires_ms": result["expires_ms"],
                            "last_error": None})
    if coll is not None:
        try:
            coll.update_one({"key": "live_session_token"},
                            {"$set": {"key": "live_session_token", "token": result["token"],
                                      "expires_ms": result["expires_ms"], "renewed_at": time.time()}},
                            upsert=True)
        except Exception:
            pass
    return result["token"]


def _ibkr_signed_request(method: str, path: str, query: Optional[dict] = None,
                         body: Optional[dict] = None) -> tuple:
    """Chiamata a una risorsa protetta, firmata HMAC-SHA256 col live session
    token. I parametri di query entrano nella base string; il body JSON no."""
    token = _ibkr_live_session_token()
    if not token:
        return 0, _IBKR_LST_CACHE.get("last_error") or "live session token non disponibile"
    oauth_params = {
        "oauth_consumer_key": _ibkr_api_env("IBKR_CONSUMER_KEY"),
        "oauth_nonce": _ibkr_oauth_nonce(),
        "oauth_signature_method": "HMAC-SHA256",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": _ibkr_api_env("IBKR_ACCESS_TOKEN"),
    }
    base_string = _ibkr_base_string(method, f"{_IBKR_API_BASE}{path}", oauth_params, query)
    oauth_params["oauth_signature"] = _ibkr_hmac_sha256_signature(base_string, token)
    return _ibkr_api_call(method, path, oauth_params, _ibkr_api_env("IBKR_REALM", "limited_poa"),
                          query=query, body=body, cookie=_IBKR_LST_CACHE.get("cookie"))


def _ibkr_open_brokerage_session() -> dict:
    """Apre la sessione di brokeraggio, senza la quale gli endpoint /iserver
    rispondono ma a vuoto. `compete=true` perché IBKR ammette una sola sessione
    per username: senza, basta la TWS aperta a far fallire il job."""
    status, payload = _ibkr_signed_request(
        "POST", "/iserver/auth/ssodh/init", query={"publish": "true", "compete": "true"})
    if status != 200:
        return {"ok": False, "status": status, "detail": str(payload)[:300]}
    tickle_status, tickle = _ibkr_signed_request("POST", "/tickle")
    if tickle_status == 200 and isinstance(tickle, dict) and tickle.get("session"):
        _IBKR_LST_CACHE["cookie"] = f"api={tickle['session']}"
    return {"ok": True, "authenticated": bool(isinstance(payload, dict) and payload.get("authenticated"))}


# ---------------------------------------------------------------------------
# Lettura di posizioni e ordini
# ---------------------------------------------------------------------------

def _ibkr_api_account_id() -> Optional[str]:
    explicit = _ibkr_api_env("IBKR_ACCOUNT_ID")
    if explicit:
        return explicit
    status, payload = _ibkr_signed_request("GET", "/portfolio/accounts")
    if status == 200 and isinstance(payload, list) and payload:
        return payload[0].get("accountId") or payload[0].get("id")
    return None


def _ibkr_api_positions(account_id: str) -> List[dict]:
    """Posizioni aperte, normalizzate nella forma che si aspetta
    `_ibkr_normalize_payload`. Le pagine si scorrono finché tornano piene:
    IBKR ne restituisce 30 per volta."""
    out, page = [], 0
    while page < 10:
        status, payload = _ibkr_signed_request("GET", f"/portfolio/{account_id}/positions/{page}")
        if status != 200 or not isinstance(payload, list) or not payload:
            break
        for row in payload:
            if not isinstance(row, dict):
                continue
            out.append({
                "symbol": row.get("ticker") or row.get("contractDesc"),
                "name": row.get("name") or row.get("contractDesc"),
                "quantity": row.get("position"),
                "avg_price": row.get("avgPrice"),
                "market_price": row.get("mktPrice"),
                "market_value": row.get("mktValue"),
                "unrealized_pnl": row.get("unrealizedPnl"),
                "currency": row.get("currency"),
                "asset_class": row.get("assetClass"),
                "exchange": row.get("listingExchange"),
            })
        if len(payload) < 30:
            break
        page += 1
    return out


def _ibkr_api_orders() -> List[dict]:
    status, payload = _ibkr_signed_request("GET", "/iserver/account/orders")
    rows = payload.get("orders") if isinstance(payload, dict) else None
    if status != 200 or not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append({
            "order_id": row.get("orderId"),
            "symbol": row.get("ticker"),
            "name": row.get("companyName"),
            "side": row.get("side"),
            "order_type": row.get("orderType"),
            "status": row.get("status"),
            "quantity": row.get("totalSize"),
            "remaining": row.get("remainingQuantity"),
            "limit_price": row.get("price"),
            "stop_price": row.get("stop_price") or row.get("auxPrice"),
            "tif": row.get("timeInForce"),
            "description": row.get("orderDesc"),
            "exchange": row.get("listingExchange"),
        })
    return out


def _ibkr_api_fetch_snapshot() -> dict:
    """Snapshot completo letto direttamente da IBKR.
    Ritorna {"positions", "orders"} oppure {"error"}."""
    if not _ibkr_api_configured():
        return {"error": "credenziali IBKR Web API non configurate"}
    session = _ibkr_open_brokerage_session()
    if not session.get("ok"):
        return {"error": f"apertura sessione fallita ({session.get('status')}): "
                         f"{session.get('detail')}"}
    account_id = _ibkr_api_account_id()
    if not account_id:
        return {"error": "nessun account IBKR trovato"}
    return {"positions": _ibkr_api_positions(account_id),
            "orders": _ibkr_api_orders(), "account_id": account_id}


# ---------------------------------------------------------------------------
# Invio email (SMTP)
# ---------------------------------------------------------------------------

def _send_alert_email(subject: str, html_body: str, to_address: str) -> dict:
    """Manda la mail dell'alert via SMTP. Come Telegram, non solleva mai: se le
    credenziali mancano il job deve comunque completare la sync."""
    host = _ibkr_api_env("SMTP_HOST", "smtp.gmail.com")
    port = int(_ibkr_api_env("SMTP_PORT", "465") or 465)
    user = _ibkr_api_env("SMTP_USER")
    password = _ibkr_api_env("SMTP_PASSWORD")
    if not user or not password or not to_address:
        return {"sent": False, "error": "SMTP_USER / SMTP_PASSWORD / destinatario non configurati"}
    try:
        import smtplib
        import ssl
        from email.message import EmailMessage
        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = user
        message["To"] = to_address
        message.set_content("Questa notifica richiede un client che mostri l'HTML.")
        message.add_alternative(html_body, subtype="html")
        context = ssl.create_default_context()
        if port == 587:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.starttls(context=context)
                server.login(user, password)
                server.send_message(message)
        else:
            with smtplib.SMTP_SSL(host, port, timeout=20, context=context) as server:
                server.login(user, password)
                server.send_message(message)
        return {"sent": True, "error": None}
    except Exception as e:
        return {"sent": False, "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Rotte
# ---------------------------------------------------------------------------

def _ibkr_fetch_positions_any_source() -> dict:
    """Posizioni dalla sorgente migliore disponibile.

    La Web API OAuth, se mai verrà abilitata sul conto, porta anche gli ordini
    e va preferita. Altrimenti si ripiega sul Flex Web Service, che di
    posizioni ne dà quante ne servono ma di ordini niente.
    """
    if _ibkr_api_configured():
        fetched = _ibkr_api_fetch_snapshot()
        if not fetched.get("error"):
            return {**fetched, "source": "webapi"}
        webapi_error = fetched["error"]
    else:
        webapi_error = None

    flex = _flex_fetch_positions()
    if flex.get("error"):
        return {"error": flex["error"], "error_code": flex.get("error_code"),
                "hint": flex.get("hint"), "webapi_error": webapi_error}
    # I cambi ufficiali di IBKR, quando la query li porta, sono migliori di
    # quelli chiesti a FMP: sono gli stessi con cui il conto è valorizzato.
    for currency, rate in (flex.get("rates") or {}).items():
        _FX_CACHE[currency] = {"ts": time.time(), "rate": rate}
    return {"positions": flex["positions"], "orders": None,
            "account": flex.get("account"),
            "report_date": flex.get("report_date"),
            "source": "flex+trades" if flex.get("trades_applied") else "flex",
            "trades_applied": flex.get("trades_applied"),
            "trades_seen": flex.get("trades_seen"),
            "trades_query_configured": flex.get("trades_query_configured"),
            "trades_error": flex.get("trades_error"),
            "flex_sections": {
                "nav": bool(flex.get("account")),
                "rates": len(flex.get("rates") or {}),
                "positions": len(flex.get("positions") or []),
            },
            "webapi_error": webapi_error}


def _ibkr_run_daily_job(notify_always: bool = False, notify: bool = True,
                        pnl_mode: str = "auto") -> dict:
    """Il giro completo: aggiorna le posizioni, calcola l'alert del giorno dopo
    e notifica. Gli ordini non li tocca a meno che la sorgente non li porti —
    restano quelli depositati dal gateway locale, dichiarando quanto sono
    vecchi invece di far finta che siano freschi."""
    fetched = _ibkr_fetch_positions_any_source()
    if fetched.get("error"):
        return {"status": "error", "error": fetched["error"],
                "error_code": fetched.get("error_code"),
                "hint": fetched.get("hint"),
                "webapi_error": fetched.get("webapi_error")}

    owner_email = _ibkr_default_owner_email()
    normalized = _ibkr_normalize_payload(fetched)
    orders = normalized["orders"] if fetched.get("orders") is not None else None
    # Il Flex fotografa una chiusura passata: la data del dato è quella del
    # report, non l'istante in cui lo scarichiamo.
    as_of = _ibkr_report_date_to_epoch(fetched.get("report_date"))
    merged = _ibkr_store_snapshot(owner_email, normalized["positions"], orders,
                                  source=fetched.get("source") or "cron",
                                  positions_as_of=as_of,
                                  account=fetched.get("account"))
    if merged is None:
        return {"status": "error", "error": "mongo non disponibile: snapshot non salvato"}

    # Storico giornaliero per il calendario. Ha un ritmo suo — si muove solo
    # quando si chiude qualcosa — quindi si rilegge ogni tot ore e non a ogni
    # giro, e un suo fallimento non deve portarsi via le posizioni appena
    # salvate: quelle sono il motivo per cui il job esiste.
    if pnl_mode == "off":
        pnl = {"status": "skipped", "reason": "disattivato dalla richiesta"}
    else:
        try:
            if pnl_mode == "reset":
                _ibkr_reset_pnl_days(owner_email)
            pnl = _ibkr_refresh_pnl_history(
                owner_email, force=pnl_mode in ("force", "reset"))
            if pnl_mode == "reset":
                pnl["reset"] = True
        except Exception as error:
            pnl = {"status": "error", "error": str(error)[:200]}

    snapshot = {"positions": merged["positions"], "orders": merged["orders"]}
    freshness = _ibkr_orders_staleness(merged)
    alert = _ibkr_alert_with_rendering(
        _ibkr_earnings_alert(snapshot, earnings=merged.get("earnings")),
        orders_freshness=freshness)

    should_notify = notify and (bool(alert["count"]) or notify_always)
    telegram = _ibkr_maybe_notify(alert, {"notify": notify, "notify_always": notify_always})
    email = {"sent": False, "error": "nessun earning da segnalare"}
    if should_notify:
        email = _send_alert_email(
            alert["subject"], alert["email_html"],
            _ibkr_api_env("ALERT_EMAIL_TO") or owner_email)

    return {
        "status": "ok",
        "source": fetched.get("source"),
        "account_id": fetched.get("account_id"),
        "report_date": fetched.get("report_date"),
        "trades_applied": fetched.get("trades_applied"),
        "trades_seen": fetched.get("trades_seen"),
        "trades_query_configured": fetched.get("trades_query_configured"),
        "trades_error": fetched.get("trades_error"),
        "flex_sections": fetched.get("flex_sections"),
        "pnl_history": pnl,
        "skipped": merged.get("skipped"),
        "position_symbols": sorted({(p.get("symbol") or "") for p in snapshot["positions"]}),
        "positions": len(snapshot["positions"]),
        "orders": len(snapshot["orders"]),
        "live_orders": sum(1 for o in snapshot["orders"] if o.get("is_live")),
        "orders_freshness": freshness,
        "stored": True,
        "target_date": alert["target_date"],
        "count": alert["count"],
        "symbols": [i["symbol"] for i in alert["items"]],
        "unresolved": alert["unresolved"],
        "telegram": telegram,
        "email": email,
    }


def _ibkr_cron_authorized() -> bool:
    """Vercel Cron manda `Authorization: Bearer $CRON_SECRET`. Si accetta anche
    il token di sync, per poter lanciare il giro a mano."""
    expected = _ibkr_api_env("CRON_SECRET")
    if expected:
        header = (request.headers.get("Authorization") or "").strip()
        token = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if token and hmac.compare_digest(token, expected):
            return True
    return _ibkr_sync_authorized()


# Prefissi delle variabili di cui il cron può elencare i *nomi* con `?diag=1`.
# Serve a distinguere "non l'ho messa" da "l'ho messa con un nome diverso" e da
# "l'ho messa sull'ambiente sbagliato": da fuori le tre si somigliano tutte, e
# su Vercel una variabile aggiunta dopo l'ultimo deploy non esiste comunque
# finché non se ne fa un altro. Solo i nomi, mai i valori.
_IBKR_DIAG_PREFIXES = ("IBKR_", "FLEX_")


def _flex_report_shape() -> dict:
    """Che cosa contiene davvero il report della query P&L.

    Serve a rispondere per prove e non per memoria a domande del tipo "il Flex
    ha lo stop che avevo messo?": la risposta dipende da quali sezioni e quali
    campi sono spuntati nel portale, e dall'esterno non si vede. Riporta i nomi
    degli elementi e degli attributi — mai i valori, tranne le enumerazioni
    (tipo d'ordine, apertura/chiusura), che non sono dati sensibili e sono
    proprio quelle da guardare.
    """
    query_id = _ibkr_pnl_query_id()
    if not query_id:
        return {"error": "nessuna query P&L configurata"}
    fetched = _flex_fetch_statement(query_id)
    if fetched.get("error"):
        return fetched
    xml = fetched["xml"]

    elementi: Dict[str, int] = {}
    for name in re.findall(r"<(\w+)[\s/>]", xml):
        elementi[name] = elementi.get(name, 0) + 1

    rows = _flex_trade_rows(xml)
    attributi = sorted({k for row in rows for k in row})

    def distinti(chiave: str) -> dict:
        out: Dict[str, int] = {}
        for row in rows:
            value = (row.get(chiave) or "").strip().upper() or "(vuoto)"
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:12])

    return {
        "elementi_nel_report": dict(sorted(elementi.items(), key=lambda kv: -kv[1])[:40]),
        "righe_eseguiti": len(rows),
        "attributi_sugli_eseguiti": attributi,
        "tipi_ordine": distinti("orderType"),
        "apertura_chiusura": distinti("openCloseIndicator"),
        "codici": distinti("notes") or distinti("code"),
    }


def _ibkr_positions_diagnostics(owner_email: str) -> dict:
    """Perché una posizione mostra il Gain/Loss a trattino.

    Il P&L delle posizioni che arrivano dal Flex non è un dato di IBKR: è
    ricalcolato col listino, e la catena ha tre anelli — il simbolo FMP
    risolto dalla mappa earnings, la quotazione, il prezzo di carico. Se ne
    salta uno il numero sparisce, e da fuori i tre casi si somigliano tutti.
    """
    doc = _ibkr_load_snapshot(owner_email) or {}
    earnings = doc.get("earnings") if isinstance(doc.get("earnings"), dict) else {}
    source = (doc.get("positions_source") or "")
    righe = []
    for position in (doc.get("positions") or []):
        symbol = (position.get("symbol") or "").upper()
        fmp_symbol = (earnings.get(symbol) or {}).get("fmp_symbol")
        quote = _fetch_quote_price(fmp_symbol) if fmp_symbol else None
        row = {**position, "fmp_symbol": fmp_symbol}
        revalued = _ibkr_revalue_position(row) if source.startswith("flex") else row
        righe.append({
            "symbol": symbol,
            "quantity": position.get("quantity"),
            "avg_price": position.get("avg_price"),
            "pnl_in_archivio": position.get("unrealized_pnl"),
            "derivata_dagli_eseguiti": bool(position.get("derived_from_trades")),
            "fmp_symbol": fmp_symbol,
            "quotazione_fmp": quote,
            "pnl_dopo_rivalutazione": revalued.get("unrealized_pnl"),
            "rivalutata": bool(revalued.get("revalued")),
            "rivalutazione_scartata": bool(revalued.get("revalue_rejected")),
        })
    return {
        "positions_source": source or None,
        "rivalutazione_attiva": source.startswith("flex"),
        "fmp_configurata": bool(_ibkr_api_env("FMP_API_KEY")),
        "simboli_senza_fmp": [r["symbol"] for r in righe if not r["fmp_symbol"]],
        "posizioni": righe,
    }


def _ibkr_env_diagnostics() -> dict:
    names = sorted(k for k in os.environ
                   if k.startswith(_IBKR_DIAG_PREFIXES))
    return {
        "variabili_viste": names,
        # Il valore no, ma sapere se è vuota o solo spazi sì: è la differenza
        # fra "manca" e "c'è ma non contiene niente".
        "pnl_query_valorizzata": bool(_ibkr_pnl_query_id()),
        "vercel_env": os.getenv("VERCEL_ENV"),
        "commit": (os.getenv("VERCEL_GIT_COMMIT_SHA") or "")[:8] or None,
    }


@app.route('/api/ibkr/cron', methods=['GET', 'POST'])
def api_ibkr_cron():
    """Job giornaliero: legge IBKR, salva, notifica. Lo chiama Vercel Cron."""
    if not _ibkr_cron_authorized():
        return jsonify({"error": "unauthorized"}), 401
    diag = (request.args.get("diag") or "").strip().lower()
    if diag == "1":
        return jsonify({"env": _ibkr_env_diagnostics()})
    if diag == "flex":
        return jsonify({"report": _flex_report_shape()})
    if diag == "pos":
        return jsonify({"posizioni": _ibkr_positions_diagnostics(
            _ibkr_default_owner_email())})
    if diag == "holdings":
        # I campi esatti su cui la pagina disegna il Gain/Loss e la barra del
        # capitale. Guardare la risposta vera è l'unico modo di distinguere
        # "il numero non c'è" da "il numero c'è ma non viene disegnato".
        payload = _ibkr_holdings_payload(_ibkr_default_owner_email(), analyze=False)
        return jsonify({
            "capital": payload.get("capital"),
            "base_currency": payload.get("base_currency"),
            "positions_source": payload.get("positions_source"),
            "posizioni": [{k: row.get(k) for k in
                           ("symbol", "quantity", "avg_price", "market_price",
                            "market_value", "market_value_base", "unrealized_pnl",
                            "daily_pnl", "currency", "revalued", "revalue_rejected")}
                          for row in (payload.get("positions") or [])],
        })
    notify_always = request.args.get("notify_always") == "1"
    # `pnl=force` rilegge lo storico anche se non è ancora scaduto, `pnl=0` lo
    # salta del tutto. Serve dopo aver configurato o corretto la query Flex:
    # senza, per vedere l'effetto di una modifica bisognerebbe aspettare che
    # scada la finestra di quattro ore.
    # `pnl=reset` butta lo storico e lo rilegge da capo: serve quando si è
    # cambiata la query e le giornate raccolte con quella vecchia sono
    # incomplete, perché la fusione per data da sola non le toglierebbe mai.
    pnl_mode = {"force": "force", "1": "force", "0": "off", "reset": "reset"}.get(
        (request.args.get("pnl") or "").strip().lower(), "auto")
    # `notify=0` per i richiami durante la giornata: aggiornano le posizioni ma
    # non devono rimandare l'alert earnings ogni mezz'ora. La notifica resta
    # attaccata al giro serale.
    result = _ibkr_run_daily_job(notify_always=notify_always,
                                 notify=request.args.get("notify") != "0",
                                 pnl_mode=pnl_mode)
    return jsonify(result), (200 if result.get("status") == "ok" else 502)


@app.route('/api/ibkr/flex-status', methods=['GET'])
@login_required
def api_ibkr_flex_status():
    """Diagnostica delle credenziali Flex, senza esporle.

    IBKR risponde 1020 a situazioni molto diverse, e le due più frequenti si
    riconoscono dalla *forma* dei valori configurati — un query id non numerico
    è il nome della query copiato al posto del numero, un token troppo corto è
    un copia-incolla tagliato. Qui si riportano lunghezze e primi caratteri, mai
    i valori interi.
    """
    if not _is_admin():
        return jsonify({"error": "forbidden"}), 403

    token = _ibkr_api_env("IBKR_FLEX_TOKEN")
    query = _ibkr_api_env("IBKR_FLEX_QUERY_ID")
    shape = {
        "token_configurato": bool(token),
        "token_lunghezza": len(token),
        "token_anteprima": (token[:3] + "…" + token[-3:]) if len(token) > 8 else None,
        "token_solo_cifre": token.isdigit() if token else None,
        "query_id": query,
        "query_id_numerico": query.isdigit() if query else None,
    }
    note = []
    if token and not token.isdigit():
        note.append("il token del Flex Web Service è normalmente tutto numerico: "
                    "controlla di non aver incollato altro")
    if query and not query.isdigit():
        note.append("il query id deve essere il NUMERO della query, non il nome "
                    "che le hai dato")
    if token and len(token) < 15:
        note.append("token più corto del previsto: forse è stato troncato")

    if not token or not query:
        return jsonify({"shape": shape, "note": note,
                        "esito": "credenziali non configurate"})

    shape["trades_query_id"] = _ibkr_api_env("IBKR_FLEX_TRADES_QUERY_ID") or None
    if not shape["trades_query_id"]:
        note.append("IBKR_FLEX_TRADES_QUERY_ID non configurato: senza la query "
                    "Trade Confirmation le posizioni restano ferme alla chiusura "
                    "precedente e gli eseguiti di oggi non compaiono")
    shape["pnl_query_id"] = _ibkr_pnl_query_id() or None
    if not shape["pnl_query_id"]:
        note.append("IBKR_FLEX_PNL_QUERY_ID non configurato: il calendario P&L "
                    "ripiega sulla query di oggi e lo storico si costruisce solo "
                    "da qui in avanti. Serve una Activity Flex con la sezione "
                    "Trades su un periodo lungo e il campo fifoPnlRealized")

    probe = _flex_fetch_positions()
    if probe.get("error"):
        return jsonify({"shape": shape, "note": note, "esito": "fallito",
                        "errore": probe["error"], "codice": probe.get("error_code"),
                        "diagnosi": probe.get("hint"),
                        # La prova parte dai server di Vercel: se le stesse
                        # credenziali funzionano dal PC dell'utente, la causa è
                        # una restrizione per IP sul token.
                        "prova_dal_tuo_pc": "python tools/ibkr_flex_check.py"})

    # Quali sezioni la query porta davvero: è la domanda che ci si pone dopo
    # averla configurata, e la risposta non si vede da nessun'altra parte.
    sezioni = {
        "Open Positions": len(probe["positions"]),
        "NAV Summary in Base": bool(probe.get("account")),
        "Currency Conversion Rate": len(probe.get("rates") or {}),
        "Trade Confirmation": (probe.get("trades_seen")
                               if probe.get("trades_seen") is not None else "query non configurata"),
    }
    mancanti = []
    if not probe.get("account"):
        mancanti.append("aggiungi la sezione 'Net Asset Value (NAV) Summary in Base' "
                        "alla query: senza, il capitale arriva solo dal gateway")
    if not (probe.get("rates") or {}):
        mancanti.append("aggiungi la sezione 'Currency Conversion Rate' alla query: "
                        "senza, i cambi vengono chiesti a FMP")
    return jsonify({"shape": shape, "note": note + mancanti, "esito": "ok",
                    "sezioni": sezioni,
                    "eseguiti_applicati": probe.get("trades_applied"),
                    "errore_eseguiti": probe.get("trades_error"),
                    "report_date": probe.get("report_date")})


@app.route('/api/ibkr/oauth-status', methods=['GET'])
@login_required
def api_ibkr_oauth_status():
    """Diagnostica dell'handshake OAuth, passo per passo.

    Serve durante la configurazione: senza, un 401 di IBKR non dice se ha
    sbagliato la chiave di firma, quella di encryption, il consumer key o il
    primo DH. Riservata agli admin perché espone il dettaglio degli errori.
    """
    if not _is_admin():
        return jsonify({"error": "forbidden"}), 403

    steps = {
        "consumer_key": bool(_ibkr_api_env("IBKR_CONSUMER_KEY")),
        "access_token": bool(_ibkr_api_env("IBKR_ACCESS_TOKEN")),
        "access_token_secret": bool(_ibkr_api_env("IBKR_ACCESS_TOKEN_SECRET")),
        "dh_prime": bool(_ibkr_api_env("IBKR_DH_PRIME")),
        "signature_key_loaded": _ibkr_api_private_key("IBKR_SIGNATURE_KEY") is not None,
        "encryption_key_loaded": _ibkr_api_private_key("IBKR_ENCRYPTION_KEY") is not None,
        "realm": _ibkr_api_env("IBKR_REALM", "limited_poa"),
    }
    if not _ibkr_api_configured():
        return jsonify({"configured": False, "steps": steps,
                        "hint": "completa le variabili IBKR_* prima di provare l'handshake"})

    lst = _ibkr_request_live_session_token()
    steps["live_session_token"] = "ok" if lst.get("token") else lst.get("error")
    if not lst.get("token"):
        return jsonify({"configured": True, "steps": steps})

    _IBKR_LST_CACHE.update({"token": lst["token"], "expires_ms": lst["expires_ms"]})
    session_result = _ibkr_open_brokerage_session()
    steps["brokerage_session"] = ("ok" if session_result.get("ok")
                                  else f"{session_result.get('status')}: {session_result.get('detail')}")
    account_id = _ibkr_api_account_id() if session_result.get("ok") else None
    steps["account_id"] = account_id or "non trovato"
    if account_id:
        steps["positions"] = len(_ibkr_api_positions(account_id))
        steps["orders"] = len(_ibkr_api_orders())
    return jsonify({"configured": True, "steps": steps})


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
