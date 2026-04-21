# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`es_gamma_analyzer` is a Flask web application that analyzes 0DTE/1DTE gamma exposure from broker PDF reports (e.g. `OpenInterest-*.pdf`) to assist trading the E-mini S&P 500 future (ES). It extracts strike-level gamma from PDFs, computes the gamma flip level, identifies key support/resistance strikes, and serves a dashboard that overlays real-time market data (SPX, ES, SPY, XSP, NVDA, MSFT, AAPL, GOOG, AMZN). It also hosts a trading journal and a pre-trade checklist.

The README is in Italian; comments and most user-facing strings are in Italian too. Preserve that when editing UI text or docstrings.

## Tech Stack

- **Backend**: Python 3.9+, Flask 3.0
- **Database**: MongoDB Atlas (PyMongo)
- **Auth**: Google OAuth 2.0 (Authlib)
- **PDF Parsing**: pdfplumber (primary), PyMuPDF (fallback)
- **Data**: pandas, numpy, yfinance
- **Frontend**: Vanilla HTML/CSS/JS, Font Awesome 6
- **Deploy**: Vercel (serverless via `api/index.py`)

## Run / develop

```bash
source .venv/bin/activate          # or: source venv/Scripts/activate on Windows
python app.py                      # serves on http://localhost:5005
```

There are no tests, no linter config, and no build step. The only runtime is `python app.py`. Flask's reloader is on (`debug=True`), so code edits take effect automatically except for OAuth/Mongo client state (which is re-initialized lazily — see "Lazy initialization" below).

Dependencies: `pip install -r requirements.txt`. Python 3.9+ (the repo uses `.venv/` at Python 3.9). `api/requirements.txt` just re-exports the root file for Vercel.

`.env` is loaded automatically via `python-dotenv` (see `.env.example`). Required for a working instance: `FLASK_SECRET_KEY`, `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `MONGODB_URI`, `ADMIN_EMAILS`.

## Project Structure

```
app.py                  # Main Flask app (~6000 lines, 118 functions)
api/index.py            # Vercel serverless entry point
templates/
  index.html            # Main dashboard (~5000 lines of inline JS/CSS)
  checklist.html        # Pre-trade checklist
  journal_*.html        # TradeZella-style trade log views
  stocks.html           # 13F superinvestor tracking
  admin.html            # Login sessions / admin view
  login.html            # Google OAuth login
uploads/                # Uploaded PDFs (gitignored)
requirements.txt        # Pinned dependencies
vercel.json             # Vercel routing (all → api/index.py)
.env.example            # Config template
```

## Key Routes

| Route | Description |
|-------|-------------|
| `GET /` | Redirect to dashboard or login |
| `GET /dashboard` | Main analysis dashboard |
| `POST /analyze` | PDF upload + gamma analysis |
| `GET /api/health` | Health check (public) |
| `GET /api/es-price` | ES Futures live price |
| `GET /api/pressure-history` | Historical pressure points |
| `POST /api/pressure-point` | Save pressure point |
| `GET /api/last-analysis` | Load user's last analysis |
| `GET /api/cot-sp500` | Weekly CFTC COT report (S&P 500) |

## Environment Variables

```
MONGODB_URI=             # MongoDB Atlas connection string
GOOGLE_CLIENT_ID=        # Google OAuth client ID
GOOGLE_CLIENT_SECRET=    # Google OAuth client secret
FLASK_SECRET_KEY=        # Session secret (random string)
ADMIN_EMAILS=            # Comma-separated admin email list
```

## MongoDB Collections

| Collection | Purpose | TTL |
|------------|---------|-----|
| `pressure_points` | Market pressure levels | 36h |
| `login_sessions` | Auth session logs | 90d |
| `last_analysis` | Per-user last PDF analysis | — |
| `es_spx_conversions` | ES/SPX conversion data | — |

## Deployment

Deployed on Vercel as a Python Serverless Function. Entry point: [api/index.py](api/index.py), routing defined in [vercel.json](vercel.json) (all paths → Flask). On Vercel (`VERCEL=1`):

- Filesystem is read-only except `/tmp`; uploads fall back to `/tmp/uploads` (see `get_upload_folder()` in [app.py](app.py)).
- `SESSION_COOKIE_SECURE` is forced on.
- `ADMIN_EMAILS` is mandatory (locally it's optional).

State (pressure history, login events, last analysis per user, ES/SPX conversions, checklist, trades) lives in MongoDB Atlas. Collections are configured via `MONGODB_*_COLLECTION` env vars and created lazily on first use.

```bash
vercel          # Manual deploy via CLI
```

Or connect GitHub repo to Vercel dashboard for auto-deploy on push.

## Architecture

Everything runs out of a single ~6000-line file: [app.py](app.py). It is deliberately monolithic and organized by banner comments (`# ==== SECTION NAME ====`). The section boundaries roughly are:

1. Imports & Flask config (`ProxyFix` for Vercel forwarded headers, cookie hardening)
2. Authentication & session — Google OAuth via Authlib, `_is_authenticated`, `_is_admin`, `login_required` decorator, `_require_login` before_request guard
3. MongoDB helpers — one module-level client, one getter per collection, each protected by try/except so Mongo outages never break request handling
4. File system helpers — upload folder selection with writable-dir fallbacks
5. Cache globals — per-symbol in-memory dicts with TTL; used everywhere instead of ad-hoc caches
6. Data parsing utilities — `_parse_pdf_number`, Nasdaq/Stooq parsers
7. Market data fetchers — `get_<symbol>_snapshot_cached()` functions fetch option chains from Nasdaq (primary) or Stooq/yfinance (fallbacks)
8. PDF extraction — two backends: `pdfplumber` for table-based extraction and `PyMuPDF` (`fitz`) for coordinate-based fallback. `extract_0dte_data`, `extract_1dte_data`, and `extract_nearest_positive_dte_data` are the main entry points
9. Gamma analysis core — `analyze_0dte`, `_find_gamma_flip`, `_identify_key_levels`
10. Web routes — split by concern: auth/admin, API endpoints (market data + Mongo CRUD), the main `/analyze` upload endpoint, the trading journal (`/journal*`), and the checklist (`/checklist`, `/api/checklist/*`)
11. Entry point

Templates live in [templates/](templates/): `index.html` is the main dashboard (~5000 lines of inline JS/CSS — the frontend lives here, not in separate assets). `journal_*.html` renders the TradeZella-style trade log. `checklist.html` is the pre-market routine. `admin.html` shows login sessions.

There is **no separate Python module for extraction/analysis** even though the README references `pdf_extractor.py`, `gamma_analyzer.py`, `report_generator.py`, `main.py` — those were consolidated into `app.py`. Ignore that part of the README.

### Cross-cutting patterns

- **Lazy initialization** — OAuth client (`_ensure_google_oauth_registered`) and each Mongo collection getter cache their handle in a module global on first use. This is important because the Flask debug reloader and Vercel's cold-start model both re-import the module; initializing at import time would cost latency and leak credentials into error paths.
- **In-memory cache with TTL** — API calls cached to avoid rate limits.
- **Fallback chains** — market data calls try the preferred source (Nasdaq JSON) and fall back transparently (SPX→SPY, stooq→Nasdaq, yfinance last). When adding a new symbol, copy an existing `get_<symbol>_snapshot_cached` and preserve the chain shape.
- **Silent Mongo/OAuth failures** — helpers wrap every Mongo/OAuth call in try/except and return `None`/empty. Never let a Mongo outage raise into a request handler.
- **Per-user scoping** — `_current_user_key()` derives a stable Mongo key from the authenticated email. All per-user persistence (`last_analysis`, `trading_checklist`, journal trades) must scope reads/writes by this key.
- **Admin allowlist** — `ADMIN_EMAILS` env var controls admin access.
- File uploads: 16 MB max, `werkzeug.utils.secure_filename`.

## Conventions

- When adding a new ticker snapshot endpoint: create a `_<SYMBOL>_SNAPSHOT_CACHE` global, a `get_<symbol>_snapshot_cached()` with the same signature as existing ones, and a `/api/<symbol>-snapshot` route. Follow the caching TTL and fallback pattern of `get_spx_snapshot_cached`.
- New Mongo collections: add `MONGODB_<NAME>_COLLECTION` env var with a sensible default, write a lazy getter like `_get_mongo_<name>_collection()`, and wrap callers in try/except.
- `/admin/*` routes must be gated by `_is_admin()`, not just `login_required`.
- File uploads are capped at 16 MB (`MAX_CONTENT_LENGTH`). Filenames must go through `secure_filename`. Prefer `get_upload_folder()` over hardcoded `uploads/`.

## Reference Docs

- [README.md](README.md) — Full guide in Italian (setup, concepts, deploy)
- [STRUCTURE.md](STRUCTURE.md) — Maps all 118 functions to code sections
- [APP_INDEX.md](APP_INDEX.md) — Application index reference

## Notes on stale docs

[STRUCTURE.md](STRUCTURE.md), [APP_INDEX.md](APP_INDEX.md), and [CODE_ORGANIZATION.md](CODE_ORGANIZATION.md) document the section layout but their line numbers are from when `app.py` was ~2870 lines. It is now ~6000 lines (journal + checklist + more market data APIs were added). Use the banner comments and the route list for navigation, not the line numbers in those files.
