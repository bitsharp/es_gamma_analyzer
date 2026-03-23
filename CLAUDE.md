# ES Gamma Analyzer — Claude Guide

## Project Overview

Financial trading analysis tool for ES (E-mini S&P 500) futures. Extracts gamma exposure data from PDFs to identify key market levels (gamma flip, support, resistance) and volatility regime.

## Tech Stack

- **Backend**: Python 3.9+, Flask 3.0
- **Database**: MongoDB Atlas (PyMongo)
- **Auth**: Google OAuth 2.0 (Authlib)
- **PDF Parsing**: pdfplumber (primary), PyMuPDF (fallback)
- **Data**: pandas, numpy, yfinance
- **Frontend**: Vanilla HTML/CSS/JS, Font Awesome 6
- **Deploy**: Vercel (serverless via `api/index.py`)

## Project Structure

```
app.py                  # Main Flask app (~6000 lines, 118 functions)
api/index.py            # Vercel serverless entry point
templates/
  index.html            # Main dashboard
  checklist.html        # Trading checklist
  journal_*.html        # Trading journal views
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

## Local Development

```bash
python -m venv venv
source venv/Scripts/activate      # Windows
pip install -r requirements.txt
cp .env.example .env               # Fill in credentials
python app.py                      # http://localhost:5005
```

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

## Architecture Patterns

- **Lazy init**: OAuth and MongoDB clients initialized on first use
- **In-memory cache with TTL**: API calls cached to avoid rate limits
- **Fallback chains**: SPX → SPY for price data; Stooq → NASDAQ
- **Admin allowlist**: `ADMIN_EMAILS` env var controls admin access
- File uploads: 16MB max, `werkzeug.utils.secure_filename`

## Deployment (Vercel)

```bash
vercel          # Manual deploy via CLI
```

Or connect GitHub repo to Vercel dashboard for auto-deploy on push. Dependencies defined in both `requirements.txt` and `api/pyproject.toml` (for `uv`).

## Reference Docs

- [README.md](README.md) — Full guide in Italian (setup, concepts, deploy)
- [STRUCTURE.md](STRUCTURE.md) — Maps all 118 functions to code sections
- [APP_INDEX.md](APP_INDEX.md) — Application index reference
