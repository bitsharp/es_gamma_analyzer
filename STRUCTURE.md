# Struttura del Codice - app.py

Questo documento descrive l'organizzazione del codice in `app.py` per facilitare la manutenzione.

## 📋 Indice delle Sezioni

### 1. IMPORTS & SETUP (Righe 1-75)
- Import delle librerie Python
- Caricamento variabili d'ambiente (.env)
- Import condizionali (OAuth, PyMongo)
- Costanti globali

### 2. FLASK APPLICATION CONFIG (Righe 53-72)
- Inizializzazione Flask app
- Configurazione middleware (ProxyFix)
- Secret key e cookie security
- OAuth initialization

### 3. AUTHENTICATION & SESSION (Righe 73-190)
**Funzioni:**
- `_ensure_google_oauth_registered()` - Registrazione lazy OAuth client
- `_google_oauth_missing_vars()` - Diagnostica variabili OAuth mancanti
- `_is_authenticated()` - Check autenticazione utente
- `_is_admin()` - Verifica permessi admin
- `_wants_json()` - Determina se risposta deve essere JSON
- `_require_login()` - Middleware autenticazione (before_request)
- `login_required()` - Decorator per route protette
- `_current_user_key()` - Genera chiave MongoDB per utente

### 4. MONGODB HELPERS (Righe 187-425)
**Variabili Globali:**
- `_MONGO_CLIENT` - Client MongoDB condiviso
- `_MONGO_COLLECTION` - Collection pressure_points
- `_MONGO_LOGIN_COLLECTION` - Collection login_sessions
- `_MONGO_LAST_ANALYSIS_COLLECTION` - Collection last_analysis

**Funzioni:**
- `_get_mongo_collection()` - Collection pressure points (36h TTL)
- `_get_mongo_login_collection()` - Collection login sessions (90 days TTL)
- `_log_login_event()` - Log eventi login/logout
- `_get_mongo_last_analysis_collection()` - Collection analisi per-utente
- `_save_last_analysis()` - Salva ultima analisi utente
- `_load_last_analysis()` - Carica ultima analisi utente

### 5. FILE SYSTEM HELPERS (Righe 427-520)
**Funzioni:**
- `_is_writable_dir()` - Verifica directory scrivibile
- `get_upload_folder()` - Determina folder upload appropriata
- `UPLOAD_FOLDER` - Path globale per upload

### 6. CACHE GLOBALS (Righe 485-540)
Variabili globali per cache API:
- `_SP500_PRICE_CACHE` - Cache prezzo S&P 500
- `_ES_PRICE_CACHE` - Cache prezzo ES Futures
- `_NVDA_SNAPSHOT_CACHE` - Cache snapshot NVDA
- `_SPY_SNAPSHOT_CACHE` - Cache snapshot SPY
- `_MSFT_SNAPSHOT_CACHE` - Cache snapshot MSFT
- `_SPX_SNAPSHOT_CACHE` - Cache snapshot SPX
- `_XSP_SNAPSHOT_CACHE` - Cache snapshot XSP
- `_AAPL_SNAPSHOT_CACHE`, `_GOOG_SNAPSHOT_CACHE`, `_AMZN_SNAPSHOT_CACHE`

### 7. DATA PARSING UTILITIES (Righe 542-650)
**Funzioni:**
- `_parse_pdf_number()` - Parse numeri da PDF (gestisce virgole, negativi)
- `_parse_nasdaq_price()` - Parse prezzi da API NASDAQ
- `_fetch_stooq_latest_close()` - Fetch dati da Stooq
- `_fetch_nasdaq_json()` - Fetch JSON da API NASDAQ
- `_fetch_nasdaq_quote()` - Fetch quote NASDAQ
- `_parse_nasdaq_month_day()` - Parse date da stringhe NASDAQ

### 8. MARKET DATA FETCHERS (Righe 650-1450)
**Funzioni Cache:**
- `get_sp500_price_cached()` - Prezzo S&P 500 con fallback SPY
- `get_es_price_cached()` - Prezzo ES Futures da Stooq
- `_get_nasdaq_stock_snapshot_cached()` - Template generico snapshot NASDAQ
- `get_nvda_snapshot_cached()` - Snapshot NVDA (supporta mode price/flip)
- `get_spy_snapshot_cached()` - Snapshot SPY ETF
- `get_msft_snapshot_cached()` - Snapshot MSFT (supporta mode price/flip)
- `get_spx_snapshot_cached()` - Snapshot SPX con fallback SPY
- `get_xsp_snapshot_cached()` - Snapshot XSP con fallback SPY
- `get_aapl_snapshot_cached()` - Snapshot AAPL
- `get_goog_snapshot_cached()` - Snapshot GOOG
- `get_amzn_snapshot_cached()` - Snapshot AMZN

### 9. PDF EXTRACTION FUNCTIONS (Righe 1461-2350)
**Funzioni Principali:**
- `extract_0dte_data()` - Estrae dati 0DTE da PDF
- `extract_1dte_data()` - Estrae dati 1DTE da PDF
- `extract_nearest_positive_dte_data()` - Fallback scadenza più vicina
- `_extract_dte_days_data()` - Estrazione generica per DTE specifico
- `_find_dte_column_mapping()` - Mappa colonne DTE nel PDF
- `_extract_dte_pair_data_pymupdf()` - Estrazione coordinate-based con PyMuPDF
- `_find_available_dtes_pymupdf()` - Trova DTE disponibili via PyMuPDF
- `_extract_text_blocks_pymupdf()` - Estrae blocchi di testo da PDF

### 10. GAMMA ANALYSIS FUNCTIONS (Righe 2100-2370)
**Funzioni:**
- `analyze_0dte()` - Analisi completa gamma exposure
- `_find_gamma_flip()` - Identifica punto di flip gamma
- `_identify_key_levels()` - Identifica livelli chiave (supporti/resistenze)
- `_find_nearest_strike()` - Trova strike più vicino a prezzo
- `_calculate_percentages()` - Calcola distanze percentuali

### 11. WEB ROUTES - Authentication (Righe 2375-2525)
**Routes:**
- `GET /` - Redirect a dashboard/login
- `GET /login` - Pagina login Google OAuth
- `GET /login/google` - Inizia flow OAuth
- `GET /auth/callback` - Callback OAuth + log login
- `GET /logout` - Logout + log evento
- `GET /admin` - Redirect a /admin/login-sessions
- `GET /admin/login-sessions` - Dashboard admin sessioni

### 12. WEB ROUTES - API Endpoints (Righe 2524-2720)
**Market Data APIs:**
- `GET /api/sp500-price` - Prezzo S&P 500
- `GET /api/es-price` - Prezzo ES Futures
- `GET /api/health` - Health check (pubblico)
- `GET /api/last-analysis` - Ultima analisi utente
- `GET /api/nvda-snapshot` - Snapshot NVDA
- `GET /api/spy-snapshot` - Snapshot SPY
- `GET /api/msft-snapshot` - Snapshot MSFT
- `GET /api/spx-snapshot` - Snapshot SPX
- `GET /api/xsp-snapshot` - Snapshot XSP
- `GET /api/aapl-snapshot` - Snapshot AAPL
- `GET /api/goog-snapshot` - Snapshot GOOG
- `GET /api/amzn-snapshot` - Snapshot AMZN

**MongoDB APIs:**
- `GET /api/pressure-history` - Storico pressure points
- `POST /api/pressure-point` - Salva pressure point

### 13. WEB ROUTES - Main App (Righe 2722-2815)
**Routes:**
- `POST /analyze` - Endpoint principale per analisi PDF
  - Accetta upload PDF
  - Estrae dati 0DTE/1DTE/nearest
  - Esegue analisi gamma
  - Salva risultati per utente
  - Ritorna JSON con livelli chiave

### 14. APPLICATION ENTRY POINT (Righe 2810-2816)
```python
if __name__ == '__main__':
    app.run(debug=True, port=5005, host='0.0.0.0')
```

## 🔑 Pattern Architetturali Utilizzati

### 1. Lazy Initialization
OAuth e MongoDB collections usano lazy initialization per permettere modifiche .env senza restart completo.

### 2. Caching Pattern
Cache in-memory con TTL per ridurre chiamate API esterne (NASDAQ, Stooq).

### 3. Fallback Chain
Molte funzioni hanno fallback multipli (es. SPX → SPY, stooq → NASDAQ).

### 4. Try-Except Wrappers
Funzioni MongoDB/OAuth wrapped in try-except per non bloccare mai il flow principale.

### 5. Decorator Pattern
`@login_required` e `@app.before_request` per authentication layer.

## 🎯 Dipendenze Principali

- **Flask**: Web framework
- **Authlib**: Google OAuth 2.0
- **PyMongo**: MongoDB client
- **pdfplumber**: PDF table extraction (primary)
- **PyMuPDF (fitz)**: PDF coordinate-based extraction (fallback)
- **pandas**: Data manipulation
- **werkzeug**: ProxyFix middleware

## 📝 Note per Manutenzione

1. **Aggiungere nuovi simboli di mercato**: Creare cache dict + funzione `get_SYMBOL_snapshot_cached()`
2. **Modificare logica analisi**: Funzioni in sezione 10 (analyze_0dte, _find_gamma_flip)
3. **Aggiungere route API**: Sezione 12, seguire pattern esistenti con caching
4. **Modificare autenticazione**: Sezione 3, funzioni _is_authenticated/_is_admin
5. **Modificare estrazione PDF**: Sezione 9, preferire PyMuPDF per performance

## 🔒 Sicurezza

- Session cookies: HttpOnly, SameSite=Lax, Secure (su Vercel)
- Admin access: Richiede ADMIN_EMAILS allowlist su Vercel
- OAuth: Lazy registration previene esposizione credenziali
- MongoDB: Connection timeout 2.5s per fail-fast
- File uploads: Max 16MB, secure_filename sanitization

## 🚀 Deployment

- **Locale**: `python app.py` (porta 5005)
- **Vercel**: Entry point in `api/index.py`, environment vars via dashboard
- **MongoDB**: Atlas con TTL indexes per collections (pressure, login_sessions)
