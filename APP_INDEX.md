# 📊 app.py - Indice delle Sezioni

## Struttura Organizzata del Codice

```
┌────────────────────────────────────────────────────────────────────┐
│  FILE: app.py (2,870 righe)                                        │
│  SEZIONI: 14                                                        │
│  STATO: ✓ Organizzato e Documentato                               │
└────────────────────────────────────────────────────────────────────┘
```

### 📋 Indice Completo

| Riga  | Sezione                                           | Descrizione                                    |
|-------|---------------------------------------------------|------------------------------------------------|
| **5** | **IMPORTS**                                       | Import librerie e dipendenze                   |
| **50** | **CONFIGURATION & GLOBALS**                      | Setup Flask, env vars, costanti globali        |
| **81** | **AUTHENTICATION & SESSION MANAGEMENT**          | OAuth Google, login, permessi admin            |
| **199** | **MONGODB HELPERS**                              | Collections, logging, persistence              |
| **443** | **FILE SYSTEM HELPERS**                          | Upload folder, directory writable checks       |
| **481** | **CACHE GLOBALS (Market Data)**                  | Cache dizionari per prezzi e snapshots         |
| **544** | **DATA PARSING & EXTRACTION UTILITIES**          | Parse numeri PDF, prezzi, date                 |
| **830** | **MARKET DATA FETCHERS**                         | NASDAQ API, option chains, stock quotes        |
| **1492** | **PDF EXTRACTION FUNCTIONS**                    | Estrazione 0DTE, 1DTE, multi-DTE da PDF        |
| **2194** | **GAMMA ANALYSIS CORE FUNCTIONS**               | Analisi gamma, flip point, key levels          |
| **2413** | **WEB ROUTES - Authentication & Admin**         | Login, logout, admin dashboard                 |
| **2566** | **WEB ROUTES - API Endpoints**                  | Market data APIs, MongoDB APIs                 |
| **2768** | **WEB ROUTES - Main Application**               | Endpoint /analyze (upload PDF)                 |
| **2859** | **APPLICATION ENTRY POINT**                     | `if __name__ == '__main__'`                    |

## 🎯 Come Navigare

### In VS Code
1. Apri `app.py`
2. Usa `Cmd+Shift+O` (Mac) o `Ctrl+Shift+O` (Windows/Linux) per vedere l'outline
3. Cerca il nome della funzione o sezione

### Trova una Sezione Specifica
```python
# Cerca nel file:
# ============================================================================
# NOME SEZIONE
# ============================================================================
```

### Vai a una Riga Specifica
- `Cmd+G` (Mac) o `Ctrl+G` (Windows/Linux)
- Inserisci il numero di riga dalla tabella sopra

## 📚 Documentazione Completa

Per maggiori dettagli, consulta:

- **[STRUCTURE.md](STRUCTURE.md)** - Documentazione tecnica completa
  - Lista funzioni per sezione
  - Pattern architetturali
  - Note per manutenzione
  - Guida sicurezza e deployment

- **[CODE_ORGANIZATION.md](CODE_ORGANIZATION.md)** - Riepilogo modifiche
  - Cosa è stato fatto
  - Vantaggi dell'organizzazione
  - Come usare la nuova struttura

## 🔧 Funzioni Principali per Sezione

### Authentication (riga 81)
- `_ensure_google_oauth_registered()` - Setup OAuth lazy
- `_is_authenticated()` - Check login
- `_is_admin()` - Check permessi admin
- `login_required()` - Decorator auth

### MongoDB (riga 199)
- `_get_mongo_collection()` - Pressure points
- `_get_mongo_login_collection()` - Login sessions
- `_log_login_event()` - Log eventi auth
- `_save_last_analysis()` - Salva analisi utente

### Market Data (riga 830)
- `get_nvda_snapshot_cached()` - NVDA option chain
- `get_spy_snapshot_cached()` - SPY option chain
- `get_sp500_price_cached()` - S&P 500 price
- `get_es_price_cached()` - ES Futures price

### PDF Extraction (riga 1492)
- `extract_0dte_data()` - Estrai dati 0DTE
- `extract_1dte_data()` - Estrai dati 1DTE
- `_extract_dte_pair_data_pymupdf()` - PyMuPDF extraction

### Gamma Analysis (riga 2194)
- `analyze_0dte()` - Analisi completa
- `_find_gamma_flip()` - Trova gamma flip
- `_identify_key_levels()` - Identifica supporti/resistenze

### Web Routes (riga 2413)
- `GET /` - Dashboard
- `GET /login` - Login page
- `GET /auth/callback` - OAuth callback
- `POST /analyze` - Analizza PDF
- `GET /api/*` - Vari endpoint API

## 💡 Quick Tips

### Aggiungere una Nuova Funzione
1. Identifica la sezione appropriata dall'indice
2. Aggiungi la funzione nella sezione corretta
3. Mantieni le funzioni correlate vicine

### Modificare Logica Esistente
1. Trova la sezione dalla tabella sopra
2. Usa `Cmd+G` per andare alla riga
3. Modifica mantenendo la struttura

### Debug di un Problema
1. Identifica quale componente è coinvolto
2. Consulta STRUCTURE.md per dettagli
3. Usa i commenti di sezione per navigare velocemente

---

**Ultimo Aggiornamento**: 7 Gennaio 2026
**Dimensione File**: 2,870 righe
**Numero Sezioni**: 14
