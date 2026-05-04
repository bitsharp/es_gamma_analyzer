# Stock Picker — Piano di implementazione

Estrazione dal corso Serafini (audio1+audio2) + foglio `Foglio Calcolo Multipli.xlsx` + analisi codebase `es_gamma_analyzer`.

---

## 1. Cosa abbiamo estratto dal corso

### Formula master (Damodaran modificata)

```
P/E teorico = 13.1 + 1.2 × growth_media_3y × 100 − sconto_paese − sconto_settore (+ premio_lusso)
Target      = P/E teorico × EPS anno successivo
Discount %  = (Target − Prezzo attuale) / Prezzo attuale
Ratio       = Discount / (DevSt% / 100)   # discount normalizzato per volatilità
```

### Sconti geografici

| Paese | Sconto |
|---|---|
| USA | 0 |
| EU | −5 |
| Italia | −5 |
| Cina | −10 |
| Paesi emergenti | −10 |
| Giappone | ~ −3 |

### Sconti settoriali

| Settore | Sconto |
|---|---|
| Financial (banche/assicur.) | −5 |
| Energy | −5 |
| Healthcare | −5 |
| Real estate | −2 |
| Utilities | −2 |
| Communication services | −1.5 |
| Tech / Industrial | 0 |
| **Lusso** | **+5 (premio)** |

Sconti combinati italiani: banca italiana = −10 (paese + settore), energy italiana = −10.

### Threshold operativi

- **VIX contango** (VX1/VX2): > 1.5% → stay long; < 1.5% → ruota su low beta o cash
- **Mediana P/E forward S&P 500** ultimi 10y: range 14–23, media 19
- **Settore IT**: forward P/E mediano 30, growth mediana 19%
- **Premio rischio azionario**: medio 3.5% (range 2.5–4.5)
- **Capitalizzazione minima** screener USA: 2 mld; Italia: 500 mln
- **Minimo titoli in portafoglio**: 5, possibilmente decorrelati

### Beta classificazione settori

- **High beta** (aggressive): Tech, Communication, Discretionary, Financial, Industrial
- **Low beta** (defensive): Healthcare, Materials, Energy, Staples, Utilities

### Logica di scoring/ranking

1. Calcola P/E teo + target per ogni ticker
2. Ordina per `Discount %` (verde > 0, rosso < 0)
3. Aggiusta per rischio con `Ratio = Discount / DevSt%`
4. Filtri qualitativi:
   - Free Cash Flow deve crescere come o più degli EPS
   - Trimestrale recente: se ha mancato → evita ~15gg; se battuta → buy on retracement
5. Portafoglio: min 5 ticker scorrelati, applica filtro VIX, ruota high/low beta

### Formule Excel chiave

```
H = (D-C)/C                              # crescita YoY
L = AVERAGE(H:J)                         # crescita media triennio
N = 13.1 + 1.2*L*100 [- sconti]          # P/E teorico
P = N * D                                # Target = P/E × EPS anno+1
T = (P-R)/R                              # Discount vs prezzo attuale
V = T / (S/100)                          # Ratio discount/volatilità
```

---

## 2. Codebase esistente — cosa riusare

### Pattern già pronti

- **Cache per-symbol** con TTL: globali `_<NAME>_CACHE = {value, fetched_at, value_by_mode}` ([app.py:1316-1390](app.py))
- **Mongo lazy init** + silent failure: `_get_mongo_<name>_collection()` ([app.py:946, 1026, 6014](app.py))
- **Per-user scoping**: `_current_user_key()` ritorna `google:{sub}` o `email:{email}` ([app.py:933](app.py))
- **Auth gate**: `login_required` decorator + `_require_login` before_request
- **Number parsing**: `_parse_pdf_number()` gestisce formati US/EU ([app.py:1400](app.py))
- **Fallback chains** per market data: Nasdaq → Stooq → yfinance
- **Frontend pattern**: nav tabs in [templates/index.html:672](templates/index.html), card layout, fetch + JS rendering

### Cosa manca (da costruire)

- Fetch fondamentali (sector, country, beta, EPS estimates, FCF)
- Screener backend (filtri + sort)
- UI tabellare per stock picking
- Calcolo Damodaran (formule sopra)
- VIX contango signal

---

## 3. Implementazione proposta

### Approccio

Tutto in `app.py` come nuove sezioni con banner comments (rispetta convenzione monolitica). Niente Tikr API (paywalled). Sorgenti dati reali:

- **yfinance** (già presente come fallback) per: sector, country, market cap, beta, prezzo, dev.st annualizzata, EPS estimates (2 anni forward affidabili)
- **Input manuale EPS** per anni 3-5 (= flusso Excel attuale dell'utente), persistito su Mongo
- **^VIX + ^VIX3M** da yfinance come proxy contango (alternativa: scraping vixcentral)

Niente mock data: dove yfinance non copre, l'utente inserisce manualmente come fa già oggi.

### Nuove componenti `app.py`

#### Cache + collezioni Mongo

```python
_STOCK_FUNDAMENTALS_CACHE = {}   # TTL 6h
_VIX_CONTANGO_CACHE = {}         # TTL 5min
_SECTOR_PERF_CACHE = {}          # TTL 1h

_get_mongo_stock_picks_collection()        # per-user watchlist + EPS
_get_mongo_screener_runs_collection()      # cache risultati (TTL 1h)
```

Schema documento `stock_picks`:

```python
{
    "user_key": "...",
    "ticker": "NVDA",
    "eps": [4.77, 8.29, 11.12, 13.28, 15.12],   # 2026..2030
    "fcf": [...],                                  # opzionale
    "country": "US",                               # US|EU|IT|CN|EM|JP
    "sector": "Tech",                              # Tech|Energy|Banks|...
    "actual_price_override": None,                 # null = usa live
    "dev_st_override": None,
    "notes": "",
    "last_updated": ...,
}
```

#### Costanti modello

```python
COUNTRY_DISCOUNTS = {
    "US": 0, "EU": -5, "IT": -5,
    "CN": -10, "EM": -10, "JP": -3,
}

SECTOR_DISCOUNTS = {
    "Tech": 0, "Industrial": 0,
    "Financial": -5, "Energy": -5, "Healthcare": -5,
    "RealEstate": -2, "Utilities": -2,
    "Comms": -1.5,
    "Lusso": +5,  # premio
}

SECTOR_BETA = {
    "Tech": "high", "Comms": "high", "Discretionary": "high",
    "Financial": "high", "Industrial": "high",
    "Healthcare": "low", "Materials": "low", "Energy": "low",
    "Staples": "low", "Utilities": "low",
}

VIX_CONTANGO_THRESHOLD = 1.5  # %
```

#### Calcolatore (pura funzione)

```python
def calculate_damodaran_target(eps_list, country, sector, current_price, dev_st_pct):
    """
    eps_list: [eps_y0, eps_y1, eps_y2, eps_y3, eps_y4]
              dove y0 = anno corrente (es. 2026)
    Ritorna dict con avg_growth, pe_theoretical, target_y1, discount, ratio
    """
    growth_yoy = [(eps_list[i+1]-eps_list[i])/eps_list[i] for i in range(4)]
    avg_growth_3y = sum(growth_yoy[:3]) / 3   # 2027-2029

    country_disc = COUNTRY_DISCOUNTS.get(country, 0)
    sector_disc  = SECTOR_DISCOUNTS.get(sector, 0)

    pe_theo = 13.1 + 1.2 * avg_growth_3y * 100 + country_disc + sector_disc
    target  = pe_theo * eps_list[1]
    discount = (target - current_price) / current_price
    ratio = discount / (dev_st_pct/100) if dev_st_pct else None

    return {
        "avg_growth_3y": avg_growth_3y,
        "pe_theoretical": pe_theo,
        "target_y1": target,
        "discount_pct": discount,
        "ratio_discount_vola": ratio,
        "verdict": "UNDERVALUED" if discount > 0 else "OVERVALUED",
    }
```

#### Fetcher fondamentali

```python
def get_fundamentals_cached(ticker, max_age_seconds=21600):
    # yfinance.Ticker(ticker).info → sector, country, marketCap, beta, currentPrice
    # yfinance.Ticker(ticker).earnings_estimates → 2 anni EPS forward
    # yfinance.Ticker(ticker).history(period="1y") → annualized stdev
    # Fallback: _fetch_stooq_latest_close per prezzo se yfinance KO
    ...
```

#### VIX contango signal

```python
def get_vix_contango_cached(max_age_seconds=300):
    # yf.Ticker("^VIX").history(period="2d") → ultimo close
    # yf.Ticker("^VIX3M").history(period="2d") → ultimo close
    # contango_pct = (vix3m - vix) / vix * 100
    # signal = "LONG" if contango > 1.5 else "DEFENSIVE"
    return {"vix": ..., "vix3m": ..., "contango_pct": ..., "signal": ...}
```

#### Endpoint REST

```
GET    /api/picker/fundamentals/<ticker>     # auto-fill: sector, country, beta, prezzo, dev.st
POST   /api/picker/analyze                    # body: {ticker, eps[], country, sector} → calcolo
GET    /api/picker/watchlist                  # tutti i picks utente, calcolo aggiornato
POST   /api/picker/watchlist                  # upsert pick
DELETE /api/picker/watchlist/<ticker>
POST   /api/picker/screener                   # filtri: country/sector/min_growth/max_pe/min_discount
GET    /api/picker/vix-signal                 # contango + raccomandazione
GET    /api/picker/sector-rotation            # 11 SPDR + perf 1m/3m/YTD + classificazione beta
```

Tutti gated da `login_required`, scoped via `_current_user_key()`, wrapped in try/except sui Mongo call.

### Frontend

**`templates/picker.html`** (nuovo) o estensione di [templates/stocks.html](templates/stocks.html):

- **Banner VIX** in alto: contango% + colore (verde >1.5%, rosso <1.5%) + raccomandazione testuale
- **Tab "Watchlist"**: tabella sortable replica del foglio Excel — colonne: Ticker, Sector, Country, EPS 26-30, Growth medio, P/E teo, Target dic26/27, Prezzo, Discount%, Ratio. Verde/rosso su Discount.
- **Tab "Aggiungi ticker"**: form, auto-fill via `/api/picker/fundamentals/<ticker>` (sector, country, beta, prezzo, dev.st pre-compilati), 5 input EPS manuali. Submit → analizza e salva.
- **Tab "Screener"**: filtri country/sector/growth/discount → query sulla watchlist (o su un universo più largo se aggiungiamo lista Tikr-like).
- **Tab "Sector rotation"**: 11 settori, perf 1m/3m/YTD, colorati per classificazione beta.

Aggiungere voce nav in [templates/index.html:672](templates/index.html) accanto al tab Stocks.

---

## 4. Limiti e tradeoff

### Cosa NON faremo (motivazione)

- **Pattern volumi sforzo/risultato** (video 2): richiede OHLCV intraday + riconoscimento candele anomale. TradingView lo fa nativamente meglio.
- **Volume profile**: stesso motivo.
- **Tracking trimestrali automatico**: yfinance dà earnings dates ma non i contenuti della call. Manuale in fase 1.
- **Tikr API**: nessuna API pubblica disponibile.

### Limiti yfinance

- EPS estimates: solo 2 anni forward affidabili (curr year + next year)
- Sector classification: GICS standard, mappare manualmente verso le nostre 8 categorie
- Country: a volte mancante per ADR esotici
- Beta: storico 5y mensile, non sempre aggiornato

### Workaround input manuale

L'utente già copia EPS da Tikr in Excel. La web app:
1. Pre-compila quello che yfinance fornisce
2. Lascia 3 input manuali (anni +2, +3, +4)
3. Salva tutto su Mongo per-utente → la prossima volta è già lì

---

## 5. Phasing

### Fase 1 — MVP (~1 giornata)

- Calcolatore Damodaran (pura funzione)
- Watchlist CRUD (Mongo + 4 endpoint)
- Auto-fill fondamentali via yfinance
- UI tabella `picker.html` con form + watchlist sortable
- Tab nav in `index.html`

Sostituisce completamente l'Excel.

### Fase 2 — Screener + VIX (~mezza giornata)

- Endpoint screener con filtri
- VIX contango signal + banner UI
- Sector rotation panel con 11 SPDR

### Fase 3 — Opzionale

- Alert (email/telegram) quando ticker watchlist scende sotto P/E teorico
- Integrazione earnings calendar
- Storia variazioni target nel tempo (chart)
- Pattern volumi semplificato (gap up/down su trimestrale)

---

## 6. Domande aperte

1. **EPS auto vs manuale**: solo manuale (replica Excel, più affidabile) o auto-fill yfinance + completamento manuale (più comodo)?
2. **VIX contango**: proxy yfinance ^VIX/^VIX3M (zero setup, meno preciso) o scraping vixcentral.com (più fedele al video)?
3. **Collezione Mongo**: condivisa con il modulo Stocks esistente in [app.py:6014](app.py) (`_get_mongo_stocks_cache_collection`) o nuova collezione separata?
4. **Universo screener**: solo watchlist personale dell'utente, o lista pre-caricata di ~500 ticker (S&P 500 + FTSE MIB + DAX) su cui far girare il calcolo?

---

## 7. Riferimenti

- Foglio sorgente: `/Users/lucataurisano/Downloads/Foglio Calcolo Multipli.xlsx` (Foglio1, righe 6-77)
- Trascrizioni corso: audio1.txt (lungo periodo + price earning), audio2.txt (volumi + portafoglio + esempi pratici)
- Codebase: [app.py](app.py), [templates/index.html](templates/index.html), [templates/stocks.html](templates/stocks.html)
- Modello Damodaran originale: intercetta 13 (qui 13.1 modificata)
- Sconti settoriali e geografici: derivati dal video (Damodaran originale aveva solo geografici)
