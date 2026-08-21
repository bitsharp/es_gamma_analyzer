# Changelog

Tutte le modifiche sostanziali a questa applicazione, in ordine cronologico inverso.
Formato ispirato a [Keep a Changelog](https://keepachangelog.com/it/1.1.0/) e versionamento [SemVer](https://semver.org/lang/it/).

La versione mostrata nell'header dell'app è letta direttamente da questo file: la prima riga `## [X.Y.Z]` è la versione corrente.

## [1.13.0] — 2026-08-21

### Corretto
- **"Forza ricalcolo" ignorava il mercato selezionato.** Il pulsante chiamava `POST /api/screener/refresh` senza `market=`, e l'endpoint in quel caso ricade sul default `US`: aprendo la scheda Italia o Germania e forzando il ricalcolo si rigeneravano i titoli americani, mentre quelli europei restavano in cache con i vecchi dati Yahoo (badge YF) fino alla scadenza naturale delle 12h. Ora passa il mercato attivo e `force=1`, così un `in_progress` rimasto appeso da un run interrotto non blocca più il ricalcolo. Su Vercel, dove il refresh è sincrono, la lista si ricarica appena finito e mostra quanti titoli sono stati ricalcolati invece del generico "Refresh avviato…".

### Aggiunto
- **`GET /api/screener/fmp-status` (admin)**: diagnostica FMP. Dice se `FMP_API_KEY` è configurata nell'ambiente (con le ultime 4 cifre, per capire *quale* chiave è deployata senza esporla) e interroga live i tre endpoint che alimentano lo screener — `profile`, `analyst-estimates`, `historical-price-eod/light` — su un ticker a scelta, default `ENI.MI`. Riporta HTTP status e messaggio d'errore vero di FMP. Serve a distinguere le tre cause di un badge YF, indistinguibili dal risultato finale: cache vecchia, chiave mancante o di un altro account, mercato non coperto dal piano.

### Tecnico
- `SCREENER_DISABLE_YF_FALLBACK=1` disattiva il fallback yfinance: i ticker su cui FMP non risponde spariscono dalla lista invece di comparire con dati Yahoo. Utile per misurare la copertura reale del piano FMP; da lasciare spento in esercizio, perché un disservizio FMP svuoterebbe lo screener.
- `_fmp_probe()` affianca `_fmp_get()` per la sola diagnostica: restituisce l'esito (status, righe, errore) invece di ingoiarlo. Il percorso normale resta silenzioso.

## [1.12.0] — 2026-08-21

### Aggiunto
- **Fondamentali FMP anche per i mercati europei e indiani.** Con il passaggio al piano **FMP Ultimate** (Global Coverage) le analyst-estimates, il profilo e lo storico prezzi rispondono anche su Borsa Italiana (`.MI`), XETRA (`.DE`) e NSE (`.NS`): lo screener IT/DE/IN non ricade più sistematicamente su Yahoo Finance, che resta come fallback per i singoli titoli senza consenso EPS.

### Modificato
- **Sconto-paese Damodaran corretto per i titoli non-US cercati a mano.** FMP espone il paese come codice ISO-2 (`IT`), yfinance come nome esteso (`Italy`): `_map_country_to_bucket()` ora accetta entrambe le grafie. Senza questa modifica, con FMP che inizia a rispondere sui ticker europei, un titolo di Piazza Affari sarebbe finito nel bucket `US` (sconto 0 invece di −5) nella ricerca singolo titolo e nel portafoglio.
- **P/E forward storico costruito sull'EPS diluito.** L'API `stable` espone `epsDiluted` in camelCase, mentre il codice leggeva `epsdiluted` (grafia della v3) e ricadeva quindi sempre sull'EPS basic. Cache invalidata (`schema_version` 2 → 3): le serie vengono ricalcolate alla prima apertura.
- **Tooltip insider più preciso**: per i titoli non-US il dato manca perché i Form 4 sono un deposito SEC che le società europee non compilano — non è una limitazione del piano FMP, e non si sblocca con l'upgrade.

### Tecnico
- `historical-price-eod/light` per il calcolo della volatilità ora passa un `from` esplicito (550 giorni). Su Ultimate lo storico arriva a 30+ anni e senza filtro si scaricavano migliaia di righe per usarne 260, a spese della banda (cap 150 GB/30gg) e del budget di 60s di Vercel.

## [1.11.0] — 2026-08-20

### Aggiunto
- **Vista compatta / completa nella lista dello screener**: nuovo toggle "Vista" accanto al filtro per zona. In **Compatta** (default) la scheda mostra intestazione, pill, badge Zona, griglia metriche e spiegazione della zona; in **Completa** si aggiungono il grafico del **P/E forward storico 5y** e le **transazioni insider 3m**. La scelta vale per la lista Damodaran e per la drill-down di settore, viene ricordata tra le sessioni (localStorage) e il cambio vista ri-renderizza dai dati già in memoria, senza nuove chiamate API. La scheda del singolo titolo cercato resta sempre completa.



### Aggiunto
- **Box COT Petrolio (CL)** nella pagina Macro: sesto contratto, accanto a S&P 500, NASDAQ-100, Euro FX, Oro e Bitcoin. Fonte tradingster, contratto `067651` — WTI-PHYSICAL NYMEX, cioè il CL (nel report Legacy la CFTC ha rinominato "CRUDE OIL, LIGHT SWEET"); il `067411` è l'ICE Europe ed è un altro contratto.


## [1.9.1] — 2026-08-12

### Modificato
- **Card COT più compatte nella pagina Macro**: fino a **6 per riga** su schermi larghi (≥1200px), poi 4 / 3 / 2 / 1 scendendo di breakpoint. Font, padding e bordi ridotti in proporzione; nell'header il ticker e la data del report vanno su una seconda riga (prima il titolo lungo veniva troncato), e il footer usa la forma breve "CFTC Legacy" invece del nome completo del report, che da solo occupava tre righe.


## [1.9.0] — 2026-08-12

### Aggiunto
- **Box COT Oro (GC) e Bitcoin (BTC)** nella pagina Macro, accanto a S&P 500, NASDAQ-100 ed Euro FX: stessi contenuti (bias narrativo, NC Long/Short/Net con variazioni WoW, % OI, traders, storico settimanale del net).

### Tecnico
- Il servizio COT esterno espone solo sp500/nasdaq100/eurofx: per oro e bitcoin il backend legge direttamente **tradingster.com** (`/cot/legacy-futures/088691` e `/133741`) e traduce la pagina nello stesso payload JSON, così il render delle card resta invariato. Lo storico settimanale (12 settimane) è estratto dalle serie dei grafici incorporate nella pagina, con le variazioni ricalcolate settimana su settimana.
- Nuovo registro `_COT_CONTRACTS` (codice contratto, ticker, label, upstream `api` o `tradingster`): `GET /api/cot/<symbol>` valida i simboli da lì e `get_cot_cached` smista la fetch. Cache 1h e stale-while-error identici per entrambe le sorgenti.


## [1.8.0] — 2026-07-30

### Aggiunto
- **Box COT NASDAQ-100 (NQ) e Euro FX (6E)** nella pagina Macro, accanto al COT S&P 500: stessi contenuti (bias narrativo, NC Long/Short/Net con variazioni WoW, % OI, traders, storico settimanale del net) per i tre contratti CFTC.

### Tecnico
- Nuovo endpoint generico `GET /api/cot/<symbol>` (sp500 / nasdaq100 / eurofx) che proxya il servizio COT esterno via backend — necessario perché l'upstream è HTTP in chiaro e dal sito in HTTPS il browser lo bloccherebbe come mixed content. Cache in-memory per simbolo (TTL 1h, stale-while-error), `?force=1` per il bypass. `/api/cot-sp500` resta come alias. Base URL configurabile via `COT_API_BASE_URL`.
- `macro.html` rifattorizzato: le card COT sono generate da una config (`COT_CONTRACTS`) con markup, fetch e render generici sul prefix — aggiungere un contratto = una riga di config.


## [1.7.0] — 2026-07-30

### Aggiunto
- **Nuova pagina Macro** (`/macro`), raggiungibile dalla voce "Macro" nella navbar di tutte le pagine. Ospita i dati macro/di posizionamento, a partire dal box **COT S&P 500** (Non-Commercial, CFTC Legacy) con bias narrativo, variazioni WoW e storico settimanale del net.

### Modificato
- Il box COT S&P 500 è stato **spostato dalla dashboard Gamma alla pagina Macro**. La card Market Pressure si allarga per occupare lo spazio liberato nella riga dei key levels.

### Tecnico
- Nuova route `/macro` (`macro_page`, protetta da `login_required`) e template `templates/macro.html`; il polling COT (30 min, con pausa quando la tab è nascosta) e il refresh manuale vivono ora solo lì. L'endpoint `/api/cot-sp500` è invariato.


## [1.6.0] — 2026-07-15

### Aggiunto
- **Volatilità VIX** (domanda 3 del processo "Argo"). Nuova card "Volatilità (VIX)" che mostra il livello VIX con **bande** (Calmo <15 · Normale · Elevato · Alto · Estremo, soglie 20/25/30 del video), la **struttura a termine** VIX/VIX3M (Contango = mercato calmo · Backwardation = stress) e i tre punti della curva (9D / 30D / 3M). Una nota lega la volatilità al regime gamma atteso (es. "Backwardation → favorisce gamma negativo / cascata").

### Tecnico
- `get_vix_snapshot_cached()` con banda, term structure e nota; fonte **indici CBOE delayed** (`quotes/_VIX`, `_VIX3M`, `_VIX9D`) con fallback yfinance per il livello VIX. Nuova route `GET /api/vix-regime`. Scelta CBOE perché l'endpoint Yahoo `v7/quote` ora richiede auth/crumb (401) e non è affidabile.


## [1.5.0] — 2026-07-13

### Aggiunto
- **Net GEX SPX live da CBOE, senza PDF** (domanda 1 del processo "Argo"). Nuova card "Net GEX — SPX live" alimentata dal feed CBOE *delayed quotes*, che fornisce open interest **e gamma reali per strike**: niente più upload del PDF e niente stima Black-Scholes. Mostra il **Net GEX in $B per punto** con **badge di regime a bande** (Bivio / debole / moderato / estremo, colorato per segno), gamma flip, Put/Call Wall, distanza dal flip in ATR e un **profilo Gamma per-strike** (barre verdi long-gamma / rosse short-gamma).
- **Doppia vista 0DTE ↔ Aggregato**: toggle che passa tra la sola scadenza 0DTE (intraday) e l'aggregato su tutte le scadenze, senza rifetch.

### Tecnico
- `get_spx_gamma_cboe_cached()`: scarica `cdn.cboe.com/.../_SPX.json`, filtra/aggrega per scope (0DTE = scadenza più vicina, All = tutte), cache TTL 8 min, fallback silenzioso. `_compute_gex_profile()` calcola Net GEX (scala **per-punto** e soglie del video: 0.5/1/3), flip proxy (zero-crossing del gamma netto per-strike) e profilo. Nuova route `GET /api/spx-gamma?scope=0dte|all|both`.
- ATR SPX via `^GSPC` riusa `_compute_atr_cached()`. Il gamma flip preferisce il proxy da GEX, con fallback all'euristica OI di `analyze_0dte` (invariata).


## [1.4.0] — 2026-07-12

### Aggiunto
- **Distanza dal Gamma Flip in ATR** (domanda 2 del processo "Argo"). Sotto la "Linea del Meteo" della card ES Key Levels compare un badge che misura quanto il prezzo è lontano dal gamma flip **in unità di ATR** invece che in punti, con etichetta e colore: `sul flip` (rosso, entro 0.3 ATR), `vicino sopra/sotto` (ambra, entro 1 ATR), `lontano sopra/sotto` (verde). La distanza si aggiorna live sul prezzo ES corrente; il tooltip del flip riporta lo stesso dato.
- La banda **±0.3 ATR** dà finalmente larghezza allo stato di regime "At Gamma Flip", che con il solo confronto d'uguaglianza prezzo/flip non scattava mai.

### Tecnico
- Nuovo helper `_compute_atr_cached()`: Wilder ATR(14) su OHLC giornaliero via yfinance (`ES=F`), cache per-simbolo con TTL 30 min e fallback silenzioso a `None` se yfinance non è disponibile.
- `analyze_0dte()` accetta un parametro opzionale `atr` e restituisce `atr`, `flip_distance_points`, `flip_distance_atr`, `flip_distance_label`. `_analyze_es_levels()` calcola l'ATR ES e lo inoltra. Le altre chiamate (SPX, NVDA, …) restano invariate (default `atr=None`).


## [1.3.0] — 2026-05-18

### Modificato
- **Rebrand → Polaris.** Il nome user-visible dell'app cambia da "ES Gamma Analyzer" a **Polaris**: navbar, title delle pagine, hero della pagina di login, alt-text del logo, User-Agent verso SEC/EDGAR e FMP. Il pitch del login è stato esteso per riflettere lo scope reale (non solo gamma 0DTE, ma anche screener Damodaran, journal, checklist, portafoglio).
- **Icona ridisegnata**: il glifo centrale passa da γ (gamma) a **P** (Polaris), su tile dark con la stessa linea del gamma flip e le candele. Aggiunto un piccolo *sparkle* dorato in alto a destra del logo come riferimento alla "stella polare". Favicon coerente con la nuova identità.

### Tecnico
- Slug interno (`es_gamma_analyzer`) mantenuto per repo, cartella, default `MONGODB_DB` e progetto Vercel: rinominarli richiederebbe una migrazione coordinata delle collection Mongo esistenti.
- 33 occorrenze utente-visibili sostituite in 11 file (6 template, app.py, 2 SVG, README, CLAUDE.md).


## [1.2.0] — 2026-05-17

### Aggiunto
- **Vista "Settori" nello Screener**: tab interna che affianca la Strategia Damodaran. Mostra una griglia di 12 settori (Tech, Comms, Discretionary, Staples, Financial, Healthcare, Industrial, Energy, Materials, Real Estate, Utilities, Lusso) con icona, sconto P/E del settore e numero di aziende qualificate vs valutate per il mercato corrente.
- **Drill-down per settore**: cliccando un settore si entra in una vista che mostra le **top 5 aziende qualificate** (stesso ranking della Damodaran: Zona Affare → Sconto → Equa → Cara, poi Discount %). Le market tab restano attive per cambiare paese senza perdere la sezione.
- **Ricerca scoped al settore**: nel drill-down la casella di ricerca cerca un ticker e applica la valutazione Damodaran. Se il ticker appartiene a un altro settore, viene mostrato un banner di mismatch e usato il `sector_disc` reale per non distorcere i calcoli.

### Tecnico
- 3 nuovi endpoint: `GET /api/screener/sectors`, `GET /api/screener/sectors/<bucket>`, `GET /api/screener/sectors/<bucket>/lookup/<ticker>`. Riusano il cache screener esistente (FMP primario, yfinance fallback) — nessun dato mock.
- Nuova mappa `_SCREENER_SECTOR_LABELS` con label IT, icona Bootstrap Icons e colore accent per ciascun bucket.


## [1.1.0] — 2026-05-14

### Aggiunto
- **Logo / brand mark** dell'applicazione: nuova icona SVG (γ stilizzata su tile dark con candle e gamma flip), usata come favicon su tutte le pagine e come logo accanto al titolo nella navbar della dashboard.
- **Apple touch icon** per quando l'app viene aggiunta alla home dello smartphone.

### Modificato
- **Pagina di login completamente ridisegnata**: layout hero su due colonne con pitch del prodotto (PDF parser, gamma flip, journal, checklist), card di login con bordo gradient e glow, bottone Google con logo nativo a 4 colori, background con radial glow + grid sottile. Su mobile collassa a colonna singola.
- `/favicon.ico` ora redireziona al nuovo SVG invece di restituire 204.

### Tecnico
- Nuova cartella [static/](static/) con [logo.svg](static/logo.svg) (512×512) e [favicon.svg](static/favicon.svg) (64×64), serviti via `url_for('static', ...)`.


## [1.0.3] — 2026-05-13

### Aggiunto
- Card **"Stato Account"** dedicata per i broker AMP/Rithmic — mostra Net Liq, Cash, Realized/Open/Total P/L, currency e nome broker direttamente dai dati importati. Per i conti Apex resta invece l'Apex Trail con tier e trailing drawdown.
- Il titolo della sezione monitor cambia automaticamente:
  - solo Apex → **"APEX TRAIL"** (giallo)
  - solo AMP/Rithmic → **"STATO ACCOUNT"** (turchese)
  - mix di broker → **"MONITOR ACCOUNT"** (grigio neutro)

### Modificato
- L'import del CSV account ora cattura anche `Broker`, `Currency`, `Open P/L` e `Total P/L` (prima venivano ignorati). Il broker determina quale card render — niente più Apex trail su conti che non sono Apex.
- La sezione monitor compare anche per account-only days (prima richiedeva almeno una trade card).


## [1.0.2] — 2026-05-13

### Modificato
- **Bottone "Importa CSV Account"** sempre visibile, indipendentemente dalla presenza di trade. Prima era nascosto dentro la sezione P&L che appariva solo dopo aver importato dei trade — chi voleva caricare solo il bilancio account a fine giornata non lo trovava.
- Il **Realized P/L** del CSV broker (campo `Realized P/L` di AMP/Rithmic/Overcharts) ora viene mostrato accanto al P&L calcolato dai trade — utile per riconciliare commissioni e slippage.
- La sezione P&L per Account ora compare anche per giornate senza trade ma con bilancio importato (es. giornate flat con solo aggiornamento balance).

### Tecnico
- Tooltip dell'import button chiarisce il formato accettato: CSV Overcharts / AMP-Rithmic con colonne Account, Cash, Net Liq, Realized P/L.


## [1.0.1] — 2026-05-13

Piccolo fix di scopribilità: il bottone d'import della checklist accetta anche export di altri broker.

### Modificato
- Bottone **"Importa CSV Apex"** rinominato in **"Importa CSV Overcharts"**. L'export Overcharts ha lo stesso layout indipendentemente dal broker collegato (Apex, AMP/Rithmic, ecc.), quindi il bottone già funzionava per tutti i broker — ora il nome non è più fuorviante.
- Tooltip aggiornato per chiarire la compatibilità.


## [1.0.0] — 2026-05-13

Prima versione tracciata. Lo screener basato sul modello Damodaran arriva al primo *milestone* completo: oltre al P/E teorico assoluto, ora compare anche il contesto storico del titolo, e diventano chiari **quando** e **a quale prezzo** ricomprarlo.

### Aggiunto
- **Sparkline P/E forward storico 5y** nelle card di screener, portfolio e lookup. Bande Q1 (verde, *zona di ricarico*), Q3 (grigia, *zona cara*) + mediana tratteggiata. Pallino bianco = punto attuale.
- **Badge "Zona ricarico storica"** quando il P/E forward attuale è ≤ Q1 della distribuzione 5y — la regola esplicita di Serafini.
- **Lista "Ultime zone Q1"**: data + **prezzo** + P/E delle ultime entrate storiche nel quartile basso. Concrete: vedi a che prezzo era effettivamente comprabile.
- **Picchi storici** sotto il grafico: max P/E (▲ rosso) e min P/E (▼ verde) con date e prezzi.
- **Toggle ⊞/⊟** per espandere il grafico a 300px con date sull'asse X e triangoli che marcano le entrate storiche in zona Q1 + il picco P/E.
- **Banner 13F nella pagina Stocks**: data dei dati correnti (regola SEC 45 giorni) e prossimo rilascio. Frasing dinamico ("domani", "tra 2 giorni", "in ritardo di N").
- **Badge versione** nella navbar di ogni pagina, cliccabile per aprire queste note.
- **`SERAFINI_RULES.md`** — documento di riferimento per il modello (P/E teorico, zone, metodologia hindsight-NTM, caveat).

### Modificato
- **Y-axis del grafico P/E** ora clippato al 5°–95° percentile invece di Q3×1.6, così la linea non si schiaccia in alto per i titoli con outlier post-IPO.
- **Altezza grafico** da 56px → 110px di default — bande e linea sono ora leggibili.
- **Label Q1/median/Q3** spostate fuori dall'SVG come overlay HTML in un gutter destro per non sovrapporsi alla linea.
- **CSS e JS del grafico** estratti in `templates/_pe_history.html` (Jinja partial) — eliminate ~380 righe di duplicazione tra screener e portfolio.

### Tecnico
- Metodologia "hindsight NTM" per ricostruire il forward P/E storico senza dover pagare il piano Premium di FMP per le stime consensus storiche.
- Cache Mongo `pe_history_cache` con TTL 7 giorni, `schema_version=2` per migrazione soft del campo `price` nei punti della serie.
- Helper `_compute_13f_period_info()` applica la regola SEC 13f-1 (45 giorni dopo fine trimestre) per determinare quale trimestre è attualmente reportabile e la prossima deadline.
