# Changelog

Tutte le modifiche sostanziali a questa applicazione, in ordine cronologico inverso.
Formato ispirato a [Keep a Changelog](https://keepachangelog.com/it/1.1.0/) e versionamento [SemVer](https://semver.org/lang/it/).

La versione mostrata nell'header dell'app è letta direttamente da questo file: la prima riga `## [X.Y.Z]` è la versione corrente.

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
