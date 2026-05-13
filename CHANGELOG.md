# Changelog

Tutte le modifiche sostanziali a questa applicazione, in ordine cronologico inverso.
Formato ispirato a [Keep a Changelog](https://keepachangelog.com/it/1.1.0/) e versionamento [SemVer](https://semver.org/lang/it/).

La versione mostrata nell'header dell'app è letta direttamente da questo file: la prima riga `## [X.Y.Z]` è la versione corrente.

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
