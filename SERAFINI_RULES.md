# Regole di Stefano Serafini — sintesi operativa

Documento di riferimento per la selezione titoli implementata in questa app.
Le regole arrivano dal corso Stock Picking Lab di Stefano Serafini (MVP =
Macro · Volumi · Probabilità). Qui resta solo la parte attualmente usata
dallo screener / portfolio; sezioni "trading discrezionale" (volumi
verticali, gamma, opzioni) sono fuori scope per ora.

---

## 1. Cos'è il multiplo "giusto"

Il prezzo di un'azione è il valore attualizzato degli utili futuri. Il
**P/E forward** (price / EPS atteso anno prossimo) è il rapporto sintetico
che il mercato paga per quella crescita futura. Da Damodaran:

```
P/E_teorico = 13.1 + 1.2 × growth + sconto_paese + sconto_settore
```

- `growth` = CAGR atteso degli EPS sul triennio successivo (espresso in %).
- `sconto_paese`: US = 0, EU = −5, IT = −5, JP = −3, CN = −10, EM = −10.
- `sconto_settore`: Tech / Comms = 0, Industrials = −2, Financials = −5,
  Energy = −5, Utilities = −3, Healthcare = −2, Staples / Discretionary = 0.
- **Eccezione**: per il **lusso quotato in EU** si applica un *premio* (+5)
  al posto dello sconto settoriale, perché il mercato accetta multipli più
  alti su lusso.

Da qui escono `pe_theoretical`, `target_y1 = pe_theoretical × forward_eps`,
`discount_pct = (target − price) / price`.

---

## 2. Le 4 zone di valutazione (campo `zone_rank`)

Ratio = `(price / forward_eps) / pe_theoretical` = quanto del multiplo
teorico il mercato sta effettivamente pagando.

| `zone_rank` | Etichetta | Ratio | Significato |
|---|---|---|---|
| 0 | **Affare** | ≤ 35% | Mercato paga meno di 1/3 del multiplo giusto |
| 1 | **Sconto** | 35–55% | Spazio significativo di espansione |
| 2 | **Equa** | 55–85% | Prezzo grossomodo allineato al modello |
| 3 | **Cara** | > 85% | Multiplo già pieno, poco margine |
| 4 | **N/D** | — | Dati insufficienti |

L'ordinamento dei top picks è **prima per zone_rank** (Affare prima),
**poi per discount_pct** decrescente.

---

## 3. Zona di ricarico STORICA (forward P/E history)

Aggiunta sopra al modello di valutazione assoluta. Risponde alla domanda
diversa: *"il P/E attuale è basso anche rispetto alla storia di questo
titolo?"*

### La regola

> **Un titolo è in zona di ricarico storica quando il suo forward P/E
> attuale è ≤ Q1 (25° percentile) della distribuzione del forward P/E
> sugli ultimi 5 anni.**

Da queste zone, storicamente, il multiplo si è sempre ricompresso verso
la mediana. Esempio canonico citato da Serafini: **NVDA vicino a 20×
forward P/E** è stato il bottom storico — sempre ricomprata da lì in poi.

### Metodologia: "hindsight NTM"

Le stime consensus storiche non sono disponibili nel piano FMP. Usiamo
quindi lo stesso trucco di Tikr: per ogni data passata D,

```
NTM_EPS(D)  = somma dei 4 EPS trimestrali con period-end > D
forward_PE(D) = price(D) / NTM_EPS(D)
```

Cioè usiamo gli utili *effettivamente riportati* nei 12 mesi successivi a
ogni data storica. È quello che il consensus "avrebbe dovuto stimare"
con perfect foresight — non identico al forward P/E vero ma molto vicino,
e bastante per identificare le zone di sconto storico.

### Output del modello

Per ogni ticker l'API `/api/screener/pe-history/<ticker>` restituisce:

- `series`: ~145 punti settimanali su ~5 anni `[{date, pe}, ...]`.
- `stats`:
  - `q1` / `median` / `q3` della distribuzione storica
  - `min` / `max` come riferimento
  - `current_pe` (forward P/E corrente, basato sul consensus attuale)
  - `current_percentile` (dove cade `current_pe` nella distribuzione storica)
  - `in_buy_zone`: bool, true se `current_pe ≤ q1`
  - `count`: numero di punti della serie

Il dato è cachato in MongoDB (`pe_history_cache`) con TTL **7 giorni** per
ticker.

### Cosa mostra la card

- Badge verde **"Zona ricarico storica"** quando `in_buy_zone = true`.
- Altrimenti badge grigio con `P/E xx.x× · NN° percentile`.
- Sparkline 5y con:
  - **Banda verde sotto Q1** = la "buy zone"
  - **Banda grigia sopra Q3** = la "zona cara" storica
  - **Mediana** come linea tratteggiata
  - **Pallino bianco** = punto attuale (verde = in zona ricarico, giallo =
    fair, rosso = > 75° perc.)
- Riga "Ultime zone Q1": fino a 3 date di entrata storica nel quartile basso
  — utile per vedere quanti mesi/anni fa il titolo è stato in zona di
  ricarico.

---

## 4. Cosa fa SCATTARE il riacquisto (combinare le due regole)

Le due viste (assoluta e relativa) si rinforzano:

| Zona assoluta | Zona storica | Lettura |
|---|---|---|
| Affare/Sconto | In zona Q1 | **Setup ideale**: sottoquotata *e* in minimo storico di multiplo |
| Affare/Sconto | Fuori da Q1 | Sottoquotata su modello ma il mercato la sta già pagando relativamente bene |
| Equa/Cara | In zona Q1 | Attenzione: forse il modello sovrastima la crescita futura — il mercato sta scontando un rallentamento |
| Equa/Cara | Fuori da Q1 | Niente edge — passare oltre |

**Confirm pattern** (parte di volumi/azione del prezzo, NON ancora in
questa app): sopra al badge serve la conferma del pattern *sforzo /
risultato* sulla barra a volume elevato successiva. Da implementare a parte.

---

## 5. Caveat importanti

- **Free Cash Flow**: il modello P/E è ingenuo verso aziende che bruciano
  cassa (es. Meta nelle fasi di forte capex su OpenAI). Verificare che
  l'FCF cresca insieme agli utili — altrimenti il multiplo non si espande
  anche se gli EPS salgono.
- **Trimestrali**: nei ~5 giorni attorno a una pubblicazione di utili la
  regola va sospesa — la volatilità implicita fa saltare tutti i conti
  fino a stabilizzazione.
- **Crescite "infinite"**: se `growth > 50%` il modello dà target irreali.
  In quei casi il mercato di solito comprime già il multiplo (es. Solar
  Edge cresce 262% ma quota 449×). Sanity check sempre il `current_percentile`.
- **Settori ciclici**: per Energy / Materials il P/E è meno informativo
  dei prezzi delle commodity sottostanti. La regola "zona Q1" funziona
  comunque ma con peso minore — vanno guardati i futures del sottostante.
- **Backwardation del VIX**: quando arriva la paura sistemica (VIX
  spot > VIX futuro), tutti i titoli high-beta scendono indipendentemente
  dal multiplo. La regola di ricarico storico è valida in regime
  normale; in backwardation prevale il filtro macro di Serafini ("flat
  o ruota su low-beta").

---

## 6. Origine dei numeri / file da consultare

- `_calculate_damodaran_target()` — formula `pe_theoretical` (in `app.py`).
- `_compute_zone_rank()` — soglie 35% / 55% / 85% per zone assolute.
- `_compute_forward_pe_history_fmp()` — costruzione serie 5y, calcolo Q1/Q3.
- `/api/screener/pe-history/<ticker>` — endpoint che alimenta la sparkline.
- Sparkline rendering: `buildPEHistorySVG()` in `templates/screener.html`
  e `templates/portfolio.html`.

---

## 7. Cosa NON è ancora in questa app (roadmap)

- Pattern volumi verticali (sforzo / risultato sulle barre ad alto volume).
- Volume profile orizzontale + aree di valore (POC / Low Volume Nodes).
- Gamma exposure overlay sui livelli SPX / NDX.
- Filtro VIX backwardation per disinvestimento azionario.
- Beta / scorrelazione settoriale a livello di portafoglio (oltre al
  doughnut esistente).
- Calendario trimestrali con flag "evitare la regola entro N giorni dalla earnings".
