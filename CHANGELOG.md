# Changelog

Tutte le modifiche sostanziali a questa applicazione, in ordine cronologico inverso.
Formato ispirato a [Keep a Changelog](https://keepachangelog.com/it/1.1.0/) e versionamento [SemVer](https://semver.org/lang/it/).

La versione mostrata nell'header dell'app è letta direttamente da questo file: la prima riga `## [X.Y.Z]` è la versione corrente.

## [1.19.1] — 2026-08-26

### Corretto
- **Le posizioni chiuse comparivano come partecipazioni da zero euro.** IBKR le lascia nell'elenco con quantità zero per il resto della giornata: oggi INTU è stata venduta e sarebbe rimasta in pagina, con controvalore nullo, in mezzo alle posizioni vere. Ora vengono scartate — e solo lo zero, perché le quantità negative sono posizioni short e vanno tenute.
- **Il giro serale del Flex riportava indietro il portafoglio.** Il Flex fotografa la chiusura precedente, il gateway locale legge il conto in tempo reale: una scrittura del Flex alle 20:00 è più *recente* ma meno *aggiornata* di una del gateway fatta durante il giorno, e sovrascrivendola avrebbe cancellato le operazioni della giornata. Successo davvero oggi: nel conto INTU è stata chiusa e QCOM aperta, e la fotografia di ieri le avrebbe rimesse com'erano. Ora ogni scrittura porta con sé la data *del dato* (`positions_as_of`) e vince il più recente, non l'ultimo arrivato.
  - Per il Flex la data è quella del report; per gateway e Web API è l'istante della lettura.
  - `updated` nella risposta di `/api/ibkr/sync` riporta ora cosa è stato scritto davvero, non cosa era stato proposto, e `skipped` spiega perché qualcosa non è passato.
- **Seguivo l'URL legacy del Flex.** La risposta di `SendRequest` contiene un elemento `<Url>` che punta al vecchio servlet `/Universal/servlet/FlexStatementService`, che la documentazione IBKR annota come legacy e dice di ignorare. È l'errore che si propaga da mezzo internet perché la risposta te lo mette in mano. Ora l'host è fisso.

### Modificato
- **`tools/ibkr_gateway_sync.py` manda anche le posizioni, di default.** Sono live, mentre quelle del Flex sono della chiusura precedente: se durante la giornata apri o chiudi qualcosa, solo il gateway se ne accorge. Il Flex resta la rete di sicurezza per i giorni a PC spento. `--no-positions` per il comportamento di prima.

### Aggiunto
- `GET /api/ibkr/flex-status` (solo admin): riporta la *forma* delle credenziali Flex — lunghezza del token, se il query id è numerico — senza mai stamparne i valori, più l'esito di una prova reale. IBKR risponde 1020 a situazioni molto diverse e le due più frequenti si riconoscono da lì: un query id non numerico è il nome della query copiato al posto del numero, un token corto è un copia-incolla tagliato.

## [1.19.0] — 2026-08-26

### Modificato
- **Le posizioni IBKR si leggono come le partecipazioni aggiunte a mano.** Via le due tabelle: ogni titolo è una scheda con la stessa forma di quelle del portafoglio manuale — nome, settore, paese, badge di zona e analisi Damodaran (P/E attuale e teorico, target, discount) — più i dati che solo IBKR ha: quantità, carico, prezzo, investito e P&L. La provenienza del dato non deve cambiare il modo in cui si legge il portafoglio.
- **Ordinamento per importo investito, di default.** È il modo in cui si guarda un portafoglio. I controvalori vengono convertiti nella valuta base del conto prima di ordinare: senza, 6.042 EUR e 4.900 USD verrebbero confrontati come numeri nudi e l'ordine sarebbe sbagliato. Il cambio arriva da IBKR quando la sorgente lo allega (il Flex lo espone come `fxRateToBase`), altrimenti da FMP con cache a 6h. `IBKR_BASE_CURRENCY` per cambiarla, default EUR.
- **Secondo ordinamento per data earnings, dalla più vicina.** Risponde a una domanda diversa — chi riporta prima — quindi è un interruttore, non il default. I titoli senza data vanno in fondo: "sconosciuto" non è "lontano", e metterli in mezzo li farebbe sembrare una scadenza reale.
- **Ogni scheda espone i propri ordini attivi in un accordion.** Chiuso mostra quanti sono e come si dividono tra acquisto e vendita; aperto, la tabella con lato, tipo, quantità, prezzo, validità e stato. Chi non ha ordini lo dice invece di lasciare un vuoto ambiguo.
- **I titoli con soli ordini pendenti hanno la stessa rappresentazione, in una lista separata.** Sono un'esposizione potenziale, non ancora un investimento: mescolarli alle posizioni falserebbe i pesi, ometterli nasconderebbe un rischio earnings che invece c'è.

### Tecnico
- Nuova rotta `GET /api/ibkr/holdings`: unisce posizioni, ordini vivi raggruppati per simbolo e analisi Damodaran, calcolata in parallelo su 8 thread come `/api/portfolio`. Riusa il simbolo FMP già risolto per gli earnings invece di ritradurlo. `?analyze=0` salta i fondamentali.
- La pagina fa tre chiamate in parallelo e disegna in due tempi: le schede compaiono appena arriva la versione senza fondamentali — che viene solo da Mongo, 0,05s — e si ridisegnano quando arriva quella con l'analisi. Tenere la pagina vuota per secondi in attesa di FMP non avrebbe avuto senso, e se l'analisi fallisce le schede restano quelle di prima invece di sparire.
- Le analisi sono in cache 6h (`IBKR_ANALYSIS_TTL`): ogni titolo sono tre chiamate a FMP e la pagina ne chiede una ventina insieme, con il rischio concreto di sbattere contro il tetto di durata della funzione. Misurato in locale: 20 chiamate al primo caricamento, zero al secondo. Gli errori restano in cache un decimo del tempo, così un FMP momentaneamente giù non congela un titolo su "non disponibile" per sei ore.
- L'ultima risposta resta in memoria nella pagina, così cambiare ordinamento non rifà nulla.

## [1.18.1] — 2026-08-26

### Corretto
- **Lo script del gateway tornava zero ordini invece di un errore.** Gli endpoint `/iserver` del Client Portal Gateway vogliono la sessione innescata da una chiamata a `/iserver/accounts`: senza, rispondono `200` con una lista vuota. Era il modo peggiore di fallire, perché "nessun ordine" è una risposta plausibile e avrebbe svuotato la lista in pagina facendo credere che non ci fossero bracket aperti. Ora la sessione viene innescata prima, e la lettura viene ritentata perché la prima chiamata avvia uno snapshot e risponde senza aspettarlo. Sul conto reale: 95 ordini, 67 vivi su 20 simboli.
- **La riautenticazione veniva dichiarata fallita quando invece riusciva.** `/iserver/reauthenticate` risponde subito `{"message":"triggered"}` ma impiega una decina di secondi a ristabilire la sessione, e lo script ricontrollava lo stato immediatamente dopo: un falso negativo garantito. Ora attende fino a 30 secondi. Se comunque non risale, il messaggio ricorda che IBKR ammette una sola sessione per utenza e che TWS o un'altra app possono averla presa.
- **Sotto-codici di comparto nelle borse IBKR.** IBKR qualifica le borse con un suffisso — `NASDAQ.NMS`, `BVME.ETF`, `LSEIOB1` — che non cambia il mercato di quotazione ma mancava in tabella. Per i titoli USA il risultato era comunque giusto, ma per caso: cadendo sul simbolo nudo. Ora si prova il codice intero e poi la radice, così i comparti non vanno enumerati uno per uno. Verificato sui 20 simboli reali: si risolvono tutti tranne `CSG1` (CSG NV, che FMP non copre affatto) e `SSLN` (un ETC, che earnings non ne ha).

## [1.18.0] — 2026-08-26

### Aggiunto
- **Sorgenti IBKR ibride: Flex Web Service per le posizioni, gateway locale per gli ordini.** L'OAuth first party è risultato non abilitabile sul conto — il Self-Service Portal risponde 501 alla scrittura del consumer key, e la spiegazione nota è che l'accesso è riservato ai conti Financial Advisor e Institutional. Il codice OAuth resta dov'è, inerte dietro `_ibkr_api_configured()`, e si riaccende da solo il giorno in cui le credenziali ci fossero.
  - Il Flex Web Service copre le posizioni senza niente da tenere acceso: token annuale, due GET, nessun gateway e nessun login. Gli ordini di lavoro però non li espone, e sono metà del valore dell'alert visto quanti bracket GTC ci sono aperti.
  - Per gli ordini c'è `tools/ibkr_gateway_sync.py`, da lanciare col Client Portal Gateway acceso: legge gli ordini vivi e li deposita su `/api/ibkr/sync`.
- **Freschezza degli ordini dichiarata ovunque.** Un alert costruito su ordini di due giorni fa ha esattamente lo stesso aspetto di uno costruito su ordini veri, quindi va detto invece che lasciato intuire. Oltre 36h (`IBKR_ORDERS_STALE_AFTER`) la pagina lo scrive in giallo accanto al titolo della tabella e nel banner, e Telegram e la mail aggiungono una riga: "l'alert copre le sole posizioni". L'intestazione del blocco dice anche da quale sorgente vengono le posizioni.

### Modificato
- **`/api/ibkr/sync` accetta payload parziali e fonde invece di sostituire.** Mandare solo `orders` aggiorna gli ordini e lascia intatte le posizioni, e viceversa. Senza questo, il giro notturno del Flex avrebbe cancellato ogni sera gli ordini raccolti di giorno dal gateway. Chiave assente significa "non ho notizie", chiave presente ma vuota significa "non c'è più niente": distinguerle serve perché un gateway che non trova ordini deve poter svuotare la lista.
  - Lo snapshot tiene ora timestamp e sorgente separati per posizioni e ordini.

### Tecnico
- Il messaggio Telegram escapa solo `& < >`, i tre caratteri documentati da Telegram. Con l'escape completo gli apostrofi diventavano `&#x27;` e comparivano tali e quali nel messaggio, perché le entità numeriche Telegram non le converte.
- Il parser Flex ritenta sul report ancora in generazione — `SendRequest` restituisce un reference code, ma il documento non è pronto subito — e legge la data di riferimento dall'attributo `toDate` di `<FlexStatement>`, che è un attributo e non un elemento.
- Nuove variabili: `IBKR_FLEX_TOKEN`, `IBKR_FLEX_QUERY_ID`, `IBKR_ORDERS_STALE_AFTER`.

## [1.17.0] — 2026-08-26

### Aggiunto
- **L'app legge Interactive Brokers da sola: OAuth 1.0a first party sulla Web API.** Con le credenziali generate dal Self-Service Portal non serve più niente in mezzo — né gateway, né sessione Claude, né PC acceso. Posizioni *e* ordini pendenti arrivano live, che è il motivo per cui si è scelta questa strada invece del Flex Web Service: il Flex non espone gli ordini di lavoro, e l'alert avrebbe perso tutti i bracket GTC.
  - L'handshake non è OAuth standard: c'è di mezzo uno scambio Diffie-Hellman. Si firma RSA-SHA256 una POST a `/oauth/live_session_token` mettendo in testa alla base string il token secret decifrato, dalla risposta si ricava il segreto condiviso e da lì un token valido 24h, e da quel momento ogni chiamata si firma HMAC-SHA256. Il passo request/access token del protocollo va saltato: per il first party quei valori vengono dal portale e chiamarlo darebbe errore.
  - Il live session token vive in memoria *e* su Mongo (`ibkr_session`): su Vercel ogni cold start rifarebbe l'handshake, che è la parte lenta e con rate limit. Si rinnova con 5 minuti di margine, perché un token che scade a metà sequenza produce un fallimento parziale, il caso peggiore da diagnosticare.
  - La firma del token che IBKR rimanda viene verificata invece che ignorata: se non combacia la colpa è quasi sempre di una chiave sbagliata, e accorgersene subito evita una serie di 401 opachi.
- **Job giornaliero server-side su Vercel Cron.** `GET /api/ibkr/cron` fa il giro completo — legge IBKR, salva lo snapshot, calcola l'alert del giorno dopo, notifica su Telegram e via mail. Schedulato alle 18:00 UTC in `vercel.json`.
- **Mail via SMTP.** Con il job che gira sul server la mail non può più passare da Gmail lato client: `_send_alert_email()` usa `smtplib` (SSL su 465 o STARTTLS su 587). Come Telegram non solleva mai — se le credenziali mancano il job completa comunque e resta il canale Telegram.
- **`GET /api/ibkr/oauth-status`, riservata agli admin.** Diagnostica dell'handshake passo per passo: quali variabili ci sono, se le due chiavi RSA si caricano, se il live session token si ottiene, se la sessione di brokeraggio si apre, quante posizioni e ordini tornano. Senza, un 401 di IBKR non dice se hai sbagliato chiave di firma, chiave di encryption, consumer key o primo DH.

### Tecnico
- Nuova dipendenza `cryptography` per firma RSA-SHA256 e decifratura PKCS#1 v1.5. Le primitive sono state verificate contro l'implementazione di riferimento di IBKR prima di toccare credenziali vere: base string identica in tutte le combinazioni, e la codifica dell'intero DH — che deve seguire la convenzione BigInteger di Java, con lo zero in testa quando il bit alto è a 1 — confrontata su 4000 casi casuali. Un handshake DH completo è stato simulato facendo entrambe le parti: token e firma coincidono.
- Le chiavi private si leggono da variabile d'ambiente in tre forme: PEM multilinea, PEM con i newline resi come `\n`, o PEM in base64. Circolano tutte e tre, e sbagliare formato produrrebbe un errore di firma incomprensibile molto più avanti.
- `compete=true` all'apertura della sessione di brokeraggio: IBKR ammette una sola sessione per username, senza quel flag basterebbe la TWS aperta a far fallire il job.
- Nuove variabili: `IBKR_CONSUMER_KEY`, `IBKR_ACCESS_TOKEN`, `IBKR_ACCESS_TOKEN_SECRET`, `IBKR_SIGNATURE_KEY`, `IBKR_ENCRYPTION_KEY`, `IBKR_DH_PRIME`, `IBKR_DH_GENERATOR`, `IBKR_REALM`, `IBKR_ACCOUNT_ID`, `CRON_SECRET`, `SMTP_*`, `ALERT_EMAIL_TO`.

## [1.16.1] — 2026-08-26

### Modificato
- **Il blocco Interactive Brokers non sparisce più quando manca lo snapshot: spiega cosa manca.** Prima, finché il job non aveva mai sincronizzato, la sezione restava nascosta — indistinguibile da una funzione rotta. Ora resta visibile e mostra un elenco di controlli: MongoDB raggiungibile, `IBKR_SYNC_TOKEN` configurato, `FMP_API_KEY` presente, Telegram attivo. In più segnala il caso più insidioso, cioè quando la sync scrive su un'email diversa da quella con cui si è fatto login: lo snapshot verrebbe salvato correttamente e la pagina continuerebbe a non vedere nulla.
  - `GET /api/ibkr/snapshot` restituisce un blocco `diagnostics` quando non trova documenti. Solo booleani e l'email dell'utente autenticato, nessun valore di token.

## [1.16.0] — 2026-08-26

### Aggiunto
- **Posizioni e ordini Interactive Brokers nella pagina portafoglio, con la data della prossima trimestrale su ogni riga.** Sopra al portafoglio manuale compare un blocco IBKR con le posizioni aperte (quantità, carico, controvalore, P&L) e gli ordini ancora eseguibili (lato, tipo, prezzo, validità). Su ogni riga un badge dice quando riporta quel titolo: rosso se è oggi o domani, ambra entro la settimana, neutro oltre. Le righe che riportano l'indomani sono evidenziate anche in tabella, così l'informazione si vede senza leggere la colonna.
  - Gli ordini in stato `REPLACED` restano nello snapshot ma non compaiono e non contano negli alert: sono la versione superata di un ordine che IBKR ha già rimpiazzato, contarli significherebbe elencare due volte lo stesso ordine.
  - Un banner in testa al blocco riassume chi riporta il giorno dopo, e dichiara esplicitamente i simboli per cui FMP non ha un calendario earnings — un "nessun earning domani" che nasconde un titolo non risolto sarebbe peggio di nessuna informazione.
- **Alert earnings sul giorno successivo, via Telegram e via mail.** Un job schedulato alle 20:00 legge posizioni e ordini da IBKR, li posta su `/api/ibkr/sync` e riceve indietro l'alert già impaginato nei tre formati (oggetto, testo Telegram, HTML mail). Il messaggio non elenca solo i ticker: per ciascuno riporta la posizione col controvalore e gli ordini pendenti con i loro prezzi, perché serve a decidere se ridurre o spostare uno stop, non solo a sapere che c'è una trimestrale.
  - Il venerdì sera l'alert guarda al lunedì invece che al sabato: le trimestrali nel weekend non escono, e una notifica che dice sempre "nessun earning domani" smette di essere letta.
  - Il giorno è calcolato su `Europe/Rome`, non su UTC né sull'ora della macchina che serve la richiesta: la notifica parte alle 20:00 italiane e parla del "giorno dopo" di chi la riceve.

### Tecnico
- IBKR non è raggiungibile dall'app: la Client Portal API richiede un gateway locale con login giornaliero, che su Vercel non può esistere. Lo snapshot arriva quindi da fuori e l'app fa da deposito e da motore di arricchimento — nuova collection `ibkr_snapshot`, un documento per proprietario, chiavata sull'email e non sullo `user_key` di sessione: un conto IBKR appartiene a una persona, non alla particolare identità Google con cui quella persona ha fatto login, e il job che lo scrive gira headless.
- Tre rotte nuove: `POST /api/ibkr/sync` (ingest + notifica, bearer token `IBKR_SYNC_TOKEN`), `GET /api/ibkr/snapshot` (lettura per la pagina, sessione) e `/api/ibkr/earnings-alert` (GET di anteprima in sessione, POST col token per rimandare la notifica senza rifare la sync). La guardia `_require_login` lascia passare `/api/ibkr/*` solo se il bearer token è valido; il controllo vero resta comunque dentro la rotta.
- **Traduzione simbolo IBKR → simbolo FMP.** IBKR nomina gli strumenti per borsa di quotazione, FMP per suffisso: Grifols è `GRF` su BM e `GRF.MC` su FMP, CSG NV è `CSG1` su AEB e `CSG1.AS`. Il simbolo nudo non basta e nemmeno è univoco — `GRF` da solo risolve a dieci strumenti in cinque paesi. La risoluzione prova i candidati in ordine di affidabilità della fonte (borsa, poi paese, poi valuta) e si ferma al primo su cui FMP risponde. Per i casi che nessuna regola può indovinare c'è `IBKR_SYMBOL_MAP` (Amplifon è `AMP2` su IBKR e `AMP.MI` su FMP, ed è già mappata).
- Le date earnings vengono da `stable/earnings` di FMP, in parallelo su 8 thread come `/api/portfolio`, con cache in memoria a 6h (`EARNINGS_CACHE_TTL`). La pagina riusa le date risolte dall'ultima sync invece di rifare una ventina di chiamate a ogni caricamento; il pulsante *Earnings* nel blocco IBKR forza il ricalcolo.
- `_telegram_send()` non solleva mai: la notifica è un canale accessorio, un token scaduto non deve far fallire la sync che l'ha innescata. Nuove variabili in `.env.example`: `IBKR_SYNC_TOKEN`, `IBKR_SYNC_USER_EMAIL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `IBKR_SYMBOL_MAP`, `EARNINGS_CACHE_TTL`, `MONGODB_IBKR_COLLECTION`.

## [1.15.0] — 2026-08-24

### Modificato
- **FMP è ora la fonte unica dello screener: rimosso il fallback su Yahoo Finance.** Prima, quando FMP non copriva un titolo, i fondamentali venivano ricalcolati da yfinance. Il problema non era la disponibilità del dato ma la sua omogeneità: la growth yfinance nasce da LTG o dal +1y, quella FMP dal CAGR del consenso sugli anni fiscali futuri: due misure diverse mescolate nella stessa lista e ordinate insieme per discount. Ora un titolo che FMP non copre semplicemente non compare, invece di comparire con numeri costruiti in un altro modo.
  - `_fetch_ticker_fundamentals()` chiama solo FMP; `_fetch_ticker_fundamentals_yf()` è stata rimossa (~120 righe) insieme alla variabile d'ambiente `SCREENER_DISABLE_YF_FALLBACK`, che regolava un fallback che non esiste più.
  - Il badge <em>YF</em> resta visibile solo sulle righe in cache calcolate prima del cambio, finché non vengono ricalcolate.
- **Anche la ricerca tipo-ahead passa da FMP.** Prima era un proxy sull'API di ricerca di Yahoo, che proponeva simboli su cui FMP non ha nulla — tipicamente le linee regionali tedesche (`AG1.F`, `AG1.MU`, `AG1.HM`): risultati selezionabili che poi portavano dritti a un "dati non disponibili". Ora interroga `search-symbol` e `search-name` in parallelo, unisce i risultati mettendo davanti i match sul ticker e ne restituisce al massimo 10, quindi propone solo simboli effettivamente analizzabili. La colonna di destra del menù mostra la valuta al posto del tipo di strumento, che FMP non espone in ricerca e che sulle quotazioni estere diceva meno.

### Tecnico
- `_fmp_get()` accetta un `_timeout` opzionale (underscore per non collidere con i parametri di query, che arrivano da `**params`). La tipo-ahead lo usa a 3 secondi invece degli 8 di default: meglio nessun risultato che un campo di ricerca bloccato.
- La ricerca risponde sempre 200 anche se FMP è irraggiungibile o solleva: verificato che nei tre casi (chiave assente, risposta nulla, eccezione) torna una lista vuota e non un 500.
- `FMP_API_KEY` diventa di fatto obbligatoria per lo screener: senza chiave la lista resta vuota. Aggiornato `.env.example`.

## [1.14.0] — 2026-08-21

### Aggiunto
- **Due nuovi mercati nello screener: Paesi Bassi e Spagna.** Schede 🇳🇱 Euronext Amsterdam (`.AS`) e 🇪🇸 Borsa di Madrid (`.MC`), 30 titoli ciascuna. Gli universi non sono compilati a memoria: vengono dallo stock screener FMP filtrato per borsa e capitalizzazione > 2B, presi in ordine di market cap. Entrambi quotano in euro, quindi i filtri della strategia (mcap ≥ 2B, guard sull'EPS) valgono senza conversioni di valuta. Sconto-paese `EU` (−5) come la Germania.
  - Esclusi da Madrid i cross-listing latinoamericani col prefisso `X` (`XVALO.MC` = Vale, `XBBDC.MC` = Bradesco): sono titoli brasiliani e avrebbero preso lo sconto-paese europeo.
  - Amsterdam porta in dote Shell in euro (`SHELL.AS`) evitando la linea di Londra, che quota in pence e falserebbe il P/E.
- **Il bucket "Lusso" viene finalmente assegnato.** Il premio di +5 sul P/E teorico era definito nella tabella degli sconti e aveva già icona e colore nella UI, ma nessun titolo poteva riceverlo: la classificazione partiva dal solo settore GICS, e il lusso non è un settore GICS — quei titoli finivano tutti in "Discretionary" (sconto 0). Ora il bucket si risolve prima dall'`industry` (FMP la espone: LVMH è `Luxury Goods`) e poi da una lista curata di ticker per i casi che l'industry non cattura — Ferrari è `Auto - Manufacturers`, Moncler e Cucinelli sono `Apparel - Manufacturers`, indistinguibili dall'abbigliamento di massa. Zara resta correttamente "Discretionary".

### Tecnico
- Nuova `_resolve_bucket(sector, industry, ticker)` usata da entrambi i fetcher (FMP e yfinance) al posto della lettura diretta di `_SCREENER_GICS_TO_BUCKET`. Il campo `industry` viene ora salvato nella riga dello screener.
- La lista `_SCREENER_LUXURY_TICKERS` è una scelta di metodo sulla strategia, non un dettaglio tecnico: contiene Ferrari, Moncler, Cucinelli, Tod's, Richemont, LVMH, Hermès, Kering e Prada, ed è pensata per essere modificata a mano.

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
