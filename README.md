# ES Gamma Analyzer

Strumento per estrarre e analizzare i dati di gamma exposure da file PDF per il trading sul future ES (E-mini S&P 500).

## 🎯 Cosa fa questo tool?

Questo progetto analizza i dati di gamma exposure delle opzioni per identificare:

- **Gamma Flip Level**: Il livello critico dove il gamma cambia da positivo a negativo
- **Livelli di Supporto**: Strike prices con alto gamma positivo (dealer comprano quando scende)
- **Livelli di Resistenza**: Strike prices con alto gamma negativo (dealer vendono quando sale)
- **Regime di Mercato**: Determina se il mercato è in regime di alta o bassa volatilità

## 📦 Installazione

```bash
# Crea un ambiente virtuale (opzionale ma consigliato)
python -m venv venv
source venv/bin/activate  # Su Windows: venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt
```

## 🚀 Utilizzo

### Utilizzo Base

```bash
python main.py percorso/al/tuo/file.pdf
```

### Con Prezzo Corrente

```bash
python main.py OpenInterest-12.pdf --current-price 6050
```

### Esporta i Livelli in CSV

```bash
python main.py OpenInterest-12.pdf --export-csv
```

### Salva Report su File

```bash
python main.py OpenInterest-12.pdf --output report.txt
```

### Personalizza Numero di Livelli

```bash
python main.py OpenInterest-12.pdf --support-levels 5 --resistance-levels 5
```

## 📊 Esempio di Output

```
======================================================================
ES FUTURES - GAMMA EXPOSURE ANALYSIS
======================================================================

📊 INFORMAZIONI
----------------------------------------------------------------------
TICKER: ES
DATE: 12/29/2025

🎯 LIVELLI CHIAVE
----------------------------------------------------------------------
Prezzo Corrente: 6050
Gamma Flip Level: 6025.00
  ➜ Livello critico dove il gamma cambia segno

📈 REGIME DI MERCATO
----------------------------------------------------------------------
Regime: Positive Gamma (Low Volatility)
Strategia: Mean reversion - vendere breakout, comprare pullback

💪 LIVELLI DI SUPPORTO (Gamma Positivo)
----------------------------------------------------------------------
Livello    Strike    Gamma Exposure
---------  --------  ----------------
S1         6000.00   1,250,000
S2         5975.00   980,000
S3         5950.00   750,000

  ℹ️  Gamma positivo = dealer comprano quando prezzo scende (supporto)

🚧 LIVELLI DI RESISTENZA (Gamma Negativo)
----------------------------------------------------------------------
Livello    Strike    Gamma Exposure
---------  --------  ------------------
R1         6075.00   -1,100,000
R2         6100.00   -850,000
R3         6125.00   -620,000

  ℹ️  Gamma negativo = dealer vendono quando prezzo sale (resistenza)

======================================================================
📋 PIANO DI TRADING
======================================================================

🎯 SCENARIO ATTUALE
----------------------------------------------------------------------
✓ Prezzo SOPRA gamma flip (6050 > 6025.00)
  • Ambiente: Bassa volatilità (gamma positivo)
  • Comportamento: Mean reversion

📌 OPPORTUNITÀ DI TRADING:
  • VENDERE rally verso resistenze
  • COMPRARE pullback verso supporti
  • Stop loss stretti (movimento limitato atteso)

  Target per SHORT:
    • R1: 6075.00
    • R2: 6100.00

  Target per LONG:
    • S1: 6000.00
    • S2: 5975.00

⚠️  LIVELLO CRITICO:
  • Watch gamma flip @ 6025.00
  • Cambio regime se prezzo attraversa questo livello
```

## 📚 Concetti Chiave

### Gamma Exposure

Il **gamma exposure** indica come i market maker (dealer) devono hedgiare le loro posizioni in opzioni:

- **Gamma Positivo** (sopra il gamma flip):
  - Dealer comprano quando il prezzo scende
  - Vendono quando il prezzo sale
  - Effetto: RIDUCE la volatilità (mean reversion)
  - Strategia: Fade i breakout, compra i dip

- **Gamma Negativo** (sotto il gamma flip):
  - Dealer vendono quando il prezzo scende
  - Comprano quando il prezzo sale
  - Effetto: AUMENTA la volatilità (trend following)
  - Strategia: Segui i breakout, evita fade

### Gamma Flip Level

Il **gamma flip** è il livello di prezzo dove il gamma totale del mercato passa da positivo a negativo. È il livello più importante per i trader perché determina il regime di volatilità del mercato.

## 🛠️ Struttura del Progetto

```
es_gamma_analyzer/
├── main.py                 # Script principale
├── pdf_extractor.py        # Estrazione dati da PDF
├── gamma_analyzer.py       # Analisi gamma exposure
├── report_generator.py     # Generazione report
├── requirements.txt        # Dipendenze Python
└── README.md              # Documentazione
```

## 📝 Note

- Assicurati che il PDF contenga una tabella con strike prices e valori di gamma exposure
- Il tool prova a identificare automaticamente le colonne rilevanti
- Se il prezzo corrente non viene fornito, il tool cerca di estrarlo dal PDF
- I livelli sono ordinati per importanza (gamma più alto/basso)

## ⚠️ Disclaimer

Questo strumento è fornito solo a scopo educativo e informativo. L'analisi generata non costituisce consulenza finanziaria. Il trading di futures comporta rischi significativi. Opera sempre con capitale che puoi permetterti di perdere.

## 📄 Licenza

MIT License - Sentiti libero di usare e modificare come preferisci.
