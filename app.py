"""
Flask web application per analisi gamma exposure 0DTE
"""
from flask import Flask, render_template, request, jsonify
import os
import time
import csv
import io
import urllib.request
import urllib.parse
from typing import Any, Dict, Optional
import tempfile
import pdfplumber
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max


def _is_writable_dir(path: str) -> bool:
    try:
        os.makedirs(path, exist_ok=True)
        test_path = os.path.join(path, ".__write_test")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_path)
        return True
    except Exception:
        return False


def get_upload_folder() -> str:
    """Return a writable folder for uploads.

    Vercel/AWS Lambda filesystems are read-only except for /tmp.
    """

    env_folder = (os.getenv("UPLOAD_FOLDER") or "").strip()
    candidates = [p for p in [env_folder, "uploads"] if p]

    tmp_base = tempfile.gettempdir() or "/tmp"
    candidates.append(os.path.join(tmp_base, "uploads"))

    for folder in candidates:
        if _is_writable_dir(folder):
            return folder

    # Last resort: /tmp
    return tmp_base


app.config['UPLOAD_FOLDER'] = get_upload_folder()


_SP500_PRICE_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


_ES_PRICE_CACHE = {
    "value": None,
    "fetched_at": 0.0,
}


def _fetch_stooq_latest_close(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetches the latest close for a symbol from Stooq (no API key).

    Returns a dict with keys: symbol, price, date, time, source.
    """

    url = f"https://stooq.com/q/l/?s={urllib.parse.quote(symbol)}&f=sd2t2ohlcv&h&e=csv"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            raw = response.read().decode("utf-8", errors="replace")

        reader = csv.DictReader(io.StringIO(raw))
        row = next(reader, None)
        if not row:
            return None

        close_val = (row.get("Close") or "").strip()
        if not close_val or close_val.upper() in {"N/D", "NA", "NULL"}:
            return None

        return {
            "symbol": (row.get("Symbol") or symbol).strip(),
            "price": float(close_val),
            "date": (row.get("Date") or "").strip(),
            "time": (row.get("Time") or "").strip(),
            "source": "stooq",
        }
    except Exception:
        return None


def get_sp500_price_cached(max_age_seconds: int = 60) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _SP500_PRICE_CACHE.get("value")
    fetched_at = float(_SP500_PRICE_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    # Prefer the index; fall back to SPY as a proxy if the index is unavailable.
    for symbol in ("^spx", "spy.us"):
        data = _fetch_stooq_latest_close(symbol)
        if data:
            if symbol != "^spx":
                data["note"] = "Proxy (SPY) used when ^SPX unavailable"
            _SP500_PRICE_CACHE["value"] = data
            _SP500_PRICE_CACHE["fetched_at"] = now
            return data

    return None


def get_es_price_cached(max_age_seconds: int = 5) -> Optional[Dict[str, Any]]:
    now = time.time()
    cached = _ES_PRICE_CACHE.get("value")
    fetched_at = float(_ES_PRICE_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) <= max_age_seconds:
        return cached

    # ES continuous future on Stooq.
    data = _fetch_stooq_latest_close("es.f")
    if not data:
        return None

    data["instrument"] = "ES Futures"
    _ES_PRICE_CACHE["value"] = data
    _ES_PRICE_CACHE["fetched_at"] = now
    return data

def extract_0dte_data(pdf_path: str) -> pd.DataFrame:
    """Estrae solo i dati 0DTE dal PDF Open Interest Matrix"""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            
            for table in tables:
                if not table or len(table) < 3:
                    continue
                
                df = pd.DataFrame(table)
                
                # Trova la riga con "STRIKE"
                strike_row = None
                for idx, row in df.iterrows():
                    if 'STRIKE' in str(row.iloc[0]).upper():
                        strike_row = idx
                        break
                
                if strike_row is None:
                    continue
                
                data_start = strike_row + 2
                
                strikes = []
                call_0dte = []
                put_0dte = []
                gamma_0dte = []
                
                for idx in range(data_start, len(df)):
                    try:
                        row = df.iloc[idx]
                        
                        strike_val = str(row.iloc[0]).strip()
                        if not strike_val or strike_val.lower() in ['none', 'nan', '']:
                            continue
                        
                        strike = float(strike_val.replace(',', ''))
                        
                        call_val = str(row.iloc[2]).strip() if len(row) > 2 else '0'
                        call = float(call_val.replace(',', '')) if call_val and call_val.lower() not in ['none', 'nan', ''] else 0
                        
                        put_val = str(row.iloc[3]).strip() if len(row) > 3 else '0'
                        put = float(put_val.replace(',', '')) if put_val and put_val.lower() not in ['none', 'nan', ''] else 0
                        
                        gamma = (call - put) * 1000
                        
                        strikes.append(strike)
                        call_0dte.append(call)
                        put_0dte.append(put)
                        gamma_0dte.append(gamma)
                        
                    except Exception as e:
                        continue
                
                if strikes:
                    return pd.DataFrame({
                        'Strike': strikes,
                        'Call_OI': call_0dte,
                        'Put_OI': put_0dte,
                        'Gamma_Exposure': gamma_0dte
                    })
    
    return pd.DataFrame()


def analyze_0dte(df: pd.DataFrame, current_price: float = None):
    """Analizza i dati 0DTE e restituisce risultati strutturati"""
    
    if df.empty:
        return {'error': 'Nessun dato 0DTE trovato'}
    
    results = {
        'current_price': current_price,
        'gamma_flip': None,
        'supports': [],
        'resistances': [],
        'stats': {}
    }
    
    # Trova gamma flip
    positive = df[df['Gamma_Exposure'] > 0].copy()
    negative = df[df['Gamma_Exposure'] < 0].copy()
    
    if not positive.empty and not negative.empty:
        pos_max_strike = positive['Strike'].max()
        neg_min_strike = negative['Strike'].min()
        gamma_flip = (pos_max_strike + neg_min_strike) / 2
        results['gamma_flip'] = round(gamma_flip, 2)
        
        # Regime
        if current_price and current_price > gamma_flip:
            results['regime'] = 'Positive Gamma (Low Volatility)'
            results['strategy'] = 'Mean reversion - vendere breakout, comprare pullback'
        elif current_price and current_price < gamma_flip:
            results['regime'] = 'Negative Gamma (High Volatility)'
            results['strategy'] = 'Trend following - seguire breakout, evitare fade'
        else:
            results['regime'] = 'At Gamma Flip'
            results['strategy'] = 'Cautela - punto di transizione'
    
    # Filtra supporti e resistenze in base al prezzo corrente
    if current_price:
        # Supporti = gamma positivo SOTTO il prezzo corrente
        supports = positive[positive['Strike'] < current_price].copy()
        # Resistenze = gamma negativo SOPRA il prezzo corrente  
        resistances = negative[negative['Strike'] > current_price].copy()
        
        # Se non ci sono abbastanza, usa anche gli altri ma con nota
        if supports.empty and not positive.empty:
            supports = positive
            results['supports_note'] = 'Livelli sopra il prezzo corrente'
        if resistances.empty and not negative.empty:
            resistances = negative
            results['resistances_note'] = 'Livelli sotto il prezzo corrente'
    else:
        supports = positive
        resistances = negative
    
    # Top 3 supporti
    if not supports.empty:
        top_supports = supports.nlargest(3, 'Gamma_Exposure')
        results['supports'] = [
            {
                'strike': float(row['Strike']),
                'call_oi': int(row['Call_OI']),
                'put_oi': int(row['Put_OI']),
                'gamma': int(row['Gamma_Exposure'])
            }
            for _, row in top_supports.iterrows()
        ]
    
    # Top 3 resistenze
    if not resistances.empty:
        top_resistances = resistances.nsmallest(3, 'Gamma_Exposure')
        results['resistances'] = [
            {
                'strike': float(row['Strike']),
                'call_oi': int(row['Call_OI']),
                'put_oi': int(row['Put_OI']),
                'gamma': int(row['Gamma_Exposure'])
            }
            for _, row in top_resistances.iterrows()
        ]
    
    # Statistiche
    total_calls = df['Call_OI'].sum()
    total_puts = df['Put_OI'].sum()
    
    results['stats'] = {
        'total_strikes': len(df),
        'strike_range': f"{df['Strike'].min():.0f} - {df['Strike'].max():.0f}",
        'total_call_oi': int(total_calls),
        'total_put_oi': int(total_puts),
        'put_call_ratio': round(total_puts / total_calls, 2) if total_calls > 0 else None
    }
    
    return results


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/sp500-price', methods=['GET'])
def sp500_price():
    data = get_sp500_price_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare il prezzo S&P 500 in questo momento"}), 503

    return jsonify(data)


@app.route('/api/es-price', methods=['GET'])
def es_price():
    data = get_es_price_cached()
    if not data:
        return jsonify({"error": "Impossibile recuperare il prezzo ES in questo momento"}), 503

    return jsonify(data)


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'Nessun file caricato'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nessun file selezionato'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Solo file PDF sono supportati'}), 400
    
    try:
        # Salva il file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Estrai prezzo corrente se fornito
        current_price = request.form.get('current_price')
        current_price = float(current_price) if current_price else None
        
        # Estrai dati
        df = extract_0dte_data(filepath)
        
        # Analizza
        results = analyze_0dte(df, current_price)
        
        # Rimuovi il file temporaneo
        os.remove(filepath)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Errore durante l\'analisi: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
