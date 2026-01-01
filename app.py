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
        'gamma_flip_zone': None,
        'supports': [],
        'resistances': [],
        'stats': {}
    }

    # Sort by strike
    df_sorted = df.sort_values('Strike').reset_index(drop=True)

    strikes = df_sorted['Strike'].astype(float).tolist()

    flip_low = None
    flip_high = None
    flip_zone_low = None
    flip_zone_high = None

    # 1) Preferred: "around price" operational flip.
    # Pick the strike ABOVE current price (within +30pts) where |Call_OI - Put_OI| is minimal.
    if current_price is not None:
        cp = float(current_price)
        window_high = cp + 30.0
        window_df = df_sorted[(df_sorted['Strike'] > cp) & (df_sorted['Strike'] <= window_high)].copy()
        if not window_df.empty:
            window_df['abs_net'] = (window_df['Call_OI'] - window_df['Put_OI']).abs()
            best_idx = window_df['abs_net'].idxmin()
            best_pos = int(df_sorted.index[df_sorted['Strike'] == float(window_df.loc[best_idx, 'Strike'])][0])

            best_strike = float(df_sorted.loc[best_pos, 'Strike'])
            prev_strike = float(df_sorted.loc[max(0, best_pos - 1), 'Strike'])
            next_strike = float(df_sorted.loc[min(len(df_sorted) - 1, best_pos + 1), 'Strike'])

            flip_low = prev_strike
            flip_high = best_strike
            flip_zone_low = prev_strike
            flip_zone_high = next_strike

    # 2) Fallback: local balance sign-change method.
    if flip_zone_low is None or flip_zone_high is None:
        W_POINTS = 25.0
        balances = []
        for s in strikes:
            puts_below = float(df_sorted[(df_sorted['Strike'] >= s - W_POINTS) & (df_sorted['Strike'] <= s)]['Put_OI'].sum())
            calls_above = float(df_sorted[(df_sorted['Strike'] >= s) & (df_sorted['Strike'] <= s + W_POINTS)]['Call_OI'].sum())
            balances.append(calls_above - puts_below)

        sign_change_candidates = []
        for i in range(1, len(strikes)):
            a = float(balances[i - 1])
            b = float(balances[i])
            if a == 0 or b == 0 or (a < 0 < b) or (a > 0 > b):
                stability = min(abs(a), abs(b))
                mid = (float(strikes[i - 1]) + float(strikes[i])) / 2
                dist = abs(mid - float(current_price)) if current_price is not None else 0.0
                sign_change_candidates.append((stability, -dist, i))

        if sign_change_candidates:
            sign_change_candidates.sort(reverse=True)
            _, _, i = sign_change_candidates[0]
            flip_low = float(strikes[i - 1])
            flip_high = float(strikes[i])

            right = float(strikes[i])
            next_strike = float(strikes[i + 1]) if (i + 1) < len(strikes) else right
            flip_zone_low = right
            flip_zone_high = next_strike

    if flip_low is not None and flip_high is not None:
        if flip_zone_low is not None and flip_zone_high is not None:
            zone_low = round(min(flip_zone_low, flip_zone_high), 2)
            zone_high = round(max(flip_zone_low, flip_zone_high), 2)
        else:
            zone_low = round(min(flip_low, flip_high), 2)
            zone_high = round(max(flip_low, flip_high), 2)

        results['gamma_flip_zone'] = {
            'low': zone_low,
            'high': zone_high
        }

        # Operational flip = midpoint of the zone
        gamma_flip = (zone_low + zone_high) / 2
        results['gamma_flip'] = round(gamma_flip, 2)

        # Regime (same semantics, using flip midpoint)
        if current_price is not None and current_price > gamma_flip:
            results['regime'] = 'Positive Gamma (Low Volatility)'
            results['strategy'] = 'Mean reversion - vendere breakout, comprare pullback'
        elif current_price is not None and current_price < gamma_flip:
            results['regime'] = 'Negative Gamma (High Volatility)'
            results['strategy'] = 'Trend following - seguire breakout, evitare fade'
        else:
            results['regime'] = 'At Gamma Flip'
            results['strategy'] = 'Cautela - punto di transizione'

        # 0DTE-style levels
        zone_low = min(results['gamma_flip_zone']['low'], results['gamma_flip_zone']['high'])
        zone_high = max(results['gamma_flip_zone']['low'], results['gamma_flip_zone']['high'])

        below_flip = df_sorted[df_sorted['Strike'] < zone_low].copy()
        # include boundary in resistances (often the first call-wall is exactly on zone_high)
        above_flip = df_sorted[df_sorted['Strike'] >= zone_high].copy()

        def _prefer_25pt_levels(df_levels: pd.DataFrame, side: str) -> pd.DataFrame:
            # Prefer strikes that are multiples of 25 when available (common "walls")
            if df_levels.empty:
                return df_levels
            df_levels = df_levels.copy()
            df_levels['is_25'] = (df_levels['Strike'] % 25 == 0)
            key_col = 'Put_OI' if side == 'put' else 'Call_OI'
            top = df_levels.nlargest(12, key_col)
            preferred = top[top['is_25']]
            if len(preferred) >= 3:
                return preferred.nlargest(3, key_col)
            # fill remaining
            remainder = top[~top['is_25']]
            combined = pd.concat([preferred, remainder], ignore_index=True)
            return combined.nlargest(3, key_col)

        # PUT supports below flip (largest Put OI)
        if not below_flip.empty:
            top_puts = _prefer_25pt_levels(below_flip, side='put')
            results['supports'] = [
                {
                    'strike': float(row['Strike']),
                    'call_oi': int(row['Call_OI']),
                    'put_oi': int(row['Put_OI']),
                    'gamma': int(row['Gamma_Exposure'])
                }
                for _, row in top_puts.iterrows()
            ]
        else:
            results['supports_note'] = 'Nessun livello sotto la zona di flip'

        # CALL resistances above flip (largest Call OI)
        if not above_flip.empty:
            top_calls = _prefer_25pt_levels(above_flip, side='call')
            results['resistances'] = [
                {
                    'strike': float(row['Strike']),
                    'call_oi': int(row['Call_OI']),
                    'put_oi': int(row['Put_OI']),
                    'gamma': int(row['Gamma_Exposure'])
                }
                for _, row in top_calls.iterrows()
            ]
        else:
            results['resistances_note'] = 'Nessun livello sopra la zona di flip'
    else:
        results['gamma_flip_note'] = 'Impossibile determinare gamma flip: nessun incrocio Call/Put trovato'
    
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
