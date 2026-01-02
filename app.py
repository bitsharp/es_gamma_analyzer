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
import re

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


def _parse_pdf_number(value: object) -> float:
    """Parse numeric strings found in PDFs.

    Handles both:
    - US style: 1,234.56
    - EU style: 1.234,56
    - Thousand separators only: 1,234 or 1.234
    """

    raw = ("" if value is None else str(value)).strip()
    if not raw or raw.lower() in {"none", "nan", ""}:
        return 0.0

    raw = raw.replace("\u00a0", "").replace(" ", "")
    raw = raw.replace("$", "")

    negative = False
    if raw.startswith("(") and raw.endswith(")"):
        negative = True
        raw = raw[1:-1]

    # Keep only digits, separators and sign
    raw = re.sub(r"[^0-9,\.\-]", "", raw)
    if not raw or raw in {"-", ".", ","}:
        return 0.0

    has_dot = "." in raw
    has_comma = "," in raw

    try:
        if has_dot and has_comma:
            # Decide decimal separator as the rightmost of the two.
            if raw.rfind(",") > raw.rfind("."):
                # EU: '.' thousands, ',' decimal
                raw = raw.replace(".", "")
                raw = raw.replace(",", ".")
            else:
                # US: ',' thousands, '.' decimal
                raw = raw.replace(",", "")
        elif has_dot:
            # If dot-groups look like thousands (e.g. 1.234 or 12.345.678), remove dots.
            if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", raw):
                raw = raw.replace(".", "")
        elif has_comma:
            # If comma-groups look like thousands, remove commas; else treat comma as decimal.
            if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", raw):
                raw = raw.replace(",", "")
            else:
                raw = raw.replace(",", ".")

        out = float(raw)
        return -out if negative else out
    except Exception:
        return 0.0


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
    """Estrae solo i dati 0DTE dal PDF Open Interest Matrix."""

    return _extract_dte_days_data(pdf_path, target_days=0)


def extract_1dte_data(pdf_path: str) -> pd.DataFrame:
    """Estrae solo i dati 1DTE dal PDF Open Interest Matrix.

    Molti PDF hanno struttura: Strike | None | Call_0DTE | Put_0DTE | Call_1DTE | Put_1DTE | ...
    """

    return _extract_dte_days_data(pdf_path, target_days=1)


def extract_nearest_positive_dte_data(pdf_path: str) -> pd.DataFrame:
    """Fallback: estrae i dati della scadenza con DTE minimo > 0 disponibile nel PDF."""

    mapping = _find_dte_column_mapping(pdf_path)
    positive_days = sorted([d for d in mapping.keys() if isinstance(d, int) and d > 0])
    for d in positive_days:
        df = _extract_dte_days_data(pdf_path, target_days=d)
        if not df.empty:
            return df
    return pd.DataFrame()


def _extract_dte_days_data(pdf_path: str, target_days: int) -> pd.DataFrame:
    mapping = _find_dte_column_mapping(pdf_path)
    pair = mapping.get(int(target_days))
    if not pair:
        return pd.DataFrame()

    call_col, put_col = pair
    return _extract_dte_pair_data(pdf_path, call_col=call_col, put_col=put_col)


def _extract_dte_pair_data(pdf_path: str, call_col: int, put_col: int) -> pd.DataFrame:
    """Estrae una coppia Call/Put da una Open Interest Matrix usando indici colonna."""

    def _to_float(value: object) -> float:
        return _parse_pdf_number(value)

    def _is_strike(value: object) -> bool:
        try:
            raw = ("" if value is None else str(value)).strip()
            if not raw:
                return False
            parsed = _parse_pdf_number(raw)
            return parsed != 0.0 or any(ch.isdigit() for ch in raw)
        except Exception:
            return False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()

            for table in tables:
                if not table or len(table) < 3:
                    continue

                max_len = max(len(r) for r in table)
                norm = [r + [""] * (max_len - len(r)) for r in table]
                df = pd.DataFrame(norm)

                # Trova la riga con "STRIKE" (può non essere solo nella prima cella)
                strike_row = None
                for idx, row in df.iterrows():
                    joined = " ".join(str(x) for x in row.tolist())
                    if 'STRIKE' in joined.upper():
                        strike_row = idx
                        break

                if strike_row is None:
                    continue

                # Trova prima riga dati dopo header (prima colonna numerica)
                data_start = None
                for ridx in range(strike_row + 1, len(df)):
                    if _is_strike(df.iloc[ridx, 0]):
                        data_start = ridx
                        break

                if data_start is None:
                    continue

                strikes: list[float] = []
                calls: list[float] = []
                puts: list[float] = []
                gammas: list[float] = []

                for ridx in range(data_start, len(df)):
                    try:
                        row = df.iloc[ridx]
                        if not _is_strike(row.iloc[0]):
                            continue

                        strike = _to_float(row.iloc[0])
                        call = _to_float(row.iloc[call_col]) if call_col < len(row) else 0.0
                        put = _to_float(row.iloc[put_col]) if put_col < len(row) else 0.0
                        gamma = (call - put) * 1000

                        strikes.append(strike)
                        calls.append(call)
                        puts.append(put)
                        gammas.append(gamma)
                    except Exception:
                        continue

                if strikes:
                    return pd.DataFrame({
                        'Strike': strikes,
                        'Call_OI': calls,
                        'Put_OI': puts,
                        'Gamma_Exposure': gammas
                    })

    return pd.DataFrame()


def _find_dte_column_mapping(pdf_path: str) -> Dict[int, tuple[int, int]]:
    """Ritorna mappa {dte_days: (call_col, put_col)} rilevata dalla tabella.

    Supporta intestazioni tipo "EWZ5\n0 DTE" con celle vuote/None tra Call e Put.
    """

    def _parse_dte_days(cell: object) -> Optional[int]:
        if cell is None:
            return None
        text = str(cell).upper().replace('\n', ' ')
        m = re.search(r'\b(\d+)\s*DTE\b', text)
        if not m:
            m = re.search(r'\b(\d+)\s*DAYS?\b', text)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 3:
                    continue

                max_len = max(len(r) for r in table)
                norm = [r + [""] * (max_len - len(r)) for r in table]
                df = pd.DataFrame(norm)

                # find STRIKE row
                strike_row = None
                for idx, row in df.iterrows():
                    joined = " ".join(str(x) for x in row.tolist())
                    if 'STRIKE' in joined.upper():
                        strike_row = idx
                        break
                if strike_row is None:
                    continue

                # prefer next row for C/P labels
                cp_row_idx = strike_row + 1
                if cp_row_idx >= len(df):
                    continue

                # determine day label per column from STRIKE header row, propagating across blanks
                days_by_col: list[Optional[int]] = [None] * df.shape[1]
                current_days: Optional[int] = None
                for col in range(df.shape[1]):
                    parsed = _parse_dte_days(df.iloc[strike_row, col])
                    if parsed is not None:
                        current_days = parsed
                    days_by_col[col] = current_days

                # map day -> call/put columns based on C/P row
                mapping: Dict[int, Dict[str, int]] = {}
                cp_row = df.iloc[cp_row_idx]
                for col in range(df.shape[1]):
                    d = days_by_col[col]
                    if d is None:
                        continue
                    cp = str(cp_row.iloc[col] or '').strip().upper()
                    if cp not in {'C', 'P'}:
                        continue
                    mapping.setdefault(int(d), {})
                    # keep the first occurrence (leftmost)
                    if cp == 'C' and 'C' not in mapping[int(d)]:
                        mapping[int(d)]['C'] = col
                    if cp == 'P' and 'P' not in mapping[int(d)]:
                        mapping[int(d)]['P'] = col

                # finalize only complete pairs
                out: Dict[int, tuple[int, int]] = {}
                for d, cols in mapping.items():
                    if 'C' in cols and 'P' in cols:
                        out[int(d)] = (int(cols['C']), int(cols['P']))

                if out:
                    return out

    return {}


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
        
        # Estrai dati: preferisci 0DTE, fallback a 1DTE; se 1DTE manca, prova la scadenza positiva più vicina.
        df = extract_0dte_data(filepath)
        if df.empty:
            df = extract_1dte_data(filepath)
        if df.empty:
            df = extract_nearest_positive_dte_data(filepath)

        # Analizza
        results = analyze_0dte(df, current_price)

        # Messaggio più chiaro se manca sia 0DTE che 1DTE
        if isinstance(results, dict) and results.get('error') == 'Nessun dato 0DTE trovato':
            results['error'] = 'Nessun dato 0DTE trovato; ho provato anche 1DTE (e la scadenza positiva più vicina) senza successo'
        
        # Rimuovi il file temporaneo
        os.remove(filepath)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Errore durante l\'analisi: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
