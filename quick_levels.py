"""
Script per estrarre i 3 livelli più importanti e gamma flip dal PDF 0DTE
"""
import pdfplumber
import pandas as pd
import sys


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


def analyze_key_levels(df: pd.DataFrame, current_price: float = None):
    """Estrae solo i 3 livelli chiave e il gamma flip"""
    
    if df.empty:
        print("❌ Nessun dato trovato")
        return
    
    print("=" * 70)
    print("ES FUTURES 0DTE - LIVELLI CHIAVE")
    print("=" * 70)
    print()
    
    # Gamma Flip
    positive = df[df['Gamma_Exposure'] > 0].copy()
    negative = df[df['Gamma_Exposure'] < 0].copy()
    
    gamma_flip = None
    if not positive.empty and not negative.empty:
        pos_max_strike = positive['Strike'].max()
        neg_min_strike = negative['Strike'].min()
        gamma_flip = (pos_max_strike + neg_min_strike) / 2
    
    # Prezzo corrente
    if current_price:
        print(f"💰 PREZZO CORRENTE: {current_price:.2f}")
    
    # Gamma Flip
    if gamma_flip:
        print(f"🎯 GAMMA FLIP LEVEL: {gamma_flip:.2f}")
        if current_price:
            if current_price > gamma_flip:
                print(f"   ➜ Prezzo SOPRA flip: Regime Low Volatility (Mean Reversion)")
            else:
                print(f"   ➜ Prezzo SOTTO flip: Regime High Volatility (Trend Following)")
    
    print()
    print("=" * 70)
    
    # Filtra per posizione rispetto al prezzo
    if current_price:
        supports = positive[positive['Strike'] < current_price].copy()
        resistances = negative[negative['Strike'] > current_price].copy()
    else:
        supports = positive
        resistances = negative
    
    # TOP 3 SUPPORTI
    print()
    print("💪 TOP 3 SUPPORTI (sotto il prezzo)")
    print("-" * 70)
    if not supports.empty:
        top3_supports = supports.nlargest(3, 'Gamma_Exposure')
        for i, (_, row) in enumerate(top3_supports.iterrows(), 1):
            print(f"S{i}  {row['Strike']:.2f}  |  Gamma: +{row['Gamma_Exposure']:,.0f}  |  Calls: {row['Call_OI']:,.0f}  Puts: {row['Put_OI']:,.0f}")
    else:
        print("   Nessun supporto sotto il prezzo corrente")
    
    # TOP 3 RESISTENZE
    print()
    print("🚧 TOP 3 RESISTENZE (sopra il prezzo)")
    print("-" * 70)
    if not resistances.empty:
        top3_resistances = resistances.nsmallest(3, 'Gamma_Exposure')
        for i, (_, row) in enumerate(top3_resistances.iterrows(), 1):
            print(f"R{i}  {row['Strike']:.2f}  |  Gamma: {row['Gamma_Exposure']:,.0f}  |  Calls: {row['Call_OI']:,.0f}  Puts: {row['Put_OI']:,.0f}")
    else:
        print("   Nessuna resistenza sopra il prezzo corrente")
    
    print()
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 quick_levels.py <file.pdf> [--current-price <prezzo>]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    current_price = None
    
    if '--current-price' in sys.argv:
        idx = sys.argv.index('--current-price')
        if idx + 1 < len(sys.argv):
            current_price = float(sys.argv[idx + 1])
    
    df = extract_0dte_data(pdf_path)
    
    if df.empty:
        print("❌ Nessun dato trovato nel PDF")
        sys.exit(1)
    
    analyze_key_levels(df, current_price)


if __name__ == '__main__':
    main()
