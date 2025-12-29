"""
Script per estrarre solo i dati 0DTE (0 Days To Expiration) dal PDF
"""
import pdfplumber
import pandas as pd
import sys
from tabulate import tabulate


def extract_0dte_data(pdf_path: str) -> pd.DataFrame:
    """Estrae solo i dati 0DTE dal PDF Open Interest Matrix"""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            
            for table in tables:
                if not table or len(table) < 3:
                    continue
                
                # Cerca la tabella con STRIKE
                df = pd.DataFrame(table)
                
                # Trova la riga con "STRIKE"
                strike_row = None
                for idx, row in df.iterrows():
                    if 'STRIKE' in str(row.iloc[0]).upper():
                        strike_row = idx
                        break
                
                if strike_row is None:
                    continue
                
                # Le prime colonne dopo STRIKE sono: C (0DTE), P (0DTE)
                # Salta le righe di header
                data_start = strike_row + 2  # Salta "STRIKE" e "C P C P..."
                
                strikes = []
                call_0dte = []
                put_0dte = []
                gamma_0dte = []
                
                for idx in range(data_start, len(df)):
                    try:
                        row = df.iloc[idx]
                        
                        # Prima colonna = Strike
                        strike_val = str(row.iloc[0]).strip()
                        if not strike_val or strike_val.lower() in ['none', 'nan', '']:
                            continue
                        
                        strike = float(strike_val.replace(',', ''))
                        
                        # La struttura è: Strike | None | Call_0DTE | Put_0DTE | Call_1DTE | Put_1DTE | ...
                        # Quindi Call 0DTE è nella colonna 2, Put 0DTE nella colonna 3
                        call_val = str(row.iloc[2]).strip() if len(row) > 2 else '0'
                        call = float(call_val.replace(',', '')) if call_val and call_val.lower() not in ['none', 'nan', ''] else 0
                        
                        put_val = str(row.iloc[3]).strip() if len(row) > 3 else '0'
                        put = float(put_val.replace(',', '')) if put_val and put_val.lower() not in ['none', 'nan', ''] else 0
                        
                        # Calcola gamma exposure per 0DTE
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
                        'Call_0DTE': call_0dte,
                        'Put_0DTE': put_0dte,
                        'Gamma_Exposure': gamma_0dte
                    })
    
    return pd.DataFrame()


def analyze_0dte(df: pd.DataFrame, current_price: float = None):
    """Analizza i dati 0DTE e stampa il report"""
    
    if df.empty:
        print("❌ Nessun dato 0DTE trovato")
        return
    
    print("=" * 70)
    print("ES FUTURES - ANALISI 0DTE (0 Days To Expiration)")
    print("=" * 70)
    print()
    
    if current_price:
        print(f"💰 Prezzo Corrente: {current_price}")
        print()
    
    # Trova gamma flip
    positive = df[df['Gamma_Exposure'] > 0].copy()
    negative = df[df['Gamma_Exposure'] < 0].copy()
    
    if not positive.empty and not negative.empty:
        # Trova il punto di crossover
        pos_max_strike = positive['Strike'].max()
        neg_min_strike = negative['Strike'].min()
        gamma_flip = (pos_max_strike + neg_min_strike) / 2
        print(f"🎯 Gamma Flip Level (0DTE): {gamma_flip:.2f}")
        print()
    
    # Filtra supporti e resistenze in base al prezzo corrente
    if current_price:
        # Supporti = gamma positivo SOTTO il prezzo corrente
        supports = positive[positive['Strike'] < current_price].copy()
        # Resistenze = gamma negativo SOPRA il prezzo corrente  
        resistances = negative[negative['Strike'] > current_price].copy()
        
        # Se non ci sono abbastanza, usa anche sopra/sotto ma con note
        use_all_positive = supports.empty
        use_all_negative = resistances.empty
    else:
        # Senza prezzo corrente, usa semplicemente gamma positivo/negativo
        supports = positive
        resistances = negative
        use_all_positive = False
        use_all_negative = False
    
    # Top supporti
    print("💪 TOP SUPPORTI 0DTE (Livelli sotto il prezzo)")
    print("-" * 70)
    if not supports.empty:
        top_supports = supports.nlargest(5, 'Gamma_Exposure')
        print(tabulate(
            top_supports[['Strike', 'Call_0DTE', 'Put_0DTE', 'Gamma_Exposure']].values,
            headers=['Strike', 'Call OI', 'Put OI', 'Gamma Exposure'],
            tablefmt='simple',
            floatfmt=(',.2f', ',.0f', ',.0f', ',.0f')
        ))
        print("\n  ℹ️  Gamma positivo = Call > Put, dealer comprano quando scende")
    elif use_all_positive:
        # Mostra comunque i livelli ma avvisa che sono sopra
        print("  ⚠️  Nessun livello sotto il prezzo corrente")
        print("  📊 Livelli con gamma positivo (sopra il prezzo):")
        top_supports = positive.nlargest(3, 'Gamma_Exposure')
        print(tabulate(
            top_supports[['Strike', 'Call_0DTE', 'Put_0DTE', 'Gamma_Exposure']].values,
            headers=['Strike', 'Call OI', 'Put OI', 'Gamma Exposure'],
            tablefmt='simple',
            floatfmt=(',.2f', ',.0f', ',.0f', ',.0f')
        ))
    else:
        print("Nessun livello di supporto trovato")
    print()
    
    # Top resistenze
    print("🚧 TOP RESISTENZE 0DTE (Livelli sopra il prezzo)")
    print("-" * 70)
    if not resistances.empty:
        top_resistances = resistances.nsmallest(5, 'Gamma_Exposure')
        print(tabulate(
            top_resistances[['Strike', 'Call_0DTE', 'Put_0DTE', 'Gamma_Exposure']].values,
            headers=['Strike', 'Call OI', 'Put OI', 'Gamma Exposure'],
            tablefmt='simple',
            floatfmt=(',.2f', ',.0f', ',.0f', ',.0f')
        ))
        print("\n  ℹ️  Gamma negativo = Put > Call, dealer vendono quando sale")
    elif use_all_negative:
        # Mostra comunque i livelli ma avvisa che sono sotto
        print("  ⚠️  Nessun livello sopra il prezzo corrente")
        print("  📊 Livelli con gamma negativo (sotto il prezzo):")
        top_resistances = negative.nsmallest(3, 'Gamma_Exposure')
        print(tabulate(
            top_resistances[['Strike', 'Call_0DTE', 'Put_0DTE', 'Gamma_Exposure']].values,
            headers=['Strike', 'Call OI', 'Put OI', 'Gamma Exposure'],
            tablefmt='simple',
            floatfmt=(',.2f', ',.0f', ',.0f', ',.0f')
        ))
    else:
        print("Nessun livello di resistenza trovato")
    print()
    
    # Statistiche
    print("📊 STATISTICHE 0DTE")
    print("-" * 70)
    print(f"Totale Strike Prices: {len(df)}")
    print(f"Range: {df['Strike'].min():.0f} - {df['Strike'].max():.0f}")
    print(f"Totale Call OI: {df['Call_0DTE'].sum():,.0f}")
    print(f"Totale Put OI: {df['Put_0DTE'].sum():,.0f}")
    print(f"Put/Call Ratio: {df['Put_0DTE'].sum() / df['Call_0DTE'].sum():.2f}" if df['Call_0DTE'].sum() > 0 else "N/A")
    print()
    
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 extract_0dte.py <file.pdf> [--current-price <prezzo>]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    current_price = None
    
    if '--current-price' in sys.argv:
        idx = sys.argv.index('--current-price')
        if idx + 1 < len(sys.argv):
            current_price = float(sys.argv[idx + 1])
    
    print("🔄 Estrazione dati 0DTE dal PDF...")
    df = extract_0dte_data(pdf_path)
    
    if df.empty:
        print("❌ Nessun dato trovato nel PDF")
        sys.exit(1)
    
    print(f"✓ Estratti {len(df)} strike prices per 0DTE")
    print()
    
    # Analizza
    analyze_0dte(df, current_price)
    
    # Opzione per esportare
    if '--export-csv' in sys.argv:
        output_file = 'dte_0_levels.csv'
        df.to_csv(output_file, index=False)
        print(f"✅ Dati esportati in: {output_file}")


if __name__ == '__main__':
    main()
