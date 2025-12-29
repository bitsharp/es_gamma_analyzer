"""
Script di debug per visualizzare i dati estratti dal PDF
"""
import pdfplumber
import pandas as pd
from pdf_extractor import GammaExposureExtractor

pdf_path = "/Users/lucataurisano/Downloads/OpenInterest-12.pdf"

print("=" * 70)
print("DEBUG - ESTRAZIONE PDF")
print("=" * 70)

# Estrai il testo completo
print("\n📄 TESTO ESTRATTO DAL PDF:")
print("-" * 70)
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        print(f"\n--- Pagina {i+1} ---")
        text = page.extract_text()
        print(text[:1000] if text else "Nessun testo trovato")

# Estrai le tabelle
print("\n\n📊 TABELLE ESTRATTE:")
print("-" * 70)
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        print(f"\n--- Pagina {i+1}: {len(tables) if tables else 0} tabelle ---")
        if tables:
            for j, table in enumerate(tables):
                print(f"\nTabella {j+1}:")
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                    print(df.head(10))
                    print(f"\nColonne: {df.columns.tolist()}")
                    print(f"Righe: {len(df)}")

# Usa l'extractor
print("\n\n🔧 DATI ESTRATTI DALL'EXTRACTOR:")
print("-" * 70)
extractor = GammaExposureExtractor(pdf_path)
gamma_df = extractor.parse_gamma_data()
print(f"\nDataFrame risultante:")
print(gamma_df)
print(f"\nColonne: {gamma_df.columns.tolist()}")
print(f"Righe: {len(gamma_df)}")
print(f"\nTipi di dato:")
print(gamma_df.dtypes)
