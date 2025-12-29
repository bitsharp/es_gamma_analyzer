"""
Script principale per analizzare gamma exposure da file PDF
"""
import sys
import argparse
from pathlib import Path
from pdf_extractor import GammaExposureExtractor
from gamma_analyzer import GammaAnalyzer
from report_generator import TradingReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Analizza gamma exposure da PDF per trading ES futures'
    )
    parser.add_argument(
        'pdf_file',
        type=str,
        help='Path al file PDF con i dati di gamma exposure'
    )
    parser.add_argument(
        '--current-price',
        type=float,
        help='Prezzo corrente del future ES (opzionale)',
        default=None
    )
    parser.add_argument(
        '--support-levels',
        type=int,
        help='Numero di livelli di supporto da identificare (default: 3)',
        default=3
    )
    parser.add_argument(
        '--resistance-levels',
        type=int,
        help='Numero di livelli di resistenza da identificare (default: 3)',
        default=3
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Esporta i livelli in formato CSV'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='File di output per il report (default: stampa su schermo)',
        default=None
    )
    
    args = parser.parse_args()
    
    # Verifica che il file esista
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"❌ Errore: File non trovato: {pdf_path}")
        sys.exit(1)
    
    print("🔄 Estrazione dati dal PDF...")
    
    # Estrai i dati
    extractor = GammaExposureExtractor(str(pdf_path))
    gamma_df = extractor.parse_gamma_data()
    metadata = extractor.get_metadata()
    
    if gamma_df.empty:
        print("❌ Errore: Nessun dato trovato nel PDF")
        print("\nTabelle estratte:")
        tables = extractor.extract_tables()
        for i, table in enumerate(tables, 1):
            print(f"\nTabella {i}:")
            print(table.head())
        sys.exit(1)
    
    print(f"✓ Estratte {len(gamma_df)} righe di dati")
    
    # Usa il prezzo dai metadati se non fornito
    current_price = args.current_price
    if current_price is None and 'current_price' in metadata:
        try:
            current_price = float(metadata['current_price'].replace(',', ''))
        except:
            pass
    
    print("🔄 Analisi gamma exposure...")
    
    # Analizza i dati
    analyzer = GammaAnalyzer(gamma_df, current_price)
    
    # Trova gamma flip
    gamma_flip = analyzer.find_gamma_flip()
    
    # Trova supporti e resistenze
    supports = analyzer.find_support_levels(args.support_levels)
    resistances = analyzer.find_resistance_levels(args.resistance_levels)
    
    # Ottieni le zone di trading
    zones = analyzer.get_trading_zones()
    
    # Calcola il profilo gamma
    gamma_profile = analyzer.calculate_gamma_profile()
    
    print("✓ Analisi completata")
    
    # Genera il report
    print("\n🔄 Generazione report...")
    
    reporter = TradingReportGenerator(zones, gamma_profile, metadata)
    report = reporter.get_full_report()
    
    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report)
        print(f"✓ Report salvato in: {output_path}")
    else:
        print("\n" + report)
    
    # Esporta CSV se richiesto
    if args.export_csv:
        csv_file = reporter.export_to_csv("es_gamma_levels.csv")
        print(f"✓ Livelli esportati in: {csv_file}")
    
    print("\n✅ Analisi completata con successo!")


if __name__ == "__main__":
    main()
