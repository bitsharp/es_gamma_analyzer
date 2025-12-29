"""
Script di esempio per testare l'analizzatore con dati simulati
"""
import pandas as pd
from gamma_analyzer import GammaAnalyzer
from report_generator import TradingReportGenerator


def create_sample_data():
    """Crea dati di esempio per testare l'analizzatore"""
    data = {
        'Strike': [5900, 5925, 5950, 5975, 6000, 6025, 6050, 6075, 6100, 6125, 6150],
        'Gamma_Exposure': [450000, 680000, 890000, 1200000, 1500000, 200000, -300000, -850000, -1100000, -750000, -450000]
    }
    
    return pd.DataFrame(data)


def main():
    print("🧪 Test con dati di esempio\n")
    
    # Crea dati di esempio
    gamma_df = create_sample_data()
    
    print("📊 Dati di esempio:")
    print(gamma_df)
    print()
    
    # Analizza
    current_price = 6050
    analyzer = GammaAnalyzer(gamma_df, current_price)
    
    # Trova i livelli
    gamma_flip = analyzer.find_gamma_flip()
    supports = analyzer.find_support_levels(3)
    resistances = analyzer.find_resistance_levels(3)
    
    # Ottieni le zone
    zones = analyzer.get_trading_zones()
    
    # Genera report
    metadata = {
        'ticker': 'ES',
        'date': '29/12/2025',
        'source': 'Dati di esempio'
    }
    
    reporter = TradingReportGenerator(zones, None, metadata)
    report = reporter.get_full_report()
    
    print(report)
    
    # Esporta CSV
    csv_file = reporter.export_to_csv("sample_gamma_levels.csv")
    print(f"\n✓ Dati esportati in: {csv_file}")


if __name__ == "__main__":
    main()
