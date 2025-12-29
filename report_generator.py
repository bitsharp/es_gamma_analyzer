"""
Modulo per formattare e visualizzare i risultati dell'analisi gamma
"""
from typing import Dict, List, Tuple
import pandas as pd
from tabulate import tabulate


class TradingReportGenerator:
    """Genera report formattati per il trading"""
    
    def __init__(self, zones: Dict, gamma_profile: pd.DataFrame = None, metadata: Dict = None):
        self.zones = zones
        self.gamma_profile = gamma_profile
        self.metadata = metadata or {}
    
    def generate_summary(self) -> str:
        """Genera un sommario testuale dell'analisi"""
        lines = []
        lines.append("=" * 70)
        lines.append("ES FUTURES - GAMMA EXPOSURE ANALYSIS")
        lines.append("=" * 70)
        
        # Metadata
        if self.metadata:
            lines.append("\n📊 INFORMAZIONI")
            lines.append("-" * 70)
            for key, value in self.metadata.items():
                lines.append(f"{key.upper()}: {value}")
        
        # Prezzo corrente e Gamma Flip
        lines.append("\n🎯 LIVELLI CHIAVE")
        lines.append("-" * 70)
        
        if self.zones.get('current_price'):
            lines.append(f"Prezzo Corrente: {self.zones['current_price']}")
        
        if self.zones.get('gamma_flip'):
            lines.append(f"Gamma Flip Level: {self.zones['gamma_flip']:.2f}")
            lines.append(f"  ➜ Livello critico dove il gamma cambia segno")
        
        # Regime di mercato
        if self.zones.get('regime'):
            lines.append(f"\n📈 REGIME DI MERCATO")
            lines.append("-" * 70)
            lines.append(f"Regime: {self.zones['regime']}")
            if self.zones.get('strategy'):
                lines.append(f"Strategia: {self.zones['strategy']}")
        
        # Supporti
        if self.zones.get('support_levels'):
            lines.append(f"\n💪 LIVELLI DI SUPPORTO (Gamma Positivo)")
            lines.append("-" * 70)
            support_data = []
            for i, (strike, gamma) in enumerate(self.zones['support_levels'], 1):
                support_data.append([f"S{i}", f"{strike:.2f}", f"{gamma:,.0f}"])
            
            lines.append(tabulate(support_data, 
                                headers=['Livello', 'Strike', 'Gamma Exposure'],
                                tablefmt='simple'))
            lines.append("\n  ℹ️  Gamma positivo = dealer comprano quando prezzo scende (supporto)")
        
        # Resistenze
        if self.zones.get('resistance_levels'):
            lines.append(f"\n🚧 LIVELLI DI RESISTENZA (Gamma Negativo)")
            lines.append("-" * 70)
            resistance_data = []
            for i, (strike, gamma) in enumerate(self.zones['resistance_levels'], 1):
                resistance_data.append([f"R{i}", f"{strike:.2f}", f"{gamma:,.0f}"])
            
            lines.append(tabulate(resistance_data,
                                headers=['Livello', 'Strike', 'Gamma Exposure'],
                                tablefmt='simple'))
            lines.append("\n  ℹ️  Gamma negativo = dealer vendono quando prezzo sale (resistenza)")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def generate_trading_plan(self) -> str:
        """Genera un piano di trading basato sull'analisi"""
        lines = []
        lines.append("\n" + "=" * 70)
        lines.append("📋 PIANO DI TRADING")
        lines.append("=" * 70)
        
        gamma_flip = self.zones.get('gamma_flip')
        current = self.zones.get('current_price')
        supports = self.zones.get('support_levels', [])
        resistances = self.zones.get('resistance_levels', [])
        
        if gamma_flip and current:
            lines.append(f"\n🎯 SCENARIO ATTUALE")
            lines.append("-" * 70)
            
            if current > gamma_flip:
                lines.append(f"✓ Prezzo SOPRA gamma flip ({current} > {gamma_flip:.2f})")
                lines.append(f"  • Ambiente: Bassa volatilità (gamma positivo)")
                lines.append(f"  • Comportamento: Mean reversion")
                lines.append(f"\n📌 OPPORTUNITÀ DI TRADING:")
                lines.append(f"  • VENDERE rally verso resistenze")
                lines.append(f"  • COMPRARE pullback verso supporti")
                lines.append(f"  • Stop loss stretti (movimento limitato atteso)")
                
                if resistances:
                    lines.append(f"\n  Target per SHORT:")
                    for i, (strike, _) in enumerate(resistances[:2], 1):
                        lines.append(f"    • R{i}: {strike:.2f}")
                
                if supports:
                    lines.append(f"\n  Target per LONG:")
                    for i, (strike, _) in enumerate(supports[:2], 1):
                        lines.append(f"    • S{i}: {strike:.2f}")
            else:
                lines.append(f"✓ Prezzo SOTTO gamma flip ({current} < {gamma_flip:.2f})")
                lines.append(f"  • Ambiente: Alta volatilità (gamma negativo)")
                lines.append(f"  • Comportamento: Trend following")
                lines.append(f"\n📌 OPPORTUNITÀ DI TRADING:")
                lines.append(f"  • SEGUIRE i breakout (non fade)")
                lines.append(f"  • EVITARE mean reversion")
                lines.append(f"  • Stop loss più larghi (movimento ampio atteso)")
                
                if resistances:
                    lines.append(f"\n  Breakout LONG sopra:")
                    for i, (strike, _) in enumerate(resistances[:2], 1):
                        lines.append(f"    • {strike:.2f}")
                
                if supports:
                    lines.append(f"\n  Breakout SHORT sotto:")
                    for i, (strike, _) in enumerate(supports[:2], 1):
                        lines.append(f"    • {strike:.2f}")
            
            # Livello chiave per cambio regime
            lines.append(f"\n⚠️  LIVELLO CRITICO:")
            lines.append(f"  • Watch gamma flip @ {gamma_flip:.2f}")
            lines.append(f"  • Cambio regime se prezzo attraversa questo livello")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def export_to_csv(self, filename: str = "gamma_levels.csv"):
        """Esporta i livelli in formato CSV"""
        data = []
        
        # Gamma flip
        if self.zones.get('gamma_flip'):
            data.append({
                'Type': 'Gamma Flip',
                'Level': self.zones['gamma_flip'],
                'Gamma': 0,
                'Importance': 'Critical'
            })
        
        # Supporti
        for i, (strike, gamma) in enumerate(self.zones.get('support_levels', []), 1):
            data.append({
                'Type': f'Support S{i}',
                'Level': strike,
                'Gamma': gamma,
                'Importance': 'High' if i == 1 else 'Medium'
            })
        
        # Resistenze
        for i, (strike, gamma) in enumerate(self.zones.get('resistance_levels', []), 1):
            data.append({
                'Type': f'Resistance R{i}',
                'Level': strike,
                'Gamma': gamma,
                'Importance': 'High' if i == 1 else 'Medium'
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        return filename
    
    def get_full_report(self) -> str:
        """Genera il report completo"""
        report = self.generate_summary()
        report += self.generate_trading_plan()
        
        # Aggiungi disclaimer
        report += "\n\n" + "=" * 70
        report += "\n⚠️  DISCLAIMER"
        report += "\n" + "=" * 70
        report += "\nQuesta analisi è basata sui dati di gamma exposure delle opzioni."
        report += "\nNon costituisce consulenza finanziaria. Trading a proprio rischio."
        report += "\n" + "=" * 70 + "\n"
        
        return report
