"""
Modulo per analizzare i dati di gamma exposure e identificare livelli chiave
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class GammaAnalyzer:
    """Analizza gamma exposure per identificare livelli di trading"""
    
    def __init__(self, gamma_df: pd.DataFrame, current_price: float = None):
        self.gamma_df = gamma_df
        self.current_price = current_price
        self.strike_col = self._find_strike_column()
        self.gamma_col = self._find_gamma_column()
        
    def _find_strike_column(self) -> str:
        """Trova la colonna con gli strike prices"""
        for col in self.gamma_df.columns:
            if 'strike' in str(col).lower() or 'price' in str(col).lower():
                return col
        # Se non trova, usa la prima colonna numerica
        for col in self.gamma_df.columns:
            if pd.api.types.is_numeric_dtype(self.gamma_df[col]):
                return col
        return self.gamma_df.columns[0] if len(self.gamma_df.columns) > 0 else None
    
    def _find_gamma_column(self) -> str:
        """Trova la colonna con il gamma exposure"""
        for col in self.gamma_df.columns:
            if 'gamma' in str(col).lower() and 'exposure' in str(col).lower():
                return col
            if 'gamma_exposure' in str(col).lower().replace(' ', '_'):
                return col
            if 'gamma' in str(col).lower():
                return col
        # Se non trova, usa la seconda colonna numerica
        numeric_cols = [col for col in self.gamma_df.columns 
                       if pd.api.types.is_numeric_dtype(self.gamma_df[col])]
        if len(numeric_cols) > 1:
            return numeric_cols[1]
        return self.gamma_df.columns[1] if len(self.gamma_df.columns) > 1 else None
    
    def find_gamma_flip(self) -> float:
        """
        Trova il livello di gamma flip (dove il gamma passa da positivo a negativo)
        Questo è il livello più importante per i trader
        """
        if self.gamma_col is None:
            return None
        
        df = self.gamma_df.copy()
        df = df[[self.strike_col, self.gamma_col]].dropna()
        
        # Converti a numerico
        df[self.strike_col] = pd.to_numeric(df[self.strike_col], errors='coerce')
        df[self.gamma_col] = pd.to_numeric(df[self.gamma_col], errors='coerce')
        df = df.dropna()
        
        if len(df) < 2:
            return None
        
        # Ordina per strike
        df = df.sort_values(self.strike_col)
        
        # Trova dove il gamma cambia segno
        gamma_values = df[self.gamma_col].values
        strikes = df[self.strike_col].values
        
        for i in range(len(gamma_values) - 1):
            if gamma_values[i] * gamma_values[i + 1] < 0:  # Cambio di segno
                # Interpola per trovare il punto esatto
                gamma_flip = strikes[i] + (strikes[i + 1] - strikes[i]) * \
                            abs(gamma_values[i]) / (abs(gamma_values[i]) + abs(gamma_values[i + 1]))
                return round(gamma_flip, 2)
        
        return None
    
    def find_support_levels(self, n_levels: int = 3) -> List[Tuple[float, float]]:
        """
        Trova i livelli di supporto basati sul gamma exposure positivo
        Gamma positivo = supporto (dealers comprano quando il prezzo scende)
        """
        df = self.gamma_df.copy()
        df = df[[self.strike_col, self.gamma_col]].dropna()
        
        df[self.strike_col] = pd.to_numeric(df[self.strike_col], errors='coerce')
        df[self.gamma_col] = pd.to_numeric(df[self.gamma_col], errors='coerce')
        df = df.dropna()
        
        # Filtra solo gamma positivo
        positive_gamma = df[df[self.gamma_col] > 0].copy()
        
        if len(positive_gamma) == 0:
            return []
        
        # Ordina per gamma decrescente
        positive_gamma = positive_gamma.sort_values(self.gamma_col, ascending=False)
        
        # Prendi i top N livelli
        top_levels = positive_gamma.head(n_levels)
        
        return [(row[self.strike_col], row[self.gamma_col]) 
                for _, row in top_levels.iterrows()]
    
    def find_resistance_levels(self, n_levels: int = 3) -> List[Tuple[float, float]]:
        """
        Trova i livelli di resistenza basati sul gamma exposure negativo
        Gamma negativo = resistenza (dealers vendono quando il prezzo sale)
        """
        df = self.gamma_df.copy()
        df = df[[self.strike_col, self.gamma_col]].dropna()
        
        df[self.strike_col] = pd.to_numeric(df[self.strike_col], errors='coerce')
        df[self.gamma_col] = pd.to_numeric(df[self.gamma_col], errors='coerce')
        df = df.dropna()
        
        # Filtra solo gamma negativo
        negative_gamma = df[df[self.gamma_col] < 0].copy()
        
        if len(negative_gamma) == 0:
            return []
        
        # Ordina per gamma (più negativo = resistenza più forte)
        negative_gamma = negative_gamma.sort_values(self.gamma_col, ascending=True)
        
        # Prendi i top N livelli
        top_levels = negative_gamma.head(n_levels)
        
        return [(row[self.strike_col], row[self.gamma_col]) 
                for _, row in top_levels.iterrows()]
    
    def get_trading_zones(self) -> Dict[str, any]:
        """
        Identifica le zone di trading basate sul gamma
        """
        gamma_flip = self.find_gamma_flip()
        support = self.find_support_levels()
        resistance = self.find_resistance_levels()
        
        zones = {
            'gamma_flip': gamma_flip,
            'support_levels': support,
            'resistance_levels': resistance,
            'current_price': self.current_price
        }
        
        if gamma_flip and self.current_price:
            if self.current_price > gamma_flip:
                zones['regime'] = 'Positive Gamma (Low Volatility)'
                zones['strategy'] = 'Mean reversion - vendere breakout, comprare pullback'
            else:
                zones['regime'] = 'Negative Gamma (High Volatility)'
                zones['strategy'] = 'Trend following - seguire i breakout, evitare fade'
        
        return zones
    
    def calculate_gamma_profile(self) -> pd.DataFrame:
        """
        Calcola il profilo completo del gamma exposure
        """
        df = self.gamma_df.copy()
        
        if self.strike_col and self.gamma_col:
            df = df[[self.strike_col, self.gamma_col]].dropna()
            df[self.strike_col] = pd.to_numeric(df[self.strike_col], errors='coerce')
            df[self.gamma_col] = pd.to_numeric(df[self.gamma_col], errors='coerce')
            df = df.dropna()
            df = df.sort_values(self.strike_col)
            
            # Aggiungi colonna con il cumulative gamma
            df['Cumulative_Gamma'] = df[self.gamma_col].cumsum()
            
            # Aggiungi flag per identificare livelli chiave
            df['Level_Type'] = 'Neutral'
            
            if len(df) > 0:
                # Identifica i picchi di gamma
                df.loc[df[self.gamma_col] > df[self.gamma_col].quantile(0.9), 'Level_Type'] = 'Strong Support'
                df.loc[df[self.gamma_col] < df[self.gamma_col].quantile(0.1), 'Level_Type'] = 'Strong Resistance'
            
            return df
        
        return pd.DataFrame()
