"""
Modulo per estrarre dati di gamma exposure da file PDF
"""
import pdfplumber
import re
import pandas as pd
from typing import Dict, List, Tuple


class GammaExposureExtractor:
    """Estrae dati di gamma exposure da PDF"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_tables(self) -> List[pd.DataFrame]:
        """Estrae tutte le tabelle dal PDF"""
        tables = []
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        if table and len(table) > 1:
                            # Converti in DataFrame
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
        
        return tables
    
    def extract_text(self) -> str:
        """Estrae tutto il testo dal PDF"""
        text = ""
        
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        
        return text
    
    def parse_gamma_data(self) -> pd.DataFrame:
        """
        Estrae e struttura i dati di gamma exposure
        Cerca colonne come: Strike, Gamma Exposure, Call Volume, Put Volume, etc.
        """
        tables = self.extract_tables()
        
        # Cerca la tabella con i dati di gamma o open interest
        gamma_df = None
        
        for i, df in enumerate(tables):
            # Salta tabelle troppo piccole
            if len(df) < 3:
                continue
                
            # Verifica se contiene colonne rilevanti
            columns_lower = [str(col).lower() for col in df.columns]
            
            # Cerca tabella con STRIKE nella prima riga
            if df.shape[1] > 2:  # Almeno 3 colonne
                first_col = str(df.iloc[0, 0]).lower() if len(df) > 0 else ""
                if 'strike' in first_col:
                    gamma_df = df
                    break
            
            if any('strike' in col for col in columns_lower) or \
               any('gamma' in col for col in columns_lower):
                gamma_df = df
                break
        
        if gamma_df is not None:
            # Pulisci i dati - formato speciale per Open Interest Matrix
            gamma_df = self._clean_open_interest_matrix(gamma_df)
        else:
            # Se non trova tabelle, prova a estrarre dal testo
            text = self.extract_text()
            gamma_df = self._parse_from_text(text)
        
        return gamma_df
    
    def _clean_open_interest_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pulisce una Open Interest Matrix e calcola il gamma exposure
        Formato: STRIKE | C | P | C | P | ... (coppie Call/Put per varie scadenze)
        """
        # La prima riga contiene "STRIKE" e altre info - rimuovila
        if 'STRIKE' in str(df.iloc[0, 0]).upper():
            df = df.iloc[1:].reset_index(drop=True)
        
        # La seconda riga contiene "C P C P..." - rimuovila
        first_row = str(df.iloc[0, 0]).strip() if len(df) > 0 else ""
        if first_row in ['C', 'c'] or len(first_row) <= 2:
            df = df.iloc[1:].reset_index(drop=True)
        
        # Rimuovi righe vuote
        df = df.dropna(how='all')
        
        # La prima colonna dovrebbe contenere gli strike prices
        # Estrai solo le colonne con dati numerici
        strikes = []
        call_volumes = []
        put_volumes = []
        
        for idx, row in df.iterrows():
            try:
                # Primo valore è lo strike
                strike_val = str(row.iloc[0]).strip()
                if not strike_val or strike_val.lower() in ['none', 'nan', '']:
                    continue
                
                strike = float(strike_val.replace(',', ''))
                
                # Somma tutti i Call e Put Open Interest
                total_calls = 0
                total_puts = 0
                
                # Scorri le colonne a coppie (C, P, C, P, ...)
                for i in range(1, len(row), 2):
                    # Colonna dispari = Call
                    if i < len(row):
                        call_val = str(row.iloc[i]).strip()
                        if call_val and call_val.lower() not in ['none', 'nan', '']:
                            try:
                                total_calls += float(call_val.replace(',', ''))
                            except:
                                pass
                    
                    # Colonna pari = Put
                    if i + 1 < len(row):
                        put_val = str(row.iloc[i + 1]).strip()
                        if put_val and put_val.lower() not in ['none', 'nan', '']:
                            try:
                                total_puts += float(put_val.replace(',', ''))
                            except:
                                pass
                
                # Calcola gamma exposure approssimato
                # Gamma exposure = (Call OI - Put OI) * qualche fattore
                # Semplificazione: Call OI positivo = supporto, Put OI = resistenza
                gamma_exposure = (total_calls - total_puts) * 1000  # Scala per visibility
                
                strikes.append(strike)
                call_volumes.append(total_calls)
                put_volumes.append(total_puts)
                
            except Exception as e:
                continue
        
        # Crea DataFrame pulito
        result_df = pd.DataFrame({
            'Strike': strikes,
            'Call_OI': call_volumes,
            'Put_OI': put_volumes,
            'Gamma_Exposure': [(c - p) * 1000 for c, p in zip(call_volumes, put_volumes)]
        })
        
        return result_df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pulisce e normalizza il DataFrame"""
        # Rimuovi righe vuote
        df = df.dropna(how='all')
        
        # Rimuovi spazi bianchi dalle colonne
        df.columns = df.columns.str.strip()
        
        # Converti colonne numeriche
        for col in df.columns:
            if 'strike' in col.lower() or 'price' in col.lower():
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
            elif 'gamma' in col.lower() or 'volume' in col.lower() or 'oi' in col.lower():
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.replace('$', ''), errors='coerce')
        
        return df
    
    def _parse_from_text(self, text: str) -> pd.DataFrame:
        """Estrae dati dal testo quando non ci sono tabelle strutturate"""
        lines = text.split('\n')
        data = []
        
        # Pattern per trovare strike prices e valori di gamma
        strike_pattern = r'(\d{3,5}(?:\.\d+)?)'
        gamma_pattern = r'(-?\$?[\d,]+(?:\.\d+)?[KMB]?)'
        
        for line in lines:
            # Cerca righe che contengono numeri che potrebbero essere strike prices
            if re.search(r'\d{4,5}', line):
                values = re.findall(r'(-?\$?[\d,]+(?:\.\d+)?[KMB]?)', line)
                if len(values) >= 2:
                    data.append(values)
        
        if data:
            # Crea DataFrame con colonne generiche
            max_cols = max(len(row) for row in data)
            df = pd.DataFrame(data)
            df.columns = [f'Col_{i}' for i in range(len(df.columns))]
            return df
        
        return pd.DataFrame()
    
    def get_metadata(self) -> Dict[str, str]:
        """Estrae metadati dal PDF (data, ticker, etc.)"""
        text = self.extract_text()
        metadata = {}
        
        # Cerca ticker (ES, SPX, etc.)
        ticker_match = re.search(r'\b(ES|SPX|SPY|QQQ)\b', text)
        if ticker_match:
            metadata['ticker'] = ticker_match.group(1)
        
        # Cerca data
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', text)
        if date_match:
            metadata['date'] = date_match.group(1)
        
        # Cerca prezzo corrente
        price_match = re.search(r'(?:current|spot|price)[\s:]+\$?([\d,]+\.?\d*)', text, re.IGNORECASE)
        if price_match:
            metadata['current_price'] = price_match.group(1)
        
        return metadata
