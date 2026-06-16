"""
data_preparation.py
Učitavanje, čišćenje i RFM izračunavanje za Online Retail II dataset
(Bez skaliranja – skaliranje se radi u train.py nakon podjele podataka)
"""

import pandas as pd
import os

# -------------------------------
# PUTANJE
# -------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAW_FILE       = os.path.join(RAW_DIR, 'online_retail_II.xlsx')
SYNTHETIC_FILE = os.path.join(RAW_DIR, 'synthetic_high_clv.csv')
RFM_CLEAN      = os.path.join(PROCESSED_DIR, 'rfm_clean.csv')


def load_and_clean():
    """Učitava oba sheet-a i čisti podatke"""
    
    print("[1/2] Učitavanje podataka...")
    
    df_1 = pd.read_excel(RAW_FILE, sheet_name='Year 2009-2010')
    df_2 = pd.read_excel(RAW_FILE, sheet_name='Year 2010-2011')
    df = pd.concat([df_1, df_2], ignore_index=True)

    # Učitaj sintetičke podatke ako postoje
    if os.path.exists(SYNTHETIC_FILE):
        df_synthetic = pd.read_csv(SYNTHETIC_FILE, dtype={'Customer ID': str})
        df_synthetic['InvoiceDate'] = pd.to_datetime(df_synthetic['InvoiceDate'])
        df = pd.concat([df, df_synthetic], ignore_index=True)
        print(f"      Originalni redovi:  {len(df_1) + len(df_2):,}")
        print(f"      Sintetički redovi:  {len(df_synthetic):,}")
        print(f"      Ukupno redova:      {len(df):,}")
    else:
        print(f"      Učitano redova: {len(df):,}")
    
    print("[2/2] Čišćenje podataka...")
    
    # Pronađi ispravne nazive kolona
    if 'Invoice' in df.columns:
        invoice_col = 'Invoice'
    elif 'InvoiceNo' in df.columns:
        invoice_col = 'InvoiceNo'
    else:
        invoice_col = None
    
    if 'Customer ID' in df.columns:
        customer_col = 'Customer ID'
    elif 'CustomerID' in df.columns:
        customer_col = 'CustomerID'
    else:
        customer_col = None
    
    if 'Price' in df.columns:
        price_col = 'Price'
    elif 'UnitPrice' in df.columns:
        price_col = 'UnitPrice'
    else:
        price_col = None
    
    # Čišćenje
    if invoice_col:
        df = df[~df[invoice_col].astype(str).str.startswith('C')]
    
    if customer_col:
        df = df.dropna(subset=[customer_col])
    
    if price_col:
        df = df[(df['Quantity'] > 0) & (df[price_col] > 0)]
    
    # Računanje RFM (neskalirano)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    df['TotalPrice'] = df['Quantity'] * df[price_col]
    
    rfm = df.groupby(customer_col).agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=(invoice_col, 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
    
    rfm.rename(columns={customer_col: 'CustomerID'}, inplace=True)
    rfm = rfm[rfm['Monetary'] > 0]
    
    print(f"      Broj kupaca: {len(rfm):,}")
    
    # Sačuvaj čiste (neskalirane) RFM podatke
    rfm.to_csv(RFM_CLEAN, index=False)
    print(f"      Sačuvano: {RFM_CLEAN}")
    
    return rfm


if __name__ == '__main__':
    print("=" * 50)
    print("DATA PREPARATION ZA CLV PROJEKAT")
    print("=" * 50)
    rfm = load_and_clean()
    print("\n" + "=" * 50)
    print("PRIPREMA PODATAKA ZAVRŠENA!")
    print("=" * 50)