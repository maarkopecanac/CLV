"""
data_preparation.py
Učitavanje, čišćenje i RFM izračunavanje za Online Retail II dataset
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
"""joblib cuva scaler za kasniju upotrebu, knn"""
import joblib

# -------------------------------
# PUTANJE
# -------------------------------

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)

RAW_FILE = os.path.join(RAW_DIR, 'online_retail_II.xlsx')
RFM_SCALED = os.path.join(PROCESSED_DIR, 'rfm_scaled.csv')


# -------------------------------
# 1. UČITAVANJE I ČIŠĆENJE
# -------------------------------

def load_and_clean():
    """Učitava oba sheet-a i čisti podatke"""
    
    print("[1/3] Učitavanje podataka...")
    
    # Učitaj oba sheet-a
    df_1 = pd.read_excel(RAW_FILE, sheet_name='Year 2009-2010')
    df_2 = pd.read_excel(RAW_FILE, sheet_name='Year 2010-2011')
    
    # Spoji ih
    df = pd.concat([df_1, df_2], ignore_index=True)
    
    print(f"      Učitano redova: {len(df):,}")
    
    # ISPIS SVIH KOLONA (da vidimo kako se zovu)
    print("\nKolone u datasetu:")
    for i, col in enumerate(df.columns.tolist()):
        print(f"      {i+1}. '{col}'")
    
    print("\n[2/3] Čišćenje podataka...")
    
    # Pokušaj sa različitim nazivima kolona
    # Prvo provjeri koje ime postoji
    
    # Za Invoice (broj fakture)
    if 'InvoiceNo' in df.columns:
        invoice_col = 'InvoiceNo'
    elif 'Invoice' in df.columns:
        invoice_col = 'Invoice'
    elif 'InvoiceNo.' in df.columns:
        invoice_col = 'InvoiceNo.'
    else:
        print("      UPOZORENJE: Ne mogu pronaći kolonu za broj fakture!")
        invoice_col = None
    
    # Za Customer ID
    if 'CustomerID' in df.columns:
        customer_col = 'CustomerID'
    elif 'Customer ID' in df.columns:
        customer_col = 'Customer ID'
    else:
        customer_col = None
    
    # Za Price
    if 'Price' in df.columns:
        price_col = 'Price'
    elif 'UnitPrice' in df.columns:
        price_col = 'UnitPrice'
    else:
        price_col = None
    
    # Primijeni čišćenje ako su kolone pronađene
    if invoice_col:
        df = df[~df[invoice_col].astype(str).str.startswith('C')]
    
    if customer_col:
        df = df.dropna(subset=[customer_col])
    
    if price_col:
        df = df[(df['Quantity'] > 0) & (df[price_col] > 0)]
    
    print(f"      Preostalo redova: {len(df):,}")
    
    return df, invoice_col, customer_col, price_col


# -------------------------------
# 2. RFM METRIKE
# -------------------------------

def compute_rfm(df, invoice_col, customer_col, price_col):
    """Računa Recency, Frequency i Monetary za svakog kupca i skalira podatke"""
    
    print("[3/3] Računanje RFM metrika...")
    
    # Konvertuj datum
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Referentni datum
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    # Ukupna cijena po stavci
    df['TotalPrice'] = df['Quantity'] * df[price_col]
    
    # Grupisanje po kupcu
    rfm = df.groupby(customer_col).agg(
        Recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        Frequency=(invoice_col, 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()
    
    # Preimenuj kolonu CustomerID
    rfm.rename(columns={customer_col: 'CustomerID'}, inplace=True)
    
    # Izbaci kupce sa Monetary = 0 (ako ih ima)
    rfm = rfm[rfm['Monetary'] > 0]
    
    print(f"      Broj kupaca prije skaliranja: {len(rfm):,}")
    
    # Ispis prije skaliranja (prvih 5 redova)
    print("\nRFM prije skaliranja (prvih 5 redova):")
    print(rfm.head())
    
    # Skaliranje
    scaler = StandardScaler()
    rfm[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(
        rfm[['Recency', 'Frequency', 'Monetary']]
    )
    
    # Ispis poslije skaliranja
    print("\nRFM poslije skaliranja (prvih 5 redova):")
    print(rfm.head())
    
    # Sačuvaj
    rfm.to_csv(RFM_SCALED, index=False)
    print(f"\n      Sačuvano: {RFM_SCALED} (ukupno {len(rfm)} kupaca)")

    scaler_path = os.path.join(PROCESSED_DIR, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"        Scaler sačuvan: {scaler_path}")
    
    return rfm


# -------------------------------
# 3. GLAVNA FUNKCIJA
# -------------------------------

if __name__ == '__main__':
    print("=" * 50)
    print("DATA PREPARATION ZA CLV PROJEKAT")
    print("=" * 50)
    
    # Učitaj i očisti podatke
    df, invoice_col, customer_col, price_col = load_and_clean()
    
    # Provjeri da li su sve kolone pronađene
    print(f"\n[INFO] Korištene kolone:")
    print(f"      - Faktura (Invoice): {invoice_col}")
    print(f"      - Kupac (Customer): {customer_col}")
    print(f"      - Cijena (Price): {price_col}")
    
    # Računaj RFM
    rfm = compute_rfm(df, invoice_col, customer_col, price_col)
    
    print("\n" + "=" * 50)
    print("PRIREMA PODATAKA ZAVRŠENA!")
    print("=" * 50)