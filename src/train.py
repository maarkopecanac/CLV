"""
train.py
K-Means klasterizacija za segmentaciju kupaca
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.cluster import KMeans

# -------------------------------
# PUTANJE
# -------------------------------

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
METRICS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RFM_SCALED = os.path.join(PROCESSED_DIR, 'rfm_scaled.csv')
RFM_LABELED = os.path.join(PROCESSED_DIR, 'rfm_labeled.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'kmeans_model.pkl')
ELBOW_PATH = os.path.join(FIGURES_DIR, 'elbow_plot.png')
SEGMENT_PROFILES = os.path.join(METRICS_DIR, 'segment_profiles.csv')


# -------------------------------
# 1. UCITAVANJE PODATAKA
# -------------------------------

def load_data():
    """Učitava skalirane RFM podatke."""
    print("[1/4] Učitavanje podataka...")
    rfm_scaled = pd.read_csv(RFM_SCALED)
    print(f"      Učitano {len(rfm_scaled)} kupaca.")
    print(f"      Kolone: {rfm_scaled.columns.tolist()}")
    return rfm_scaled


# -------------------------------
# 2. ELBOW METOD (WCSS)
# -------------------------------

def elbow_method(rfm_scaled: pd.DataFrame, k_min: int = 1, k_max: int = 10):
    """Crta Elbow grafik (WCSS) i vraća listu WCSS vrijednosti."""
    print(f"\n[2/4] Elbow metod (WCSS) za K = {k_min} do {k_max}...")
    features = rfm_scaled[['Recency', 'Frequency', 'Monetary']].values
    wcss = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(features)
        wcss.append(km.inertia_)
        print(f"      K = {k}: WCSS = {km.inertia_:,.2f}")

    # Elbow graf
    k_values = range(k_min, k_max + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, wcss, marker='o', linewidth=2, color='steelblue', markersize=8)
    plt.title('Elbow metod za odabir optimalnog broja klastera', fontsize=14)
    plt.xlabel('Broj klastera (K)', fontsize=12)
    plt.ylabel('WCSS (inertia)', fontsize=12)
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=150)
    plt.close()
    print(f"      Elbow graf sačuvan: {ELBOW_PATH}")

    return wcss


# -------------------------------
# 3. TRENIRANJE K-MEANS MODELA
# -------------------------------

def train_kmeans(rfm_scaled: pd.DataFrame, n_clusters: int):
    """Trenira K-Means model sa zadatim brojem klastera."""
    print(f"\n[3/4] Primjena K-Means sa K = {n_clusters}...")
    features = rfm_scaled[['Recency', 'Frequency', 'Monetary']].values

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        random_state=42
    )
    kmeans.fit(features)

    # Cuvanje modela
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)

    print(f"      Model sačuvan: {MODEL_PATH}")
    print(f"      Inertia (WCSS): {kmeans.inertia_:,.2f}")
    return kmeans


# -------------------------------
# 4. DODJELA SEGMENATA I CUVANJE
# -------------------------------

def assign_and_save(rfm_scaled: pd.DataFrame, kmeans):
    """Dodjeljuje segmente kupcima i čuva rezultate."""
    print("\n[4/4] Cuvanje rezultata...")
    features = rfm_scaled[['Recency', 'Frequency', 'Monetary']].values
    rfm_scaled = rfm_scaled.copy()
    rfm_scaled['Segment'] = kmeans.predict(features)

    # Ispis distribucije segmenata
    print("\n      Distribucija segmenata:")
    segment_counts = rfm_scaled['Segment'].value_counts().sort_index()
    for seg, count in segment_counts.items():
        print(f"      - Segment {seg}: {count} kupaca ({count/len(rfm_scaled)*100:.1f}%)")

    # Racunanje profila segmenata (prosjeci)
    print("\n      Profil segmenata (prosjeci Recency, Frequency, Monetary):")
    profile = rfm_scaled.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    print(profile.to_string())

    # Cuvanje profila
    profile.to_csv(SEGMENT_PROFILES)
    print(f"\n      Profili segmenata sačuvani: {SEGMENT_PROFILES}")

    # Cuvanje kompletnih podataka sa segmentima
    rfm_scaled.to_csv(RFM_LABELED, index=False)
    print(f"      Podaci sa segmentima sačuvani: {RFM_LABELED}")

    return rfm_scaled


# -------------------------------
# 5. GLAVNA FUNKCIJA
# -------------------------------

def run(n_clusters: int = 3):
    """
    Glavna funkcija.
    n_clusters - broj klastera
    """
    print("=" * 60)
    print("K-MEANS KLASTERIZACIJA ZA CLV PROJEKAT")
    print("=" * 60)

    # Ucitaj podatke
    rfm_scaled = load_data()

    # Elbow metod
    elbow_method(rfm_scaled, k_min=1, k_max=10)

    # Treniraj K-Means
    kmeans = train_kmeans(rfm_scaled, n_clusters=n_clusters)

    # Dodijeli segmente i sacuvaj
    result = assign_and_save(rfm_scaled, kmeans)

    print("\n" + "=" * 60)
    print(f"KLASTERIZACIJA ZAVRSENA! (k = {n_clusters})")
    print("=" * 60)

    return result, kmeans


if __name__ == '__main__':
    # Pokreni sa 3 klastera
    run(n_clusters=3)