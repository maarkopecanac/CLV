"""
train.py
K-Means klasterizacija za segmentaciju kupaca
(skaliranje nakon podjele podataka)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------------
# PUTANJE
# -------------------------------

PROCESSED_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
MODELS_DIR       = os.path.join(os.path.dirname(__file__), '..', 'models')
FIGURES_DIR      = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
METRICS_DIR      = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RFM_CLEAN        = os.path.join(PROCESSED_DIR, 'rfm_clean.csv')
RFM_TRAIN        = os.path.join(PROCESSED_DIR, 'rfm_train.csv')
RFM_TEST         = os.path.join(PROCESSED_DIR, 'rfm_test.csv')
RFM_LABELED      = os.path.join(PROCESSED_DIR, 'rfm_labeled.csv')
MODEL_PATH       = os.path.join(MODELS_DIR,    'kmeans_model.pkl')
SCALER_PATH      = os.path.join(PROCESSED_DIR, 'scaler.pkl')
ELBOW_PATH       = os.path.join(FIGURES_DIR,   'elbow_plot.png')
SEGMENT_PROFILES = os.path.join(METRICS_DIR,   'segment_profiles.csv')
SEGMENT_MAP_PATH = os.path.join(METRICS_DIR,   'segment_map.pkl')


# -------------------------------
# POMOĆNE FUNKCIJE
# -------------------------------

def build_segment_map(rfm_df):
    """Mapira klaster_id -> CLV naziv na osnovu kompozitnog RFM skora."""
    avg = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    avg['CLV_Score'] = -avg['Recency'] + avg['Frequency'] + avg['Monetary']
    avg_sorted = avg['CLV_Score'].sort_values()
    
    n = len(avg_sorted)
    
    # Dinamički nazivi – bez hardkodiranja!
    if n == 3:
        labels = ['Low CLV', 'Mid CLV', 'High CLV']
    elif n == 4:
        labels = ['Low CLV', 'Mid-Low CLV', 'Mid-High CLV', 'High CLV']
    elif n == 5:
        labels = ['Very Low CLV', 'Low CLV', 'Mid CLV', 'High CLV', 'Very High CLV']
    else:
        labels = [f'CLV_{i+1}' for i in range(n)]
    
    return {int(cluster_id): label for cluster_id, label in zip(avg_sorted.index, labels)}


def find_optimal_k(wcss, k_min=2, k_max=10):
    """Pronalazi 'lakat' – optimalan broj klastera."""
    if len(wcss) < 3:
        return 3
    diffs = [abs(wcss[i] - wcss[i+1]) for i in range(len(wcss)-1)]
    max_diff = max(diffs)
    if max_diff > 0:
        diffs_norm = [d / max_diff for d in diffs]
    else:
        diffs_norm = diffs
    for i, diff in enumerate(diffs_norm):
        if diff < 0.2:
            return i + k_min + 1
    return 3


# -------------------------------
# 1. UČITAVANJE PODATAKA
# -------------------------------

def load_data():
    print("[1/6] Učitavanje čistih RFM podataka...")
    rfm = pd.read_csv(RFM_CLEAN)
    print(f"      Učitano {len(rfm)} kupaca.")
    return rfm


# -------------------------------
# 2. PODJELA PODATAKA
# -------------------------------

def split_data(rfm):
    """Dijeli podatke na train (80%) i test (20%)."""
    print("\n[2/6] Podjela podataka (train 80%, test 20%)...")
    
    rfm_train, rfm_test = train_test_split(
        rfm, test_size=0.2, random_state=42
    )
    
    print(f"      Train: {len(rfm_train)} kupaca (80%)")
    print(f"      Test:  {len(rfm_test)} kupaca (20%)")
    
    return rfm_train, rfm_test


# -------------------------------
# 3. SKALIRANJE (samo na train)
# -------------------------------

def scale_data(rfm_train, rfm_test):
    """Skalira podatke: fit na train, transform na train i test."""
    print("\n[3/6] Skaliranje podataka (fit na train, transform na test)...")
    
    scaler = StandardScaler()
    
    rfm_train_scaled = rfm_train.copy()
    rfm_train_scaled[['Recency', 'Frequency', 'Monetary']] = scaler.fit_transform(
        rfm_train[['Recency', 'Frequency', 'Monetary']]
    )
    
    rfm_test_scaled = rfm_test.copy()
    rfm_test_scaled[['Recency', 'Frequency', 'Monetary']] = scaler.transform(
        rfm_test[['Recency', 'Frequency', 'Monetary']]
    )
    
    joblib.dump(scaler, SCALER_PATH)
    print(f"      Scaler sačuvan: {SCALER_PATH}")
    
    return rfm_train_scaled, rfm_test_scaled


# -------------------------------
# 4. ELBOW METOD
# -------------------------------

def elbow_method(rfm_train_scaled, k_min=2, k_max=10):
    """Crta Elbow grafik na train podacima."""
    print(f"\n[4/6] Elbow metod (WCSS) na train podacima...")
    features = rfm_train_scaled[['Recency', 'Frequency', 'Monetary']].values
    wcss = []
    
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(features)
        wcss.append(km.inertia_)
        print(f"      K = {k}: WCSS = {km.inertia_:,.2f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(k_min, k_max + 1), wcss, marker='o', linewidth=2, color='steelblue')
    plt.title('Elbow metod za odabir optimalnog broja klastera', fontsize=14)
    plt.xlabel('Broj klastera (K)', fontsize=12)
    plt.ylabel('WCSS (inertia)', fontsize=12)
    plt.xticks(range(k_min, k_max + 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=150)
    plt.close()
    print(f"      Elbow graf sačuvan: {ELBOW_PATH}")
    
    return wcss


# -------------------------------
# 5. TRENIRANJE K-MEANS
# -------------------------------

def train_kmeans(rfm_train_scaled, n_clusters):
    """Trenira K-Means na train podacima."""
    print(f"\n[5/6] Treniranje K-Means na train podacima (k = {n_clusters})...")
    features = rfm_train_scaled[['Recency', 'Frequency', 'Monetary']].values
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(features)
    
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    
    print(f"      Model sačuvan: {MODEL_PATH}")
    print(f"      Inertia (WCSS): {kmeans.inertia_:,.2f}")
    return kmeans


# -------------------------------
# 6. DODJELA SEGMENATA
# -------------------------------

def assign_segments(rfm_train_scaled, rfm_test_scaled, kmeans):
    """Dodjeljuje segmente i čuva rezultate."""
    print("\n[6/6] Dodjela segmenata...")
    
    rfm_train_scaled = rfm_train_scaled.copy()
    rfm_test_scaled = rfm_test_scaled.copy()
    
    rfm_train_scaled['Segment'] = kmeans.predict(
        rfm_train_scaled[['Recency', 'Frequency', 'Monetary']].values
    )
    rfm_test_scaled['Segment'] = kmeans.predict(
        rfm_test_scaled[['Recency', 'Frequency', 'Monetary']].values
    )
    
    segment_map = build_segment_map(rfm_train_scaled)
    joblib.dump(segment_map, SEGMENT_MAP_PATH)
    
    print("\n      Mapiranje klastera:")
    for k, v in segment_map.items():
        print(f"      Klaster {k} -> {v}")
    
    rfm_train_scaled['CLV_Segment'] = rfm_train_scaled['Segment'].map(segment_map)
    rfm_test_scaled['CLV_Segment'] = rfm_test_scaled['Segment'].map(segment_map)
    
    rfm_all = pd.concat([rfm_train_scaled, rfm_test_scaled], ignore_index=True)
    
    profile = rfm_all.groupby('CLV_Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    print("\n      Profil segmenata (skalirane vrijednosti):")
    print(profile.to_string())
    profile.to_csv(SEGMENT_PROFILES)
    
    rfm_all.to_csv(RFM_LABELED, index=False)
    rfm_train_scaled.to_csv(RFM_TRAIN, index=False)
    rfm_test_scaled.to_csv(RFM_TEST, index=False)
    
    print(f"\n      Sačuvano: {RFM_LABELED}")
    print(f"      Sačuvano: {RFM_TRAIN}")
    print(f"      Sačuvano: {RFM_TEST}")
    
    return rfm_all


# -------------------------------
# 7. GLAVNA FUNKCIJA
# -------------------------------

def run():
    print("=" * 60)
    print("K-MEANS KLASTERIZACIJA")
    print("=" * 60)
    
    rfm = load_data()
    rfm_train, rfm_test = split_data(rfm)
    rfm_train_scaled, rfm_test_scaled = scale_data(rfm_train, rfm_test)
    wcss = elbow_method(rfm_train_scaled)
    optimal_k = find_optimal_k(wcss)
    print(f"\n[INFO] Automatski odabrani K = {optimal_k}")
    kmeans = train_kmeans(rfm_train_scaled, optimal_k)
    rfm_all = assign_segments(rfm_train_scaled, rfm_test_scaled, kmeans)
    
    print("\n" + "=" * 60)
    print(f"KLASTERIZACIJA ZAVRŠENA! (k = {optimal_k})")
    print("=" * 60)
    
    return rfm_all, kmeans


if __name__ == '__main__':
    run()