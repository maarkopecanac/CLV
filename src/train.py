"""
train.py
K-Means klasterizacija za segmentaciju kupaca (100% podataka)
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib          # <-- DODATO
import os
from sklearn.cluster import KMeans

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

RFM_SCALED       = os.path.join(PROCESSED_DIR, 'rfm_scaled.csv')
RFM_LABELED      = os.path.join(PROCESSED_DIR, 'rfm_labeled.csv')
MODEL_PATH       = os.path.join(MODELS_DIR,    'kmeans_model.pkl')
ELBOW_PATH       = os.path.join(FIGURES_DIR,   'elbow_plot.png')
SEGMENT_PROFILES = os.path.join(METRICS_DIR,   'segment_profiles.csv')
SEGMENT_MAP_PATH = os.path.join(METRICS_DIR,   'segment_map.pkl')

# -------------------------------
# POMOĆNE FUNKCIJE
# -------------------------------

def build_segment_map(rfm_df):
    """Mapira klaster_id -> Low/Mid/High CLV na osnovu prosječne Monetary."""
    avg_monetary = rfm_df.groupby('Segment')['Monetary'].mean().sort_values()
    labels = ['Low CLV', 'Mid CLV', 'High CLV']
    segment_map = {int(cluster_id): label
                   for cluster_id, label in zip(avg_monetary.index, labels)}
    return segment_map

def load_data():
    print("[1/4] Učitavanje skaliranih RFM podataka...")
    rfm = pd.read_csv(RFM_SCALED)
    print(f"      Učitano {len(rfm)} kupaca.")
    return rfm

def elbow_method(rfm, k_min=1, k_max=10):
    print(f"\n[2/4] Elbow metod (WCSS) na svim podacima...")
    features = rfm[['Recency', 'Frequency', 'Monetary']].values
    wcss = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(features)
        wcss.append(km.inertia_)
        print(f"      K = {k}: WCSS = {km.inertia_:,.2f}")

    plt.figure(figsize=(8,5))
    plt.plot(range(k_min, k_max+1), wcss, marker='o', linewidth=2, color='steelblue')
    plt.title('Elbow metod za odabir K (svi kupci)', fontsize=14)
    plt.xlabel('Broj klastera (K)')
    plt.ylabel('WCSS (inertia)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=150)
    plt.close()
    print(f"      Elbow graf sačuvan: {ELBOW_PATH}")

def train_kmeans(rfm, n_clusters=3):
    print(f"\n[3/4] Treniranje K-Means na svim podacima (k = {n_clusters})...")
    features = rfm[['Recency', 'Frequency', 'Monetary']].values
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(features)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"      Model sačuvan: {MODEL_PATH}")
    print(f"      Inertia: {kmeans.inertia_:,.2f}")
    return kmeans

def assign_segments(rfm, kmeans):
    print("\n[4/4] Dodjela segmenata i čuvanje rezultata...")
    rfm = rfm.copy()
    rfm['Segment'] = kmeans.predict(rfm[['Recency', 'Frequency', 'Monetary']].values)
    
    segment_map = build_segment_map(rfm)
    joblib.dump(segment_map, SEGMENT_MAP_PATH)   # <-- čuva mapiranje
    print(f"\n      Mapiranje klastera: {segment_map}")
    
    rfm['CLV_Segment'] = rfm['Segment'].map(segment_map)
    
    # Profil segmenata
    profile = rfm.groupby('CLV_Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    print("\n      Profil segmenata (skalirane vrijednosti):")
    print(profile)
    profile.to_csv(SEGMENT_PROFILES)
    
    rfm.to_csv(RFM_LABELED, index=False)
    print(f"\n      Podaci sa segmentima sačuvani: {RFM_LABELED}")
    print(f"      Profili segmenata: {SEGMENT_PROFILES}")
    return rfm

def run(n_clusters=3):
    print("=" * 60)
    print("K-MEANS KLASTERIZACIJA (100% PODATAKA)")
    print("=" * 60)
    rfm = load_data()
    elbow_method(rfm, k_min=1, k_max=10)
    kmeans = train_kmeans(rfm, n_clusters)
    rfm_labeled = assign_segments(rfm, kmeans)
    print("\n" + "=" * 60)
    print("KLASTERIZACIJA ZAVRŠENA!")
    print("=" * 60)
    return rfm_labeled, kmeans

if __name__ == '__main__':
    run(n_clusters=3)