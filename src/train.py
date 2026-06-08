"""
train.py
K-Means klasterizacija za segmentaciju kupaca
"""

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

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
RFM_TRAIN        = os.path.join(PROCESSED_DIR, 'rfm_train.csv')
RFM_TEST         = os.path.join(PROCESSED_DIR, 'rfm_test.csv')
MODEL_PATH       = os.path.join(MODELS_DIR,    'kmeans_model.pkl')
ELBOW_PATH       = os.path.join(FIGURES_DIR,   'elbow_plot.png')
SEGMENT_PROFILES = os.path.join(METRICS_DIR,   'segment_profiles.csv')
SEGMENT_MAP_PATH = os.path.join(METRICS_DIR,   'segment_map.pkl')


# -------------------------------
# POMOĆNA FUNKCIJA: dinamičko mapiranje
# -------------------------------

def build_segment_map(rfm_df):
    """
    Mapira klaster_id -> CLV naziv na osnovu kompozitnog RFM skora.

    Zašto kompozitni skor a ne samo Monetary?
    CLV se definiše kroz sve tri dimenzije:
      - Recency:   manji = bolje (kupac je nedavno aktivan)
      - Frequency: veći  = bolje (kupac kupuje često)
      - Monetary:  veći  = bolje (kupac troši više)

    Pošto su podaci skalirani (StandardScaler), sve tri dimenzije
    su na istoj skali i možemo ih direktno kombinovati.
    Recency invertujemo (množimo sa -1) jer je logika obrnuta.

    Klaster sa najvećim skorom -> High CLV
    Klaster sa najmanjim skorom -> Low CLV
    """
    avg = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()

    # Kompozitni skor: Recency invertovan jer manji znači bolje
    avg['CLV_Score'] = -avg['Recency'] + avg['Frequency'] + avg['Monetary']

    # Sortiraj po skoru rastuce: najmanji skor = Low CLV, najveci = High CLV
    avg_sorted = avg['CLV_Score'].sort_values()

    labels = ['Low CLV', 'Mid CLV', 'High CLV']
    segment_map = {int(cluster_id): label
                   for cluster_id, label in zip(avg_sorted.index, labels)}
    return segment_map


# -------------------------------
# 1. UČITAVANJE PODATAKA
# -------------------------------

def load_data():
    print("[1/5] Učitavanje skaliranih RFM podataka...")
    rfm = pd.read_csv(RFM_SCALED)
    print(f"      Učitano {len(rfm)} kupaca.")
    return rfm


# -------------------------------
# 2. PODJELA PODATAKA (80% train, 20% test)
# -------------------------------

def split_data(rfm):
    """
    Dijeli podatke na train (80%) i test (20%).

    K-Means trenira na 80% i dodjeljuje labele tim podacima.
    Za test skup K-Means samo predviđa (predict), ne trenira (fit).
    KNN zatim trenira na 80% i evaluira se na 20% –
    podacima koje nije vidio, što daje realnu tačnost.
    """
    print("\n[2/5] Podjela podataka (train 80%, test 20%)...")

    X       = rfm[['Recency', 'Frequency', 'Monetary']].values
    indices = rfm.index.values

    X_train, X_test, idx_train, idx_test = train_test_split(
        X, indices, test_size=0.2, random_state=42
    )

    rfm_train = rfm.iloc[idx_train].copy()
    rfm_test  = rfm.iloc[idx_test].copy()

    print(f"      Train: {len(rfm_train)} kupaca (80%)")
    print(f"      Test:  {len(rfm_test)} kupaca (20%)")

    return rfm_train, rfm_test


# -------------------------------
# 3. ELBOW METOD na train podacima
# -------------------------------

def elbow_method(rfm_train, k_min=1, k_max=10):
    """
    Crta Elbow grafik na train podacima.
    Tražimo tačku gdje WCSS prestaje naglo padati – 'lakat' krive.
    """
    print(f"\n[3/5] Elbow metod (WCSS) na train podacima...")
    features = rfm_train[['Recency', 'Frequency', 'Monetary']].values
    wcss = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        km.fit(features)
        wcss.append(km.inertia_)
        print(f"      K = {k}: WCSS = {km.inertia_:,.2f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(k_min, k_max + 1), wcss, marker='o', linewidth=2,
             color='steelblue', markersize=8)
    plt.title('Elbow metod za odabir optimalnog broja klastera (train skup)', fontsize=14)
    plt.xlabel('Broj klastera (K)', fontsize=12)
    plt.ylabel('WCSS (inertia)', fontsize=12)
    plt.xticks(range(k_min, k_max + 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(ELBOW_PATH, dpi=150)
    plt.close()
    print(f"      Elbow graf sačuvan: {ELBOW_PATH}")


# -------------------------------
# 4. TRENIRANJE K-MEANS (samo na train podacima)
# -------------------------------

def train_kmeans(rfm_train, n_clusters=3):
    """
    Trenira K-Means isključivo na train podacima.
    Test podaci simuliraju nove, neviđene kupce.
    """
    print(f"\n[4/5] Treniranje K-Means na train podacima (k = {n_clusters})...")
    features = rfm_train[['Recency', 'Frequency', 'Monetary']].values

    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(features)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)

    print(f"      Model sačuvan: {MODEL_PATH}")
    print(f"      Inertia (WCSS): {kmeans.inertia_:,.2f}")
    return kmeans


# -------------------------------
# 5. DODJELA SEGMENATA
# -------------------------------

def assign_segments(rfm_train, rfm_test, kmeans):
    """
    Dodjeljuje segmente svim kupcima.

    Mapiranje se gradi iz train skupa (kompozitni RFM skor)
    i čuva kao segment_map.pkl za kasniju upotrebu u predict.py.
    """
    print("\n[5/5] Dodjela segmenata...")

    rfm_train = rfm_train.copy()
    rfm_test  = rfm_test.copy()

    rfm_train['Segment'] = kmeans.predict(
        rfm_train[['Recency', 'Frequency', 'Monetary']].values
    )
    rfm_test['Segment'] = kmeans.predict(
        rfm_test[['Recency', 'Frequency', 'Monetary']].values
    )

    # Kompozitni RFM skor za mapiranje
    segment_map = build_segment_map(rfm_train)
    joblib.dump(segment_map, SEGMENT_MAP_PATH)

    print(f"\n      Mapiranje klastera (po kompozitnom RFM skoru):")
    for k, v in segment_map.items():
        print(f"      Klaster {k} -> {v}")

    rfm_train['CLV_Segment'] = rfm_train['Segment'].map(segment_map)
    rfm_test['CLV_Segment']  = rfm_test['Segment'].map(segment_map)
    rfm_all = pd.concat([rfm_train, rfm_test], ignore_index=True)

    # Ispis distribucije
    print("\n      Distribucija segmenata (train skup):")
    for seg, count in rfm_train['CLV_Segment'].value_counts().items():
        print(f"      - {seg}: {count} kupaca ({count/len(rfm_train)*100:.1f}%)")

    print("\n      Distribucija segmenata (test skup):")
    for seg, count in rfm_test['CLV_Segment'].value_counts().items():
        print(f"      - {seg}: {count} kupaca ({count/len(rfm_test)*100:.1f}%)")

    # Profil segmenata
    profile = rfm_all.groupby('CLV_Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
    print("\n      Profil segmenata (skalirane prosječne vrijednosti):")
    print(profile.to_string())

    profile.to_csv(SEGMENT_PROFILES)
    rfm_all.to_csv(RFM_LABELED,  index=False)
    rfm_train.to_csv(RFM_TRAIN,  index=False)
    rfm_test.to_csv(RFM_TEST,    index=False)

    print(f"\n      Sačuvano: {RFM_LABELED}")
    print(f"      Sačuvano: {RFM_TRAIN}")
    print(f"      Sačuvano: {RFM_TEST}")
    print(f"      Sačuvano: {SEGMENT_PROFILES}")

    return rfm_all, rfm_train, rfm_test


# -------------------------------
# 6. GLAVNA FUNKCIJA
# -------------------------------

def run(n_clusters=3):
    print("=" * 60)
    print("K-MEANS KLASTERIZACIJA ZA CLV PROJEKAT")
    print("=" * 60)

    rfm                              = load_data()
    rfm_train, rfm_test              = split_data(rfm)
    elbow_method(rfm_train)
    kmeans                           = train_kmeans(rfm_train, n_clusters)
    rfm_all, rfm_train, rfm_test     = assign_segments(rfm_train, rfm_test, kmeans)

    print("\n" + "=" * 60)
    print(f"KLASTERIZACIJA ZAVRŠENA! (k = {n_clusters})")
    print("=" * 60)
    print("\nNapomena: K-Means je treniran na 80% podataka.")
    print("Preostalih 20% koristi KNN klasifikator za evaluaciju.")

    return rfm_all, kmeans


if __name__ == '__main__':
    run(n_clusters=3)