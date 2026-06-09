"""
evaluate.py
Vizualizacija i analiza segmenata kupaca
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

PROCESSED_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
FIGURES_DIR    = os.path.join(os.path.dirname(__file__), '..', 'results', 'figures')
METRICS_DIR    = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics')

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

RFM_LABELED      = os.path.join(PROCESSED_DIR, 'rfm_labeled.csv')
SEGMENT_MAP_PATH = os.path.join(METRICS_DIR,   'segment_map.pkl')
SEGMENTS_TABLE   = os.path.join(METRICS_DIR,   'segments_table.csv')
BAR_PLOT         = os.path.join(FIGURES_DIR,   'segment_distribution.png')
SCATTER_PLOT     = os.path.join(FIGURES_DIR,   'scatter_recency_frequency.png')


def load_data():
    print("[1/4] Učitavanje podataka...")
    rfm = pd.read_csv(RFM_LABELED)
    print(f"      Učitano {len(rfm)} kupaca.")
    return rfm


def segments_table(rfm):
    print("\n[2/4] Tabela profila segmenata...")
    
    if 'CLV_Segment' not in rfm.columns:
        segment_map = joblib.load(SEGMENT_MAP_PATH)
        rfm['CLV_Segment'] = rfm['Segment'].map(segment_map)
    
    table = rfm.groupby('CLV_Segment').agg(
        Broj_kupaca=('CustomerID', 'count'),
        Recency=('Recency', 'mean'),
        Frequency=('Frequency', 'mean'),
        Monetary=('Monetary', 'mean')
    ).round(2).reset_index()
    
    table['Postotak'] = (table['Broj_kupaca'] / len(rfm) * 100).round(1)
    table = table.sort_values('Monetary').reset_index(drop=True)
    
    print(table.to_string(index=False))
    table.to_csv(SEGMENTS_TABLE, index=False)
    print(f"      Sačuvano: {SEGMENTS_TABLE}")
    return table, rfm


def plot_bar(rfm):
    print("\n[3/4] Bar plot (distribucija kupaca)...")
    
    order = rfm.groupby('CLV_Segment')['Monetary'].mean().sort_values().index.tolist()
    counts = rfm['CLV_Segment'].value_counts().reindex(order)
    
    n = len(order)
    colormap = plt.cm.tab10
    colors = [colormap(i / n) for i in range(n)]
    
    plt.figure(figsize=(9, 5))
    bars = plt.bar(counts.index, counts.values, color=colors, edgecolor='white', width=0.6)
    
    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.title('Broj kupaca po CLV segmentu', fontsize=14)
    plt.xlabel('CLV Segment', fontsize=12)
    plt.ylabel('Broj kupaca', fontsize=12)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(BAR_PLOT, dpi=150)
    plt.close()
    print(f"      Sačuvano: {BAR_PLOT}")


def plot_scatter(rfm):
    print("\n[4/4] Scatter plot (Recency vs Frequency)...")
    
    order = rfm.groupby('CLV_Segment')['Monetary'].mean().sort_values().index.tolist()
    n = len(order)
    colormap = plt.cm.tab10
    colors = {segment: colormap(i / n) for i, segment in enumerate(order)}
    
    plt.figure(figsize=(10, 6))
    for segment, group in rfm.groupby('CLV_Segment'):
        plt.scatter(group['Recency'], group['Frequency'],
                    c=[colors[segment]], label=segment, alpha=0.6, s=25)
    
    plt.title('Raspored kupaca – Recency vs Frequency', fontsize=14)
    plt.xlabel('Recency (skalirano)', fontsize=12)
    plt.ylabel('Frequency (skalirano)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(SCATTER_PLOT, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Sačuvano: {SCATTER_PLOT}")


def run():
    print("=" * 55)
    print("  EVALUACIJA CLV PROJEKTA")
    print("=" * 55)
    rfm = load_data()
    table, rfm = segments_table(rfm)
    plot_bar(rfm)
    plot_scatter(rfm)
    print("\n" + "=" * 55)
    print("  EVALUACIJA ZAVRŠENA!")
    print("=" * 55)


if __name__ == '__main__':
    run()