"""
app.py
Streamlit aplikacija za predikciju CLV segmenta
"""

import streamlit as st
import joblib
import os
import sys

# Dodaj putanju do foldera gdje se nalazi predict.py (npr. 'src/', pored
# train.py) - PRILAGODI naziv foldera 'src' ako se kod kod tebe zove drugačije.
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from predict import group_to_3_clv

# -------------------------------
# PUTANJE
# -------------------------------

SCALER_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'scaler.pkl')
KNN_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'knn_model.pkl')
SEGMENT_MAP_PATH = os.path.join(BASE_DIR, 'results', 'metrics', 'segment_map.pkl')


# -------------------------------
# FUNKCIJE
# -------------------------------

@st.cache_resource
def load_models():
    """Učitava scaler, KNN model i mapiranje segmenata (kešira se)."""
    scaler = joblib.load(SCALER_PATH)
    knn = joblib.load(KNN_MODEL_PATH)
    segment_map = joblib.load(SEGMENT_MAP_PATH)
    return scaler, knn, segment_map


def predict_segment(recency, frequency, monetary, scaler, knn, segment_map):
    """Predviđa CLV segment za novog kupca."""
    X_new = scaler.transform([[recency, frequency, monetary]])
    segment_id = knn.predict(X_new)[0]
    segment_name = segment_map[int(segment_id)]
    return group_to_3_clv(segment_name)


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.set_page_config(
    page_title="CLV Segment Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Lifetime Value Segment Predictor")
st.markdown("Unesite podatke o kupcu da biste predvidjeli njegov CLV segment.")

# Sidebar sa objašnjenjem
with st.sidebar:
    st.header("ℹ️ O aplikaciji")
    st.markdown("""
    Ova aplikacija koristi **KNN klasifikator** za predikciju CLV segmenta kupca.
    
    **Ulazni parametri:**
    - **Recency**: Broj dana od posljednje kupovine
    - **Frequency**: Ukupan broj kupovina
    - **Monetary**: Ukupna potrošnja u £
    
    **Izlaz:**
    - 🔴 **Low CLV** (niska vrijednost)
    - 🟡 **Mid CLV** (srednja vrijednost)
    - 🟢 **High CLV** (visoka vrijednost)
    """)

# Učitaj modele
try:
    scaler, knn, segment_map = load_models()
    models_loaded = True
    st.success("✅ Modeli uspješno učitani")
except Exception as e:
    models_loaded = False
    st.error(f"❌ Greška pri učitavanju modela: {e}")
    st.info("Provjerite da li postoje fajlovi:\n- `data/processed/scaler.pkl`\n- `models/knn_model.pkl`\n- `results/metrics/segment_map.pkl`")

# Forma za unos
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        recency = st.number_input(
            "📅 Recency (dana)",
            min_value=1,
            max_value=730,
            value=1,
            help="Broj dana od posljednje kupovine"
        )
    
    with col2:
        frequency = st.number_input(
            "🔄 Frequency (broj kupovina)",
            min_value=1,
            max_value=500,
            value=1,
            help="Ukupan broj faktura"
        )
    
    with col3:
        monetary = st.number_input(
            "💰 Monetary (£)",
            min_value=0.0,
            max_value=200000.0,
            value=0.0,
            help="Ukupna potrošnja u funtama"
        )
    
    submitted = st.form_submit_button("🔮 Predvidi segment", type="primary")

# Predikcija
if submitted and models_loaded:
    with st.spinner("Predviđanje u toku..."):
        segment = predict_segment(recency, frequency, monetary, scaler, knn, segment_map)
    
    # Prikaz rezultata
    st.divider()
    st.subheader("📈 Rezultat predikcije")
    
    if segment == 'High CLV':
        st.success(f"### 🟢 **{segment}**")
        st.markdown("Ovaj kupac spada u **visoko vrijedan segment**.")
    elif segment == 'Mid CLV':
        st.info(f"### 🟡 **{segment}**")
        st.markdown("Ovaj kupac spada u **srednje vrijedan segment**.")
    else:
        st.warning(f"### 🔴 **{segment}**")
        st.markdown("Ovaj kupac spada u **nisko vrijedan segment**.")