# Customer Lifetime Value (CLV) Segmentacija kupaca

Projekat iz mašinskog učenja za segmentaciju kupaca na osnovu RFM analize i K-Means klasterizacije.

---

## 📌 Opis projekta

Sistem segmentira kupce na osnovu njihovog ponašanja (Recency, Frequency, Monetary) koristeći:
- **K-Means klasterizaciju** za otkrivanje prirodnih grupa kupaca
- **KNN klasifikator** za predikciju segmenta za novog kupca
- **Streamlit aplikaciju** za interaktivnu predikciju

---

## 📁 Struktura projekta
CLV/
├── data/
│ ├── raw/ # sirovi podaci (Online Retail II)
│ └── processed/ # očišćeni i skalirani podaci
├── src/
│ ├── data_preparation.py # učitavanje, čišćenje, RFM
│ ├── train.py # K-Means klasterizacija
│ ├── evaluate.py # evaluacija i vizualizacija
│ └── predict.py # KNN klasifikator
├── models/ # sačuvani modeli (.pkl)
├── results/
│ ├── figures/ # grafovi (elbow, scatter, bar)
│ └── metrics/ # tabele i izvještaji (.csv, .xlsx)
├── app/ # Streamlit aplikacija
├── requirements.txt
└── README.md

Pokretanje

### 1. Kloniraj repozitorijum

```bash
git clone https://github.com/maarkopecanac/CLV.git
cd CLV

python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash

pip install -r requirements.txt

# Priprema podataka (RFM)
python src/data_preparation.py

# K-Means klasterizacija (automatski odabir K)
python src/train.py

# KNN klasifikator (trening + evaluacija)
python src/predict.py

# Evaluacija i vizualizacija
python src/evaluate.py

# Pokretanje Streamlit aplikacije
streamlit run app/app.py