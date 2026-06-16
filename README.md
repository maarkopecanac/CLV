# Customer Lifetime Value (CLV) Segmentacija kupaca

Projekat iz mašinskog učenja za segmentaciju kupaca na osnovu RFM analize i K-Means klasterizacije.

---

## 📌 Opis projekta

Sistem segmentira kupce na osnovu njihovog ponašanja (Recency, Frequency, Monetary) koristeći:
- **K-Means klasterizaciju** za otkrivanje prirodnih grupa kupaca
- **KNN klasifikator** za predikciju segmenta za novog kupca
- **Streamlit aplikaciju** za interaktivnu predikciju

---

### Pokretanje

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