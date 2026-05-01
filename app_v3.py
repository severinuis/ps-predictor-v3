import streamlit as st
import joblib
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- CONFIGURATION ---
st.set_page_config(page_title="PPS Predictor V3.2 PRO", page_icon="⚽", layout="wide")

# --- CONNEXION GOOGLE SHEETS ---
try:
    conn = st.connection("gsheets", type=GSheetsConnection)
except Exception as e:
    st.error("Erreur de connexion aux Google Sheets. Vérifiez vos Secrets.")

@st.cache_resource
def load_models():
    try:
        # Assure-toi que les fichiers sont dans le même dossier que app.py
        h = joblib.load('model_v3_home.pkl')
        a = joblib.load('model_v3_away.pkl')
        return h, a
    except Exception as e:
        st.warning(f"Modèles non trouvés : {e}")
        return None, None

m_home, m_away = load_models()

# --- INTERFACE ---
st.title("🏆 PPS Predictor V3.2 : Expert Analysis System")

# Vérification de sécurité pour les modèles
if m_home is None or m_away is None:
    st.error("⚠️ Les modèles IA ne sont pas chargés. Le bouton d'analyse sera désactivé.")

with st.sidebar:
    st.header("📈 Cotes 1N2 (Market)")
    c_dom = st.number_input("Cote Domicile [1]", 1.01, 20.0, 1.85)
    c_nul = st.number_input("Cote Match Nul [N]", 1.01, 20.0, 3.40)
    c_ext = st.number_input("Cote Extérieur [2]", 1.01, 20.0, 4.20)
    p_c_dom, p_c_nul, p_c_ext = 1/c_dom, 1/c_nul, 1/c_ext

col1, col2 = st.columns(2)
with col1:
    st.header("🏠 Équipe Domicile")
    n_dom = st.text_input("Nom de l'équipe", "Gérone")
    r_dom = st.number_input(f"Rating ClubElo ({n_dom})", 1000.0, 2200.0, 1880.0)
    rank_dom = st.number_input(f"Rang ({n_dom})", 1, 20, 3)
    f_dom = st.text_input(f"Forme (V N D) - {n_dom}", "2 1 2")
    bm_dom = st.number_input(f"Buts marqués (5m) - {n_dom}", 0, 25, 8)
    be_dom = st.number_input(f"Buts encaissés (5m) - {n_dom}", 0, 25, 5)

with col2:
    st.header("🚀 Équipe Extérieur")
    n_ext = st.text_input("Nom de l'équipe ", "Majorque")
    r_ext = st.number_input(f"Rating ClubElo ({n_ext})", 1000.0, 2200.0, 1720.0)
    rank_ext = st.number_input(f"Rang ({n_ext})", 1, 20, 15)
    f_ext = st.text_input(f"Forme (V N D) - {n_ext}", "1 2 2")
    bm_ext = st.number_input(f"Buts marqués (5m) - {n_ext}", 0, 25, 4)
    be_ext = st.number_input(f"Buts encaissés (5m) - {n_ext}", 0, 25, 6)

st.markdown("---")
h2h_input = st.text_input(f"Saisir : Victoires {n_dom} / Nuls / Victoires {n_ext}", "1 1 1")

# Bouton de prédiction avec sécurité
if st.button("🚀 LANCER L'ANALYSE PPS V3.2") and m_home is not None:
    try:
        # Calcul des notes avec protection contre les valeurs négatives
        v1, n1, d1 = map(int, f_dom.split())
        v2, n2, d2 = map(int, f_ext.split())

        def calc_n(v, n, bm, be): 
            score = ((v*3+n)/15)*50 + (bm/12.5)*25 + (100-(be/10)*100)*0.25
            return max(0, score)

        note_d = calc_n(v1, n1, bm_dom, be_dom)
        note_e = calc_n(v2, n2, bm_ext, be_ext)
        diff_elo = r_dom - r_ext
        ecart_r = rank_ext - rank_dom
        vh, nh, dh = map(int, h2h_input.split())

        # Prédiction
        features = ['Forme_Dom', 'Forme_Ext', 'Diff_Elo', 'Ecart_Rang', 'H2H_Dom', 'Prob_C_Dom', 'Prob_C_Nul', 'Prob_C_Ext']
        data = pd.DataFrame([[note_d, note_e, diff_elo, ecart_r, (vh*3+nh), p_c_dom, p_c_nul, p_c_ext]], columns=features)

        pred_h = m_home.predict(data)[0]
        pred_a = m_away.predict(data)[0]

        s_h, s_a = int(max(0,round(pred_h))), int(max(0,round(pred_a)))

        st.success(f"🔮 Résultat Prédit : {s_h} - {s_a}")
        st.metric("Tendance", f"{n_dom}" if s_h > s_a else f"{n_ext}" if s_a > s_h else "Nul")

        # Sauvegarde Google Sheets
        new_row = pd.DataFrame([{
            "Date": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
            "Equipe_Dom": n_dom, "Equipe_Ext": n_ext,
            "Score_Predit": f"{s_h}-{s_a}"
        }])

        # Lecture et mise à jour
        old_df = conn.read(worksheet="Pronos")
        updated_df = pd.concat([old_df, new_row], ignore_index=True)
        conn.update(worksheet="Pronos", data=updated_df)

        st.balloons()
    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")
