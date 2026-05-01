import streamlit as st
import joblib
import pandas as pd
import numpy as np
from streamlit_gsheets import GSheetsConnection

# --- CONFIGURATION ---
st.set_page_config(page_title="PPS Predictor V3.2 PRO", page_icon="⚽", layout="wide")

# --- PARAMÈTRES DE CONNEXION ---
# Ton URL de partage Google Sheet
URL_SHEET = "https://docs.google.com/spreadsheets/d/1ZNpORNMR2sNMy4_DlmSl4z2DuD0UeFfii2gzREytyIk/edit?usp=sharing"

# --- CHARGEMENT DES MODÈLES ---
@st.cache_resource
def load_models():
    try:
        m_h = joblib.load('model_v3_home.pkl')
        m_a = joblib.load('model_v3_away.pkl')
        return m_h, m_a
    except:
        return None, None

m_home, m_away = load_models()

# --- CONNEXION GOOGLE SHEETS ---
conn = st.connection("gsheets", type=GSheetsConnection)

# --- INTERFACE UTILISATEUR ---
st.title("🏆 PPS Predictor V3.2 : Expert Analysis System")
st.markdown("Système de prédiction intelligent pour **Precision Play Stats**")

if m_home is None:
    st.error("⚠️ Modèles IA introuvables. Lancez d'abord le script d'entraînement (train_v3.py).")

with st.sidebar:
    st.header("📈 Cotes 1N2 (Market)")
    c_dom = st.number_input("Cote Domicile [1]", 1.01, 20.0, 1.85)
    c_nul = st.number_input("Cote Match Nul [N]", 1.01, 20.0, 3.40)
    c_ext = st.number_input("Cote Extérieur [2]", 1.01, 20.0, 4.20)

    # Calcul des probabilités implicites
    p_c_dom, p_c_nul, p_c_ext = 1/c_dom, 1/c_nul, 1/c_ext

col1, col2 = st.columns(2)

with col1:
    st.header("🏠 Équipe Domicile")
    n_dom = st.text_input("Nom de l'équipe", "Gérone")
    r_dom = st.number_input(f"Rating ClubElo ({n_dom})", 1000.0, 2200.0, 1880.0)
    rank_dom = st.number_input(f"Rang Classement ({n_dom})", 1, 20, 3)
    f_dom = st.text_input(f"Forme (V N D) - {n_dom}", "2 1 2")
    bm_dom = st.number_input(f"Buts marqués (5m) - {n_dom}", 0, 25, 8)
    be_dom = st.number_input(f"Buts encaissés (5m) - {n_dom}", 0, 25, 5)

with col2:
    st.header("🚀 Équipe Extérieur")
    n_ext = st.text_input("Nom de l'équipe ", "Majorque")
    r_ext = st.number_input(f"Rating ClubElo ({n_ext})", 1000.0, 2200.0, 1720.0)
    rank_ext = st.number_input(f"Rang Classement ({n_ext})", 1, 20, 15)
    f_ext = st.text_input(f"Forme (V N D) - {n_ext}", "1 2 2")
    bm_ext = st.number_input(f"Buts marqués (5m) - {n_ext}", 0, 25, 4)
    be_ext = st.number_input(f"Buts encaissés (5m) - {n_ext}", 0, 25, 6)

st.markdown("---")
st.subheader(f"⚔️ Face-à-Face : {n_dom} vs {n_ext}")
h2h_input = st.text_input(f"Saisir : Victoires {n_dom} / Nuls / Victoires {n_ext}", "1 1 1")

# --- LOGIQUE DE CALCUL ET PRÉDICTION ---
if st.button("🚀 LANCER L'ANALYSE PPS V3.2") and m_home is not None:
    try:
        # 1. Traitement des données de forme
        v1, n1, d1 = map(int, f_dom.split())
        v2, n2, d2 = map(int, f_ext.split())
        vh, nh, dh = map(int, h2h_input.split())

        # Fonction de calcul de note SKYT affinée
        def calc_note(v, n, bm, be):
            note = ((v*3 + n)/15)*50 + (bm/12.5)*25 + (100 - (be/10)*100)*0.25
            return max(0, note)

        note_d = calc_note(v1, n1, bm_dom, be_dom)
        note_e = calc_note(v2, n2, bm_ext, be_ext)
        diff_elo = r_dom - r_ext
        ecart_r = rank_ext - rank_dom
        h2h_pts = (vh * 3 + nh)

        # 2. Préparation pour l'IA
        features = ['Forme_Dom', 'Forme_Ext', 'Diff_Elo', 'Ecart_Rang', 'H2H_Dom', 'Prob_C_Dom', 'Prob_C_Nul', 'Prob_C_Ext']
        input_data = pd.DataFrame([[note_d, note_e, diff_elo, ecart_r, h2h_pts, p_c_dom, p_c_nul, p_c_ext]], columns=features)

        # 3. Prédiction
        p_home = m_home.predict(input_data)[0]
        p_away = m_away.predict(input_data)[0]

        s_h, s_a = int(max(0, round(p_home))), int(max(0, round(p_away)))

        # 4. Affichage des résultats
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Score {n_dom}", s_h)
        c2.metric("État du Match", "MATCH NUL" if s_h == s_a else f"VICTOIRE {n_dom}" if s_h > s_a else f"VICTOIRE {n_ext}")
        c3.metric(f"Score {n_ext}", s_a)

        st.success(f"🔮 Analyse terminée : Pronostic PPS suggéré {s_h} - {s_a}")

        # 5. Sauvegarde sur Google Sheets
        new_entry = pd.DataFrame([{
            "Date": pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
            "Equipe_Dom": n_dom,
            "Equipe_Ext": n_ext,
            "Score_Predit": f"{s_h}-{s_a}",
            "Diff_Elo": diff_elo,
            "Note_PPS": round((note_d + note_e)/2, 2)
        }])

        # Lecture de l'existant et mise à jour
        existing_df = conn.read(spreadsheet=URL_SHEET, worksheet="Pronos")
        updated_df = pd.concat([existing_df, new_entry], ignore_index=True)
        conn.update(spreadsheet=URL_SHEET, worksheet="Pronos", data=updated_df)

        st.toast("Données sauvegardées dans Google Sheets !", icon="✅")
        st.balloons()

    except Exception as e:
        st.error(f"Erreur lors de l'analyse : {e}")

# --- SECTION HISTORIQUE ---
st.markdown("---")
st.subheader("📊 Derniers Pronostics Enregistrés")
try:
    df_history = conn.read(spreadsheet=URL_SHEET, worksheet="Pronos")
    st.dataframe(df_history.tail(5), use_container_width=True)
except:
    st.info("L'historique s'affichera ici après votre première analyse.")
