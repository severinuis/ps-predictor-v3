import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. CONFIGURATION ET CHARGEMENT
st.set_page_config(page_title="PPS Predictor V3", page_icon="🏆", layout="wide")

@st.cache_resource
def load_models():
    # Assure-toi que ces fichiers existent via le script d'entraînement V3
    m_home = joblib.load('model_v3_home.pkl')
    m_away = joblib.load('model_v3_away.pkl')
    return m_home, m_away

try:
    model_home, model_away = load_models()
except:
    st.error("⚠️ Modèles V3 (.pkl) introuvables. Lancez d'abord l'entraînement.")
    st.stop()

# --- FONCTIONS DE CALCUL ---
def get_note_forme(v, n, d, bm, be):
    pts = ((v * 3 + n * 1) / 15) * 100
    att = (bm / 12.5) * 100
    defe = 100 - ((be / 10) * 100)
    return max(0, min(100, (pts * 0.5 + att * 0.25 + defe * 0.25)))

# --- INTERFACE UTILISATEUR ---
st.title("⚽ PPS Predictor V3 : Expert System")
st.markdown("Combinaison : **Stats Saison + Forme Récente + Classement + H2H**")

col1, col2 = st.columns(2)

with col1:
    st.header("🏠 Équipe Domicile")
    nom_dom = st.text_input("Nom de l'équipe", "Gérone")
    rang_dom = st.number_input(f"Rang Classement ({nom_dom})", 1, 20, 3)

    st.subheader("🔥 Forme (5 derniers matchs)")
    res_dom = st.text_input(f"Résultats {nom_dom} (ex: 2 1 2 pour 2V 1N 2D)", "2 1 2")
    bm_f_dom = st.number_input(f"Buts marqués (5 m.) - {nom_dom}", 0, 25, 8)
    be_f_dom = st.number_input(f"Buts encaissés (5 m.) - {nom_dom}", 0, 25, 5)

with col2:
    st.header("🚀 Équipe Extérieur")
    nom_ext = st.text_input("Nom de l'équipe ", "Majorque")
    rang_ext = st.number_input(f"Rang Classement ({nom_ext})", 1, 20, 15)

    st.subheader("🔥 Forme (5 derniers matchs)")
    res_ext = st.text_input(f"Résultats {nom_ext} (ex: 1 2 2 pour 1V 2N 2D)", "1 2 2")
    bm_f_ext = st.number_input(f"Buts marqués (5 m.) - {nom_ext}", 0, 25, 4)
    be_f_ext = st.number_input(f"Buts encaissés (5 m.) - {nom_ext}", 0, 25, 6)

st.markdown("---")
st.header("⚔️ Historique Face-à-Face (3 derniers matchs)")
c1, c2, c3 = st.columns(3)
v_h = c1.number_input(f"Victoires de {nom_dom}", 0, 3, 1)
n_h = c2.number_input("Matchs Nuls", 0, 3, 1)
d_h = c3.number_input(f"Victoires de {nom_ext}", 0, 3, 1)

if st.button("🚀 LANCER L'ANALYSE FINALE V3"):
    # 1. Traitement des données de forme
    v1, n1, d1 = map(int, res_dom.split())
    v2, n2, d2 = map(int, res_ext.split())

    note_dom = get_note_forme(v1, n1, d1, bm_f_dom, be_f_dom)
    note_ext = get_note_forme(v2, n2, d2, bm_f_ext, be_f_ext)

    # 2. Calcul des variables V3
    ecart_rang = rang_ext - rang_dom # Plus c'est haut, plus Dom est favori
    pts_h2h = (v_h * 3) + n_h        # Points pris par Dom contre Ext

    # 3. Prédiction via les modèles .pkl
    # L'ordre doit correspondre à ton script d'entraînement : Forme_Dom, Forme_Ext, Ecart_Rang, H2H_Dom
    input_data = pd.DataFrame([[note_dom, note_ext, ecart_rang, pts_h2h]], 
                              columns=['Forme_Dom', 'Forme_Ext', 'Ecart_Rang', 'H2H_Dom'])

    pred_h = model_home.predict(input_data)[0]
    pred_a = model_away.predict(input_data)[0]

    # 4. Calcul des Probabilités (Softmax)
    diff = pred_h - pred_a
    p_dom = (1 / (1 + np.exp(-diff))) * 100
    p_nul = max(5, 25 - abs(diff * 8))
    facteur = (100 - p_nul) / 100
    p_dom_f, p_ext_f = p_dom * facteur, (100 - p_dom) * facteur

    # 5. AFFICHAGE DES RÉSULTATS
    st.markdown("---")
    s_h, s_a = int(max(0, round(pred_h))), int(max(0, round(pred_a)))

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader(f"🔮 Score Prédit : {s_h} - {s_a}")
        if s_h > s_a: st.success(f"Conseil : Victoire {nom_dom}")
        elif s_a > s_h: st.success(f"Conseil : Victoire {nom_ext}")
        else: st.info("Conseil : Match Nul")

    with res_col2:
        st.write("**Probabilités PPS :**")
        st.progress(int(p_dom_f), text=f"🏠 {nom_dom} : {p_dom_f:.1f}%")
        st.progress(int(p_nul), text=f"🤝 Match Nul : {p_nul:.1f}%")
        st.progress(int(p_ext_f), text=f"🚀 {nom_ext} : {p_ext_f:.1f}%")

    st.info(f"💡 Note de forme calculée : {nom_dom} ({note_dom:.1f}) | {nom_ext} ({note_ext:.1f})")
