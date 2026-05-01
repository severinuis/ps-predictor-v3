"""
app_v3.py — PPS Predictor V3.2 PRO
------------------------------------
• Format cotes 1 / N / 2 avec normalisation de la marge bookmaker
• Connexion Google Sheets sécurisée (conn jamais None silencieux)
• Scaler chargé et appliqué avant prédiction
• H2H_Dom cohérent avec train_v3.py (V*3 + N)
• Parsing de forme robuste (regex)
• Probabilités 1N2 calculées et affichées
• Distribution de Poisson pour les probabilités de score exact
• calc_note_forme() bornée à [0, 100]
• Sauvegarde GSheets protégée
"""

import re
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from itertools import product

# ── Connexion Google Sheets (optionnel) ──────────────────────────────────────
try:
    from streamlit_gsheets import GSheetsConnection
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

# ============================================================
# CONFIGURATION PAGE
# ============================================================
st.set_page_config(
    page_title="PPS Predictor V3.2 PRO",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# STYLE
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(135deg, #00c853, #1565c0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    .score-box {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 900;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        background: linear-gradient(135deg, #e8f5e9, #e3f2fd);
        border: 2px solid #00c853;
        margin: 1rem 0;
    }
    .prob-bar-label { font-size: 0.8rem; color: #555; }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        border-left: 4px solid #00c853;
        padding-left: 0.5rem;
        margin: 1rem 0 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CONNEXION GOOGLE SHEETS
# ============================================================
conn = None
if GSHEETS_AVAILABLE:
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Google Sheets non connecté : {e}")

# ============================================================
# CHARGEMENT DES MODÈLES
# ============================================================
@st.cache_resource
def load_models():
    try:
        m_h     = joblib.load('model_v3_home.pkl')
        m_a     = joblib.load('model_v3_away.pkl')
        scaler  = joblib.load('scaler_v3.pkl')
        features = joblib.load('features_v3.pkl')
        return m_h, m_a, scaler, features
    except Exception as e:
        return None, None, None, None

m_home, m_away, scaler, FEATURES = load_models()

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def normaliser_proba(c_dom: float, c_nul: float, c_ext: float):
    """Supprime la marge bookmaker et normalise à 1.0."""
    pd_, pn_, pe_ = 1/c_dom, 1/c_nul, 1/c_ext
    total = pd_ + pn_ + pe_
    return pd_/total, pn_/total, pe_/total


def parse_forme(s: str):
    """Parse '2 1 2' ou '2-1-2' ou '2,1,2' → (v, n, d)."""
    parts = re.split(r'[\s,\-]+', s.strip())
    if len(parts) != 3:
        raise ValueError("Entrez exactement 3 valeurs (V N D), ex: 2 1 2")
    return tuple(map(int, parts))


def calc_note_forme(v: int, n: int, bm: int, be: int, matchs: int = 5) -> float:
    """Note de forme bornée entre 0 et 100."""
    pts_max  = matchs * 3
    note_pts = ((v * 3 + n) / pts_max) * 50
    note_att = min((bm / (matchs * 2.5)) * 30, 30)
    note_def = max(30 - (be / (matchs * 2.5)) * 30, 0)
    return round(min(max(note_pts + note_att + note_def, 0), 100), 2)


def poisson_proba(lambda_h: float, lambda_a: float, max_buts: int = 6):
    """
    Calcule les probabilités 1X2 et la matrice de scores via distribution de Poisson.
    Retourne (prob_dom, prob_nul, prob_ext, matrice_scores).
    """
    matrice = np.zeros((max_buts + 1, max_buts + 1))
    for i, j in product(range(max_buts + 1), range(max_buts + 1)):
        matrice[i][j] = (
            math.exp(-lambda_h) * lambda_h**i / math.factorial(i) *
            math.exp(-lambda_a) * lambda_a**j / math.factorial(j)
        )
    prob_dom = float(np.sum(np.tril(matrice, -1)))
    prob_nul = float(np.trace(matrice))
    prob_ext = float(np.sum(np.triu(matrice, 1)))
    return prob_dom, prob_nul, prob_ext, matrice


def top_scores(matrice: np.ndarray, n: int = 5):
    """Retourne les n scores les plus probables."""
    idx = np.dstack(np.unravel_index(np.argsort(matrice.ravel())[::-1], matrice.shape))[0]
    return [(int(i), int(j), float(matrice[i][j])) for i, j in idx[:n]]


def sauvegarder_gsheets(conn, data: dict):
    """Sauvegarde sécurisée dans Google Sheets."""
    try:
        old_df = conn.read(worksheet="Pronos", ttl=0)
        new_row = pd.DataFrame([data])
        updated = pd.concat([old_df, new_row], ignore_index=True)
        conn.update(worksheet="Pronos", data=updated)
        return True
    except Exception as e:
        st.warning(f"⚠️ Sauvegarde Google Sheets échouée : {e}")
        return False


# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-title">🏆 PPS Predictor V3.2 PRO</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Expert Football Analysis System — Poisson Model</div>', unsafe_allow_html=True)

if m_home is None:
    st.error("⚠️ Modèles IA non chargés. Lancez d'abord `python train_v3.py` pour générer les fichiers `.pkl`.")

# ============================================================
# SIDEBAR — COTES 1 N 2
# ============================================================
with st.sidebar:
    st.markdown("### 📊 Cotes Bookmaker (1 / N / 2)")
    st.caption("Entrez les cotes décimales du marché.")

    c_dom = st.number_input("Cote **1** (Domicile)", min_value=1.01, max_value=50.0, value=1.85, step=0.05,
                             help="Ex: 1.85 → victoire domicile")
    c_nul = st.number_input("Cote **N** (Nul)",      min_value=1.01, max_value=50.0, value=3.40, step=0.05,
                             help="Ex: 3.40 → match nul")
    c_ext = st.number_input("Cote **2** (Extérieur)",min_value=1.01, max_value=50.0, value=4.20, step=0.05,
                             help="Ex: 4.20 → victoire extérieur")

    # Normalisation immédiate
    p_c_dom, p_c_nul, p_c_ext = normaliser_proba(c_dom, c_nul, c_ext)
    marge = (1/c_dom + 1/c_nul + 1/c_ext - 1) * 100

    st.markdown("---")
    st.markdown("**Probabilités implicites normalisées :**")
    st.progress(p_c_dom, text=f"1 (Dom) : {p_c_dom*100:.1f}%")
    st.progress(p_c_nul, text=f"N (Nul) : {p_c_nul*100:.1f}%")
    st.progress(p_c_ext, text=f"2 (Ext) : {p_c_ext*100:.1f}%")
    st.caption(f"Marge bookmaker estimée : **{marge:.1f}%**")

    st.markdown("---")
    st.markdown("### 🔗 Google Sheets")
    if conn is not None:
        st.success("✅ Connecté")
    else:
        st.warning("Non connecté — les pronos ne seront pas sauvegardés.")

# ============================================================
# FORMULAIRE ÉQUIPES
# ============================================================
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-header">🏠 Équipe Domicile</div>', unsafe_allow_html=True)
    n_dom    = st.text_input("Nom", "Gérone", key="n_dom")
    r_dom    = st.number_input(f"Rating ClubElo — {n_dom}", 1000.0, 2200.0, 1880.0, step=10.0)
    rank_dom = st.number_input(f"Rang — {n_dom}", 1, 30, 3)
    f_dom    = st.text_input(f"Forme V N D (5 derniers matchs) — {n_dom}", "2 1 2",
                              help="Ex: 3 1 1 → 3 victoires, 1 nul, 1 défaite")
    bm_dom   = st.number_input(f"Buts marqués (5 matchs) — {n_dom}", 0, 30, 8)
    be_dom   = st.number_input(f"Buts encaissés (5 matchs) — {n_dom}", 0, 30, 5)

with col2:
    st.markdown('<div class="section-header">🚀 Équipe Extérieur</div>', unsafe_allow_html=True)
    n_ext    = st.text_input("Nom", "Majorque", key="n_ext")
    r_ext    = st.number_input(f"Rating ClubElo — {n_ext}", 1000.0, 2200.0, 1720.0, step=10.0)
    rank_ext = st.number_input(f"Rang — {n_ext}", 1, 30, 15)
    f_ext    = st.text_input(f"Forme V N D (5 derniers matchs) — {n_ext}", "1 2 2",
                              help="Ex: 1 2 2 → 1 victoire, 2 nuls, 2 défaites")
    bm_ext   = st.number_input(f"Buts marqués (5 matchs) — {n_ext}", 0, 30, 4)
    be_ext   = st.number_input(f"Buts encaissés (5 matchs) — {n_ext}", 0, 30, 6)

st.markdown("---")
st.markdown('<div class="section-header">⚔️ Confrontations Directes (H2H)</div>', unsafe_allow_html=True)
h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
with h2h_col1:
    vh = st.number_input(f"Victoires {n_dom}", 0, 20, 1)
with h2h_col2:
    nh = st.number_input("Nuls", 0, 20, 1)
with h2h_col3:
    dh = st.number_input(f"Victoires {n_ext}", 0, 20, 1)

st.markdown("---")

# ============================================================
# BOUTON ANALYSE
# ============================================================
btn_disabled = (m_home is None)
if st.button("🚀 LANCER L'ANALYSE PPS V3.2", disabled=btn_disabled, type="primary", use_container_width=True):
    try:
        # ── Parsing forme ──────────────────────────────────────────
        v1, n1, d1 = parse_forme(f_dom)
        v2, n2, d2 = parse_forme(f_ext)

        # ── Notes de forme ─────────────────────────────────────────
        note_dom = calc_note_forme(v1, n1, bm_dom, be_dom)
        note_ext = calc_note_forme(v2, n2, bm_ext, be_ext)

        # ── Variables numériques ────────────────────────────────────
        diff_elo  = r_dom - r_ext
        ecart_rng = rank_ext - rank_dom
        h2h_pts   = vh * 3 + nh          # cohérent avec train_v3.py

        # ── DataFrame de prédiction ────────────────────────────────
        input_data = pd.DataFrame([[
            note_dom, note_ext, diff_elo, ecart_rng,
            h2h_pts, p_c_dom, p_c_nul
        ]], columns=FEATURES)

        # ── Scaling + Prédiction ───────────────────────────────────
        input_scaled = scaler.transform(input_data)
        lambda_h = max(0.1, m_home.predict(input_scaled)[0])
        lambda_a = max(0.1, m_away.predict(input_scaled)[0])

        s_h = int(round(lambda_h))
        s_a = int(round(lambda_a))

        # ── Poisson ────────────────────────────────────────────────
        prob_dom, prob_nul, prob_ext, matrice = poisson_proba(lambda_h, lambda_a)
        scores_top = top_scores(matrice, n=6)

        # ── Résultats ─────────────────────────────────────────────
        st.markdown("## 📊 Résultats de l'Analyse")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            tendance = n_dom if s_h > s_a else (n_ext if s_a > s_h else "Match Nul")
            st.markdown(f'<div class="score-box">⚽ {s_h} – {s_a} ⚽</div>', unsafe_allow_html=True)
            st.metric("Score prédit (λ)", f"{lambda_h:.2f} – {lambda_a:.2f}")
            st.metric("Tendance", tendance)

        with res_col2:
            st.markdown("**Probabilités Poisson 1N2 :**")
            col_p1, col_pn, col_p2 = st.columns(3)
            col_p1.metric(f"1 — {n_dom}", f"{prob_dom*100:.1f}%")
            col_pn.metric("N — Nul",     f"{prob_nul*100:.1f}%")
            col_p2.metric(f"2 — {n_ext}", f"{prob_ext*100:.1f}%")

            st.markdown("**Scores les plus probables :**")
            for i, (gh, ga, p) in enumerate(scores_top):
                bar_val = min(p / scores_top[0][2], 1.0)
                st.progress(bar_val, text=f"  {gh}–{ga}  →  {p*100:.1f}%")

        # ── Comparaison cotes vs Poisson ──────────────────────────
        st.markdown("---")
        st.markdown("**📐 Comparaison cotes vs modèle Poisson :**")
        cmp_col1, cmp_col2, cmp_col3 = st.columns(3)
        delta_dom = (prob_dom - p_c_dom) * 100
        delta_nul = (prob_nul - p_c_nul) * 100
        delta_ext = (prob_ext - p_c_ext) * 100
        cmp_col1.metric(f"1 (Dom)", f"{prob_dom*100:.1f}%", f"{delta_dom:+.1f}% vs cotes",
                        delta_color="normal")
        cmp_col2.metric("N (Nul)", f"{prob_nul*100:.1f}%", f"{delta_nul:+.1f}% vs cotes",
                        delta_color="normal")
        cmp_col3.metric(f"2 (Ext)", f"{prob_ext*100:.1f}%", f"{delta_ext:+.1f}% vs cotes",
                        delta_color="normal")
        st.caption("Un delta positif (+) indique que le modèle estime cette issue plus probable que les cotes.")

        # ── Sauvegarde Google Sheets ───────────────────────────────
        if conn is not None:
            record = {
                "Date":         pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"),
                "Equipe_Dom":   n_dom,
                "Equipe_Ext":   n_ext,
                "Score_Predit": f"{s_h}-{s_a}",
                "Lambda_Dom":   round(lambda_h, 3),
                "Lambda_Ext":   round(lambda_a, 3),
                "Prob_Dom":     round(prob_dom, 4),
                "Prob_Nul":     round(prob_nul, 4),
                "Prob_Ext":     round(prob_ext, 4),
                "Cote_1":       c_dom,
                "Cote_N":       c_nul,
                "Cote_2":       c_ext,
            }
            ok = sauvegarder_gsheets(conn, record)
            if ok:
                st.success("✅ Prono sauvegardé dans Google Sheets !")

        st.balloons()

    except ValueError as ve:
        st.error(f"❌ Erreur de saisie : {ve}")
    except Exception as e:
        st.error(f"❌ Erreur inattendue : {e}")

# ============================================================
# HISTORIQUE GOOGLE SHEETS
# ============================================================
if conn is not None:
    with st.expander("📋 Historique des pronos (Google Sheets)", expanded=False):
        try:
            hist = conn.read(worksheet="Pronos", ttl=60)
            if hist is not None and not hist.empty:
                st.dataframe(hist.tail(20), use_container_width=True)
            else:
                st.info("Aucun prono enregistré pour le moment.")
        except Exception as e:
            st.warning(f"Impossible de lire l'historique : {e}")
