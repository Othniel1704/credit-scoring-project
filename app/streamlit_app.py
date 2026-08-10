"""
🏦 Application Web de Credit Scoring — Streamlit

Cette application permet de :
1. Saisir les données d'un emprunteur
2. Obtenir un score de risque de défaut en temps réel
3. Visualiser les performances du modèle
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────────────
# CONFIGURATION DE LA PAGE
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Scoring — Prédiction de Défaut",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────
# STYLES CSS PERSONNALISÉS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Imports Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
        margin: 0;
    }

    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
        margin: 1rem 0;
    }

    .score-value {
        font-size: 4rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }

    .score-label {
        font-size: 1.1rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Risk badge */
    .risk-badge {
        display: inline-block;
        padding: 0.6rem 2rem;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1rem;
        letter-spacing: 1px;
    }

    .risk-low {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
    }

    .risk-medium {
        background: linear-gradient(135deg, #fdcb6e, #e17055);
        color: white;
    }

    .risk-high {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
    }

    /* Progress bar */
    .progress-container {
        background: #2d3436;
        border-radius: 10px;
        height: 16px;
        margin: 1.5rem 0;
        overflow: hidden;
    }

    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }

    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a3d);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
    }

    .info-card h4 {
        color: #74b9ff;
        margin: 0 0 0.5rem 0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .info-card p {
        color: #dfe6e9;
        margin: 0;
        font-size: 1.5rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        color: #74b9ff;
        font-size: 1.4rem;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2d3436;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE
# ─────────────────────────────────────────────────────
def get_model_paths():
    """Recherche les chemins des fichiers modèle dans plusieurs emplacements possibles."""
    possible_base_dirs = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd(),
        os.path.join(os.getcwd(), ".."),
    ]
    for bdir in possible_base_dirs:
        m_path = os.path.abspath(os.path.join(bdir, "models", "logistic_model.pkl"))
        s_path = os.path.abspath(os.path.join(bdir, "models", "scaler.pkl"))
        f_path = os.path.abspath(os.path.join(bdir, "models", "feature_names.pkl"))
        if os.path.exists(m_path) and os.path.exists(s_path):
            return m_path, s_path, f_path
    return None, None, None

@st.cache_resource
def load_model_cached(model_path, scaler_path, features_path):
    """Charge le modèle et le scaler en mémoire."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_names = joblib.load(features_path) if features_path and os.path.exists(features_path) else None
    return model, scaler, feature_names

def load_model():
    """Fonction principale de chargement avec résilience."""
    m_path, s_path, f_path = get_model_paths()
    if not m_path:
        return None, None, None
    try:
        return load_model_cached(m_path, s_path, f_path)
    except Exception as e:
        return None, None, None


# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 Credit Scoring</h1>
    <p>Modèle de prédiction de défaut de paiement — Régression Logistique</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# SIDEBAR : SAISIE DES DONNÉES
# ─────────────────────────────────────────────────────
st.sidebar.markdown("## 📝 Profil de l'emprunteur")
st.sidebar.markdown("---")

age = st.sidebar.slider("🎂 Âge", min_value=18, max_value=100, value=40)

revenu = st.sidebar.number_input(
    "💰 Revenu mensuel ($)",
    min_value=0, max_value=100000, value=5400, step=100
)

taux_utilisation = st.sidebar.slider(
    "💳 Taux d'utilisation du crédit renouvelable",
    min_value=0.0, max_value=1.1, value=0.3, step=0.01,
    help="Solde utilisé / limite de crédit totale"
)

ratio_endettement = st.sidebar.slider(
    "📊 Ratio d'endettement",
    min_value=0.0, max_value=5000.0, value=0.4, step=0.01,
    help="Mensualités de dettes / revenu mensuel brut"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Historique de paiement")

retards_30_59 = st.sidebar.number_input(
    "Retards 30-59 jours", min_value=0, max_value=17, value=0
)

retards_60_89 = st.sidebar.number_input(
    "Retards 60-89 jours", min_value=0, max_value=17, value=0
)

retards_90 = st.sidebar.number_input(
    "Retards 90+ jours", min_value=0, max_value=17, value=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏠 Situation financière")

nb_credits = st.sidebar.number_input(
    "Nombre de crédits ouverts", min_value=0, max_value=58, value=8
)

nb_immobilier = st.sidebar.number_input(
    "Prêts immobiliers", min_value=0, max_value=54, value=1
)

nb_dependants = st.sidebar.number_input(
    "Personnes à charge", min_value=0, max_value=20, value=0
)

btn_calculer = st.sidebar.button("🔍 Calculer le Score", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Scoring", "📈 Performance du Modèle", "ℹ️ À Propos"])

# ── TAB 1 : SCORING ────────────────────────────────
with tab1:
    if btn_calculer:
        model, scaler, feature_names = load_model()

        if model is None:
            st.warning(
                "⚠️ Le modèle n'a pas été trouvé. "
                "Exécutez d'abord `python -m src.model_training` depuis la racine du projet "
                "pour entraîner et sauvegarder le modèle."
            )
        else:
            # Créer le DataFrame avec les données saisies
            input_data = pd.DataFrame([{
                'RevolvingUtilizationOfUnsecuredLines': taux_utilisation,
                'age': age,
                'NumberOfTime30-59DaysPastDueNotWorse': retards_30_59,
                'DebtRatio': ratio_endettement,
                'MonthlyIncome': revenu,
                'NumberOfOpenCreditLinesAndLoans': nb_credits,
                'NumberOfTimes90DaysLate': retards_90,
                'NumberRealEstateLoansOrLines': nb_immobilier,
                'NumberOfTime60-89DaysPastDueNotWorse': retards_60_89,
                'NumberOfDependents': nb_dependants
            }])

            # Prédiction
            input_scaled = scaler.transform(input_data)
            proba_defaut = model.predict_proba(input_scaled)[0][1]
            score_pct = round(proba_defaut * 100, 1)

            # Déterminer la classe de risque
            if proba_defaut < 0.10:
                classe = "Faible"
                badge_class = "risk-low"
                color = "#00b894"
            elif proba_defaut < 0.30:
                classe = "Moyen"
                badge_class = "risk-medium"
                color = "#e17055"
            else:
                classe = "Élevé"
                badge_class = "risk-high"
                color = "#e74c3c"

            # Affichage du score
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown(f"""
                <div class="score-card">
                    <div class="score-label">Probabilité de défaut</div>
                    <div class="score-value" style="color: {color};">{score_pct}%</div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {min(score_pct, 100)}%; background: {color};"></div>
                    </div>
                    <div class="risk-badge {badge_class}">Risque {classe}</div>
                </div>
                """, unsafe_allow_html=True)

            # Détails
            st.markdown('<div class="section-header">📋 Détails du profil analysé</div>', unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""
                <div class="info-card">
                    <h4>🎂 Âge</h4>
                    <p>{age} ans</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="info-card">
                    <h4>💰 Revenu</h4>
                    <p>${revenu:,}</p>
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div class="info-card">
                    <h4>💳 Utilisation crédit</h4>
                    <p>{taux_utilisation:.0%}</p>
                </div>
                """, unsafe_allow_html=True)
            with c4:
                st.markdown(f"""
                <div class="info-card">
                    <h4>📊 Endettement</h4>
                    <p>{ratio_endettement:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

            # Importance des variables (coefficients du modèle)
            st.markdown('<div class="section-header">📊 Importance des variables dans la décision</div>',
                        unsafe_allow_html=True)

            if feature_names:
                coef_df = pd.DataFrame({
                    'Variable': feature_names,
                    'Coefficient': model.coef_[0]
                }).sort_values('Coefficient', key=abs, ascending=True)

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 5))
                colors = ['#e74c3c' if c > 0 else '#00b894' for c in coef_df['Coefficient']]
                ax.barh(coef_df['Variable'], coef_df['Coefficient'], color=colors)
                ax.set_xlabel('Coefficient (impact sur le risque de défaut)', fontsize=11)
                ax.set_title('Coefficients du modèle de régression logistique', fontsize=13, fontweight='bold')
                ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                st.info(
                    "🔴 **Rouge** = augmente le risque de défaut | "
                    "🟢 **Vert** = diminue le risque de défaut"
                )

    else:
        st.markdown("""
        ### 👈 Remplissez le profil de l'emprunteur

        Utilisez le panneau latéral pour saisir les informations financières
        de l'emprunteur, puis cliquez sur **Calculer le Score** pour obtenir
        une prédiction de risque de défaut.

        ---

        **Comment fonctionne le scoring ?**

        Le modèle de **régression logistique** analyse 10 variables financières
        pour calculer la probabilité qu'un emprunteur fasse défaut (retard de
        paiement ≥ 90 jours) dans les 2 prochaines années.

        Le score est converti en 3 classes de risque :
        - 🟢 **Faible** : probabilité < 10%
        - 🟠 **Moyen** : probabilité entre 10% et 30%
        - 🔴 **Élevé** : probabilité > 30%
        """)


# ── TAB 2 : PERFORMANCE ───────────────────────────
with tab2:
    st.markdown("## 📈 Performance du Modèle")
    st.markdown("---")

    possible_img_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images"),
        os.path.join(os.getcwd(), "images"),
        os.path.join(os.getcwd(), "..", "images")
    ]
    images_dir = next((d for d in possible_img_dirs if os.path.exists(d)), possible_img_dirs[0])

    col1, col2 = st.columns(2)

    with col1:
        roc_path = os.path.join(images_dir, "roc_curve.png")
        if os.path.exists(roc_path):
            st.image(roc_path, caption="Courbe ROC", use_container_width=True)
        else:
            st.warning("📊 Image non trouvée. Exécutez `python -m src.model_evaluation`")

    with col2:
        cm_path = os.path.join(images_dir, "confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Matrice de Confusion", use_container_width=True)
        else:
            st.warning("📊 Image non trouvée. Exécutez `python -m src.model_evaluation`")

    heatmap_path = os.path.join(images_dir, "correlation_heatmap.png")
    if os.path.exists(heatmap_path):
        st.image(heatmap_path, caption="Matrice de Corrélation", use_container_width=True)
    else:
        st.warning("📊 Image non trouvée. Exécutez `python -m src.model_evaluation`")

    st.markdown("""
    ### 📝 À propos du modèle

    | Caractéristique | Valeur |
    |---|---|
    | **Algorithme** | Régression Logistique |
    | **Dataset** | Give Me Some Credit (Kaggle) |
    | **Observations** | ~150 000 emprunteurs |
    | **Variables** | 10 features financières |
    | **Gestion du déséquilibre** | `class_weight='balanced'` |
    | **Découpage** | 70% train / 30% test (stratifié) |
    | **Standardisation** | StandardScaler (z-score) |
    """)


# ── TAB 3 : À PROPOS ──────────────────────────────
with tab3:
    st.markdown("## ℹ️ À Propos du Projet")
    st.markdown("---")

    st.markdown("""
    ### 🎯 Objectif

    Ce projet implémente un **modèle de scoring de risque de crédit** pour prédire
    si un emprunteur fera défaut (retard de paiement ≥ 90 jours) dans les 2 prochaines années.

    Il s'agit d'un cas d'usage classique et fondamental dans le secteur bancaire,
    utilisant des techniques de **machine learning supervisé**.

    ### 📊 Dataset

    **Give Me Some Credit** — [Kaggle Competition](https://www.kaggle.com/competitions/GiveMeSomeCredit)

    - 150 000 emprunteurs
    - 10 variables socio-financières
    - Variable cible : défaut de paiement à 2 ans

    ### 🔧 Technologies

    | Technologie | Utilisation |
    |---|---|
    | **Python 3.10+** | Langage principal |
    | **pandas** | Manipulation de données |
    | **scikit-learn** | Modélisation ML |
    | **matplotlib / seaborn** | Visualisation |
    | **Streamlit** | Application web |
    | **joblib** | Sérialisation du modèle |

    ### 👤 Auteur

    **Kouakou Konan**

    - 🔗 [GitHub](https://github.com/Othniel1704)

    ### 📝 Licence

    Ce projet est sous licence **MIT**.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #636e72; font-size: 0.85rem;'>"
    "🏦 Credit Scoring — Développé par Kouakou Konan — "
    "<a href='https://github.com/Othniel1704' style='color: #74b9ff;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True
)
