# 🏦 Credit Scoring — Prédiction de Défaut de Paiement

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

> Modèle de **machine learning** pour prédire le risque de défaut de paiement d'un emprunteur, accompagné d'une **application web interactive** pour le scoring en temps réel.

---

## 📋 Sommaire

- [Résultats Clés](#-résultats-clés)
- [Application Web](#️-application-web)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Méthodologie](#-méthodologie)
- [Technologies](#-technologies-utilisées)
- [Auteur](#-auteur)

---

## 📊 Résultats Clés

Le modèle de **régression logistique** a été entraîné sur le dataset [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit) (Kaggle) contenant **150 000 emprunteurs**.

| Métrique | Valeur |
|---|---|
| **AUC-ROC** | ~0.80 |
| **Observations** | 150 000 |
| **Variables** | 10 features financières |
| **Gestion du déséquilibre** | `class_weight='balanced'` |

### Classes de risque

Le modèle segmente les emprunteurs en **3 classes de risque** :

| Classe | Taux de défaut | Interprétation |
|---|---|---|
| 🟢 **Faible** | ~1-2% | Emprunteurs fiables |
| 🟠 **Moyen** | ~4-6% | Surveillance recommandée |
| 🔴 **Élevé** | ~14-18% | Risque significatif — examen approfondi |

> La classe "Élevé" concentre environ **75-80% de tous les défauts** du portefeuille testé.

---

## 🖥️ Application Web

L'application **Streamlit** permet de :
- ✅ Saisir le profil financier d'un emprunteur
- ✅ Obtenir un **score de risque en temps réel**
- ✅ Visualiser la **classe de risque** (Faible / Moyen / Élevé)
- ✅ Consulter les **performances du modèle** (courbe ROC, matrice de confusion)
- ✅ Comprendre l'**importance des variables** dans la décision

<!-- Capture d'écran à ajouter après le premier lancement -->
<!-- ![Application](images/app_screenshot.png) -->

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Othniel1704/credit-scoring-project.git
cd credit-scoring-project
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Télécharger le dataset

Télécharger `cs-training.csv` depuis [Kaggle - Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data) et le placer dans le dossier `data/`.

---

## 💡 Utilisation

### Entraîner le modèle

```bash
python -m src.model_training
```

Cela génère les fichiers suivants dans `models/` :
- `logistic_model.pkl` — Le modèle entraîné
- `scaler.pkl` — Le StandardScaler

### Évaluer le modèle et générer les graphiques

```bash
python -m src.model_evaluation
```

Cela sauvegarde les graphiques dans `images/` :
- `roc_curve.png`
- `confusion_matrix.png`
- `correlation_heatmap.png`

### Lancer l'application web

```bash
streamlit run app/streamlit_app.py
```

L'application s'ouvre automatiquement dans votre navigateur à l'adresse `http://localhost:8501`.

---

## 📁 Structure du Projet

```
credit-scoring-project/
├── README.md                        # Ce fichier
├── requirements.txt                 # Dépendances Python
├── .gitignore                       # Fichiers ignorés par Git
├── LICENSE                          # Licence MIT
├── notebooks/
│   └── scoring_credit.ipynb         # Notebook d'analyse complet
├── data/
│   └── Data Dictionary.xls          # Dictionnaire de données
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Nettoyage des données
│   ├── model_training.py            # Entraînement du modèle
│   └── model_evaluation.py          # Évaluation et visualisation
├── models/                          # Modèles sérialisés (générés)
├── app/
│   └── streamlit_app.py             # Application web Streamlit
└── images/                          # Graphiques (générés)
```

---

## 📈 Méthodologie

1. **Chargement et compréhension** des données (150 000 emprunteurs, 10 variables)
2. **Nettoyage** : imputation des valeurs manquantes par la médiane, correction des outliers
3. **Analyse exploratoire (EDA)** : distributions, corrélations, déséquilibre de classes
4. **Modélisation** : Régression Logistique avec gestion du déséquilibre
5. **Évaluation** : AUC-ROC, matrice de confusion, rapport de classification
6. **Classes de risque** : segmentation Faible / Moyen / Élevé par terciles
7. **Déploiement** : application web Streamlit pour le scoring interactif

---

## 🔧 Technologies Utilisées

| Technologie | Version | Utilisation |
|---|---|---|
| **Python** | 3.10+ | Langage principal |
| **pandas** | ≥ 2.0 | Manipulation de données |
| **numpy** | ≥ 1.24 | Calcul numérique |
| **scikit-learn** | ≥ 1.3 | Machine Learning |
| **matplotlib** | ≥ 3.7 | Visualisation |
| **seaborn** | ≥ 0.12 | Visualisation statistique |
| **Streamlit** | ≥ 1.30 | Application web |
| **joblib** | ≥ 1.3 | Sérialisation du modèle |

---

## 🤝 Auteur

**Cedric Adannou**

- 🔗 GitHub : [@Othniel1704](https://github.com/Othniel1704)

---

## 📝 Licence

Ce projet est sous licence **MIT** — voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

⭐ **Si ce projet vous est utile, n'hésitez pas à mettre une étoile !**
