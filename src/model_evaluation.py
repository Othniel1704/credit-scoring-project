"""
Module d'évaluation du modèle de Credit Scoring.

Ce module fournit les fonctions pour évaluer les performances
du modèle (AUC-ROC, matrice de confusion, rapport de classification)
et pour créer les classes de risque (Faible / Moyen / Élevé).
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
import joblib

from src.data_preprocessing import preprocess_pipeline

# Style matplotlib / seaborn
sns.set_theme(style='whitegrid')


def evaluate_model(
    model,
    X_test_scaled: np.ndarray,
    y_test: pd.Series
) -> Tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"\n[+] AUC-ROC : {auc:.4f}")
    print(f"\n[+] Rapport de classification :\n")
    print(classification_report(
        y_test, y_pred,
        target_names=['Pas de defaut', 'Defaut']
    ))

    return y_pred, y_proba


def plot_roc_curve(
    y_test: pd.Series,
    y_proba: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    auc = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, label=f'Modele (AUC = {auc:.3f})',
            color='#2196F3', linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray',
            label='Hasard (AUC = 0.5)', linewidth=1.5)
    ax.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
    ax.set_xlabel('Taux de Faux Positifs', fontsize=13)
    ax.set_ylabel('Taux de Vrais Positifs', fontsize=13)
    ax.set_title('Courbe ROC -- Modele de Credit Scoring', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Courbe ROC sauvegardee -> {save_path}")

    plt.close()


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    matrice = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matrice,
        display_labels=['Pas de defaut', 'Defaut']
    )
    disp.plot(cmap='Blues', ax=ax)
    ax.set_title('Matrice de Confusion', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Matrice de confusion sauvegardee -> {save_path}")

    plt.close()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    matrice_correlation = df.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(
        matrice_correlation,
        annot=True, fmt='.2f',
        cmap='coolwarm', center=0,
        ax=ax, linewidths=0.5
    )
    ax.set_title('Matrice de Correlation', fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[+] Heatmap de correlation sauvegardee -> {save_path}")

    plt.close()


def create_risk_classes(
    y_proba: np.ndarray,
    X_test: np.ndarray,
    y_test: pd.Series,
    feature_names: list
) -> pd.DataFrame:
    resultats = pd.DataFrame(X_test, columns=feature_names)
    resultats['score_probabilite_defaut'] = y_proba
    resultats['defaut_reel'] = y_test.values

    resultats['classe_risque'] = pd.qcut(
        resultats['score_probabilite_defaut'],
        q=3,
        labels=['Faible', 'Moyen', 'Eleve']
    )

    synthese = resultats.groupby('classe_risque', observed=True).agg(
        nb_emprunteurs=('defaut_reel', 'count'),
        nb_defauts_reels=('defaut_reel', 'sum'),
        taux_defaut_pct=('defaut_reel', lambda x: round(x.mean() * 100, 2))
    )

    print("\n[+] Synthese par classe de risque :")
    print(synthese)

    return resultats


def main() -> None:
    print("=" * 50)
    print("EVALUATION DU MODELE DE CREDIT SCORING")
    print("=" * 50)

    model = joblib.load("models/logistic_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    X_test_scaled = joblib.load("models/X_test_scaled.pkl")
    y_test = joblib.load("models/y_test.pkl")
    feature_names = joblib.load("models/feature_names.pkl")

    print("[+] Modele et donnees mecharges.")

    y_pred, y_proba = evaluate_model(model, X_test_scaled, y_test)

    print("\nGeneration des graphiques...")
    plot_roc_curve(y_test, y_proba, save_path="images/roc_curve.png")
    plot_confusion_matrix(y_test, y_pred, save_path="images/confusion_matrix.png")

    df = preprocess_pipeline("data/cs-training.csv")
    plot_correlation_heatmap(df, save_path="images/correlation_heatmap.png")

    print("\nCreation des classes de risque...")
    create_risk_classes(y_proba, X_test_scaled, y_test, feature_names)

    print("\n" + "=" * 50)
    print("[+] EVALUATION TERMINEE")
    print("=" * 50)


if __name__ == '__main__':
    main()
