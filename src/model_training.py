"""
Module d'entraînement du modèle de Credit Scoring.

Ce module gère la préparation des données, l'entraînement
d'un modèle de régression logistique, et la sauvegarde
du modèle et du scaler pour une utilisation ultérieure.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

from src.data_preprocessing import preprocess_pipeline


def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Sépare les données en train/test et applique la standardisation.

    Étapes :
    1. Séparation X (features) / y (cible)
    2. Découpage train/test (70/30) avec stratification
    3. Standardisation (z-score) via StandardScaler

    Paramètres
    ----------
    df : pd.DataFrame
        Le DataFrame nettoyé.

    Retourne
    --------
    Tuple contenant :
        - X_train_scaled : données d'entraînement standardisées
        - X_test_scaled  : données de test standardisées
        - y_train        : cibles d'entraînement
        - y_test         : cibles de test
        - scaler         : l'objet StandardScaler (pour réutilisation)
    """
    # Séparation features / cible
    X = df.drop(columns=['SeriousDlqin2yrs'])
    y = df['SeriousDlqin2yrs']

    print(f"   * Variables explicatives : {X.shape[1]} colonnes")
    print(f"   * Variable cible : {y.value_counts().to_dict()}")

    # Découpage train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )
    print(f"   * Jeu d'entraînement : {X_train.shape[0]} lignes")
    print(f"   * Jeu de test : {X_test.shape[0]} lignes")

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("   * Standardisation appliquée")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_model(X_train_scaled: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    """
    Entraîne un modèle de régression logistique.

    Le paramètre class_weight='balanced' est utilisé pour gérer
    le déséquilibre du dataset (~93% non-défaut vs ~7% défaut).

    Paramètres
    ----------
    X_train_scaled : np.ndarray
        Données d'entraînement standardisées.
    y_train : pd.Series
        Variable cible d'entraînement.

    Retourne
    --------
    LogisticRegression
        Le modèle entraîné.
    """
    model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    print("[+] Modèle de régression logistique entraîné.")
    return model


def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    model_path: str = "models/logistic_model.pkl",
    scaler_path: str = "models/scaler.pkl"
) -> None:
    """
    Sauvegarde le modèle et le scaler avec joblib.

    Paramètres
    ----------
    model : LogisticRegression
        Le modèle entraîné.
    scaler : StandardScaler
        Le scaler ajusté sur les données d'entraînement.
    model_path : str
        Chemin de sauvegarde du modèle.
    scaler_path : str
        Chemin de sauvegarde du scaler.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"[+] Modèle sauvegardé -> {model_path}")
    print(f"[+] Scaler sauvegardé -> {scaler_path}")


def main() -> None:
    """
    Pipeline principal : prétraitement → entraînement → sauvegarde.
    """
    print("=" * 50)
    print("ENTRAÎNEMENT DU MODÈLE DE CREDIT SCORING")
    print("=" * 50)

    # 1. Prétraitement
    df = preprocess_pipeline("data/cs-training.csv")

    # 2. Préparation des données
    print("\nPréparation des données...")
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(df)

    # 3. Entraînement
    print("\nEntraînement du modèle...")
    model = train_model(X_train_scaled, y_train)

    # 4. Sauvegarde
    print("\nSauvegarde du modèle...")
    save_model(model, scaler)

    # 5. Sauvegarde des données de test pour l'évaluation
    # (on sauvegarde aussi X_test_scaled et y_test pour l'évaluation)
    joblib.dump(X_test_scaled, "models/X_test_scaled.pkl")
    joblib.dump(y_test, "models/y_test.pkl")
    joblib.dump(df.drop(columns=['SeriousDlqin2yrs']).columns.tolist(), "models/feature_names.pkl")
    print("[+] Données de test sauvegardées pour l'évaluation.")

    print("\n" + "=" * 50)
    print("[+] PIPELINE D'ENTRAÎNEMENT TERMINÉ")
    print("=" * 50)


if __name__ == '__main__':
    main()
