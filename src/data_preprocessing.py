"""
Module de prétraitement des données pour le Credit Scoring.

Ce module fournit les fonctions nécessaires pour charger, nettoyer
et préparer les données du dataset "Give Me Some Credit" avant
la modélisation.
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, index_col=0)
    print(f"[+] Donnees chargees : {df.shape[0]} lignes, {df.shape[1]} colonnes.")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    mediane_revenu = df['MonthlyIncome'].median()
    mediane_dependents = df['NumberOfDependents'].median()

    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(mediane_revenu)
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(mediane_dependents)

    nb_manquants = df.isnull().sum().sum()
    print(f"[+] Valeurs manquantes traitees. Restantes : {nb_manquants}")
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Supprimer les lignes avec un âge de 0
    nb_avant = len(df)
    df = df[df['age'] > 0]
    nb_supprimes = nb_avant - len(df)
    print(f"   * {nb_supprimes} ligne(s) supprimee(s) (age = 0)")

    # 2. Plafonner les colonnes de retards de paiement à 17
    colonnes_retards = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate'
    ]
    for col in colonnes_retards:
        df[col] = df[col].clip(upper=17)

    # 3. Plafonner au 99e percentile
    for col in ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio']:
        seuil_99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=seuil_99)
        print(f"   * {col} plafonne a {seuil_99:.2f}")

    print(f"[+] Outliers traites. {len(df)} lignes restantes.")
    return df


def preprocess_pipeline(filepath: str) -> pd.DataFrame:
    print("=" * 50)
    print("PIPELINE DE PRETRAITEMENT")
    print("=" * 50)

    df = load_data(filepath)
    df = handle_missing_values(df)
    df = handle_outliers(df)

    print("=" * 50)
    print("[+] PRETRAITEMENT TERMINE")
    print(f"   Dimensions finales : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print("=" * 50)

    return df


if __name__ == '__main__':
    # Exécution directe pour tester le pipeline
    df_clean = preprocess_pipeline("data/cs-training.csv")
    print("\nAperçu des données nettoyées :")
    print(df_clean.head())
    print("\nStatistiques descriptives :")
    print(df_clean.describe())
