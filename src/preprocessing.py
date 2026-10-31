"""Chargement de la base et construction des variables dérivées (cellules 0, 1 et 7 du notebook)."""
import numpy as np
import pandas as pd

from .config import FICHIER_SAV, SOURCES, ZONE_MAP


def charger_base(chemin=FICHIER_SAV):
    import pyreadstat
    df, meta = pyreadstat.read_sav(str(chemin))
    return df, meta


def _freq_min(row, cols):
    vals = [row[v] for v in cols if not pd.isna(row[v]) and row[v] > 0]
    return min(vals) if vals else np.nan


def construire_variables(df):
    """Retourne une copie de df avec les variables dérivées du mémoire."""
    df = df.copy()

    # Utilisation (variables 'l') : 1 si au moins une source utilisée
    vars_util = ['l' + s for s in SOURCES]
    for v in vars_util:
        df[v] = (df[v].fillna(0) > 0).astype(float)
    df['utilisation_info'] = (df[vars_util].sum(axis=1) > 0).astype(float)

    # Réalisme perçu (variables 'u')
    vars_real = ['u' + s for s in SOURCES]
    for v in vars_real:
        df[v] = df[v].fillna(0)
    df['realisme_percu'] = (df[vars_real].sum(axis=1) > 0).astype(float)

    # Accessibilité : OU logique, 1 si accès à AU MOINS UNE des 10 sources
    # (et non une somme de pourcentages, qui dépasserait 100 %)
    vars_acces = ['s' + s for s in SOURCES]
    df['accessibilite'] = (df[vars_acces] == 1).any(axis=1).astype(float)

    # Fréquence de réception : code minimal parmi les sources ('a'), médiane si manquant
    vars_freq = ['a' + s for s in SOURCES]
    df['freq_reception'] = df.apply(_freq_min, axis=1, cols=vars_freq)
    df['freq_reception'] = df['freq_reception'].fillna(df['freq_reception'].median())

    # Imputation par la médiane + variables dérivées
    for v in ['supdis', 'tailm', 'nactif']:
        df[v] = df[v].fillna(df[v].median())
    df['smetri_bin'] = (df['smetri'] == 1).astype(float)
    df['zone'] = df['depart'].map(ZONE_MAP)
    return df
