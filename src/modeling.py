"""VIF, régression logistique, diagnostics d'influence (cellules 12 à 15, puis 5)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, roc_auc_score

from .config import VARS_CANDIDATES, FREQ_LABELS, LABELS_MODELE
from .descriptive import signif


def preparer_candidats(df):
    df_cand = df[VARS_CANDIDATES].dropna().copy()
    return df_cand


def calcul_vif(df_cand):
    X = df_cand.drop(columns=['utilisation_info'])
    X = pd.get_dummies(X, columns=['zone'], drop_first=True).astype(float)
    Xc = sm.add_constant(X)
    vif = pd.DataFrame({'Variable': X.columns,
                        'VIF': [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]})
    vif['Statut'] = vif['VIF'].apply(lambda v: 'PROBLEME SERIEUX' if v > 10
                                     else ('Colinearite moderee' if v > 5 else 'OK'))
    return vif.sort_values('VIF', ascending=False)


def matrice_modele(df_cand):
    """Dummies : référence = Quotidien (fréquence) et Sud (zone)."""
    d = df_cand.copy()
    d['freq_reception'] = d['freq_reception'].map(FREQ_LABELS)
    d = pd.get_dummies(d, columns=['freq_reception', 'zone'], drop_first=False).astype(float)
    return d.drop(columns=['freq_reception_Quotidien', 'zone_Sud'])


def ajuster_logit(df_mod):
    Y = df_mod['utilisation_info']
    Xc = sm.add_constant(df_mod.drop(columns='utilisation_info'))
    return sm.Logit(Y, Xc).fit(disp=0), Y, Xc


def observations_influentes(modele, Xc):
    """Observation influente = les 4 critères réunis (résidu studentisé, levier, Cook, DFBETAS)."""
    infl = modele.get_influence()
    n, k = Xc.shape
    seuils = {'levier': 2 * k / n, 'cook': 4 / n, 'dfbeta': 2 / np.sqrt(n)}
    masque = ((np.abs(infl.resid_studentized) > 2) &
              (infl.hat_matrix_diag > seuils['levier']) &
              (infl.cooks_distance[0] > seuils['cook']) &
              (np.abs(infl.dfbetas).max(axis=1) > seuils['dfbeta']))
    return masque, seuils


def performance(modele, Y):
    pred = modele.predict()
    cm = confusion_matrix(Y, (pred > 0.5).astype(int))
    return {'matrice_confusion': cm,
            'taux_classification': round((cm[0, 0] + cm[1, 1]) / cm.sum() * 100, 1),
            'auc': roc_auc_score(Y, pred),
            'pseudo_r2': modele.prsquared}


def tableau_odds_ratios(modele):
    rows = []
    for var in modele.params.index:
        coef, se, p = modele.params[var], modele.bse[var], modele.pvalues[var]
        rows.append(['Const' if var == 'const' else var, LABELS_MODELE.get(var, var),
                     round(coef, 4), round(se, 4), round(np.exp(coef), 4), f"{p:.4f} {signif(p)}"])
    return pd.DataFrame(rows, columns=['Variable', 'Description', 'Coefficient',
                                       'Erreur standard', 'Odds Ratio', 'p-value'])
