# -*- coding: utf-8 -*-
"""
Analyse de l'utilisation de l'information climatique par les producteurs.

Version ordonnée du notebook Memoire2.ipynb : mêmes calculs, même résultats.


# %% 1. Imports et paramètres
import sys
import numpy as np
import pandas as pd
import pyreadstat
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, roc_auc_score

FICHIER_SAV = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].lower().endswith('.sav')
               else r"F:\Mémoire_pris_au_Sérieux\Base Producteurs IDID 2013 V3.sav")


# 10 sources d'infos climatiques
SOURCES = ['cecp', 'ong', 'coop', 'four', 'vois', 'radio', 'tele', 'fone', 'int', 'jour']

SOURCES_LABELS = {
    'cecp':  ('CeCPA', 'Formel/institutionnel'),
    'ong':   ('ONG', 'Formel/institutionnel'),
    'coop':  ('Coopérative', 'Formel/institutionnel'),
    'four':  ('Fournisseurs', 'Formel/institutionnel'),
    'vois':  ('Voisins/Parents', 'Informel/social'),
    'radio': ('Radio', 'Médias'),
    'tele':  ('Télévision', 'Médias'),
    'fone':  ('Téléphone', 'Médias'),
    'int':   ('Internet', 'Médias'),
    'jour':  ('Journaux', 'Médias'),
}

CAT_MAP = {
    'Formel/institutionnel': ['scecp', 'song', 'scoop', 'sfour'],
    'Médias':                ['sradio', 'stele', 'sfone', 'sint', 'sjour'],
    'Informel/social':       ['svois'],
}

# %% 2. Chargement de la base
df, meta = pyreadstat.read_sav(FICHIER_SAV)
print("Base chargée :", df.shape[0], "producteurs")


# Utilisation : au moins une source utilisée
vars_util = ['l' + s for s in SOURCES]
for v in vars_util:
    df[v] = df[v].fillna(0)
    df[v] = (df[v] > 0).astype(float)
df['utilisation_info'] = (df[vars_util].sum(axis=1) > 0).astype(float)

# Réalisme perçu : au moins une source jugée utile
vars_real = ['u' + s for s in SOURCES]
for v in vars_real:
    df[v] = df[v].fillna(0)
df['realisme_percu'] = (df[vars_real].sum(axis=1) > 0).astype(float)

# Accessibilité : accès à au moins une des 10 sources (OU logique, pas une somme)
vars_acces = ['s' + s for s in SOURCES]
df['accessibilite'] = (df[vars_acces] == 1).any(axis=1).astype(float)

# Fréquence de réception : la plus élevée parmi les sources reçues
vars_freq = ['a' + s for s in SOURCES]
def get_freq(row):
    vals = [row[v] for v in vars_freq if not pd.isna(row[v]) and row[v] > 0]
    return min(vals) if vals else np.nan
df['freq_reception'] = df.apply(get_freq, axis=1)
df['freq_reception'] = df['freq_reception'].fillna(df['freq_reception'].median())

# Imputations par la médiane, situation matrimoniale et zone
for v in ['supdis', 'tailm', 'nactif']:
    df[v] = df[v].fillna(df[v].median())
df['smetri_bin'] = (df['smetri'] == 1).astype(float)
zone_map = {1: 'Nord', 5: 'Nord', 3: 'Centre', 2: 'Sud', 4: 'Sud', 6: 'Sud'}
df['zone'] = df['depart'].map(zone_map)

print("Utilisation_info:", df['utilisation_info'].value_counts().to_dict())
print("Realisme_percu:", df['realisme_percu'].value_counts().to_dict())
n_acces = int(df['accessibilite'].sum())
print("Accessibilite:", n_acces, "/", len(df), "=", round(n_acces / len(df) * 100, 1), "%")

# %% 4. Tableau 2 : accès par source et par catégorie
rows = []
for suffix, (label, categorie) in SOURCES_LABELS.items():
    n = int((df['s' + suffix] == 1).sum())
    rows.append({'Source': label, 'Catégorie': categorie,
                 'N avec accès': n, '% accès': round(n / len(df) * 100, 1)})
tab2 = pd.DataFrame(rows).sort_values('% accès', ascending=False).reset_index(drop=True)
tab2['Rang'] = tab2.index + 1
print(tab2.to_string(index=False))

# Par catégorie : union (un producteur est compté une fois par catégorie)
rows_cat = []
for cat, cols in CAT_MAP.items():
    n = int((df[cols] == 1).any(axis=1).sum())
    rows_cat.append({'Catégorie': cat, 'N avec accès': n, '% accès': round(n / len(df) * 100, 1)})
tab2_cat = pd.DataFrame(rows_cat)
print("\n--- Accès par catégorie ---")
print(tab2_cat.to_string(index=False))

# %% 5. Figure 2 : accès par source et par catégorie
color_map = {'Formel/institutionnel': '#4472C4', 'Médias': '#2E75B6', 'Informel/social': '#ED7D31'}
tab2_cat_sorted = tab2_cat.sort_values('% accès', ascending=False)
colors_src = [color_map[c] for c in tab2['Catégorie']]
colors_cat = [color_map[c] for c in tab2_cat_sorted['Catégorie']]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].barh(tab2['Source'][::-1], tab2['% accès'][::-1], color=colors_src[::-1])
axes[0].set_xlabel("Taux d'accès (%)")
for i, v in enumerate(tab2['% accès'][::-1]):
    axes[0].text(v + 1, i, f'{v}%', va='center', fontsize=9)

axes[1].bar(tab2_cat_sorted['Catégorie'], tab2_cat_sorted['% accès'], color=colors_cat)
axes[1].set_ylabel("Taux d'accès (%)")
axes[1].set_ylim(0, 100)
for i, v in enumerate(tab2_cat_sorted['% accès']):
    axes[1].text(i, v + 2, f'{v}%', ha='center', fontsize=10, fontweight='bold')

handles = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
plt.savefig('Figure2_Acces_Sources.png', dpi=150, bbox_inches='tight')
plt.show()

# %% 6. Tableau 3 : utilisation selon les caractéristiques (variables qualitatives)
n_total = len(df)
rows = []

def add_qual_block(var, label, modalites, order):
    tab = pd.crosstab(df[var], df['utilisation_info'])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'ns'
    first = True
    for k in order:
        if k not in tab.index:
            continue
        n_mod = int(tab.loc[k].sum())
        n_oui = int(tab.loc[k, 1]) if 1 in tab.columns else 0
        n_non = int(tab.loc[k, 0]) if 0 in tab.columns else 0
        rows.append([
            label if first else '', modalites[k],
            round(n_oui / n_mod * 100, 1), round(n_non / n_mod * 100, 1),
            round(n_mod / n_total * 100, 1),
            'Chi²' if first else '', round(chi2, 3) if first else '',
            f"{p:.4f} {sig}" if first else ''
        ])
        first = False

add_qual_block('realisme_percu', 'Réalisme perçu', {0: 'Non réaliste', 1: 'Réaliste'}, [0, 1])
add_qual_block('accessibilite', 'Accessibilité de la source', {0: 'Sans accès', 1: 'Avec accès'}, [0, 1])
add_qual_block('freq_reception', 'Fréquence de réception',
               {1: 'Quotidien', 2: 'Hebdomadaire', 3: 'Mensuel', 4: 'Saisonnier'}, [1, 2, 3, 4])
add_qual_block('sex', 'Sexe', {0: 'Masculin', 1: 'Féminin'}, [0, 1])
add_qual_block('alpha', 'Alphabétisation', {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('poscel', "Possession d'un téléphone", {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('posradio', "Possession d'une radio", {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('postele', "Possession d'une télévision", {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('acred', 'Accès au crédit formel', {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('crefor', 'Accès au crédit informel', {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('vulg', 'Contact avec les services de vulgarisation', {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('group', "Membre d'un groupement/coopérative", {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('proch', 'A des proches dans le village', {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('smetri_bin', 'Situation matrimoniale (marié)', {0: 'Non', 1: 'Oui'}, [0, 1])
add_qual_block('infocc', 'Connaissance du changement climatique', {0: 'Non', 1: 'Oui'}, [0, 1])

tab3_qual = pd.DataFrame(rows, columns=[
    'Caractéristiques', 'Modalité', 'Oui (%)', 'Non (%)', 'Ensemble (%)',
    'Test', 'Statistique (χ²)', 'p-value'])
print(tab3_qual.to_string(index=False))

# %% 7. Tableau 3 (suite) : variables quantitatives
vars_quant = [
    ('agep',   'Âge (années)'),
    ('educ',   "Niveau d'instruction (années)"),
    ('supdis', 'Superficie disponible (ha)'),
    ('tailm',  'Taille du ménage'),
    ('nactif', "Nombre d'actifs agricoles"),
    ('parna',  'Part revenu non agricole (Sur 10)'),
    ('parpv',  'Part revenu végétal (Sur 10)'),
]
rows_q = []
for var, label in vars_quant:
    g0 = df[df['utilisation_info'] == 0][var].dropna()
    g1 = df[df['utilisation_info'] == 1][var].dropna()
    g_all = df[var].dropna()
    t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
    sig = '***' if p_val < 0.01 else '**' if p_val < 0.05 else '*' if p_val < 0.10 else 'ns'
    rows_q.append([
        label,
        f"{g1.mean():.2f} ({g1.std():.2f})",
        f"{g0.mean():.2f} ({g0.std():.2f})",
        f"{g_all.mean():.2f} ({g_all.std():.2f})",
        't Student', round(t_stat, 3), f"{p_val:.4f} {sig}"])
tab3_quant = pd.DataFrame(rows_q, columns=[
    'Variable', 'Oui (moy±ET)', 'Non (moy±ET)', 'Ensemble (moy±ET)',
    'Test', 'Statistique (t)', 'p-value'])
print(tab3_quant.to_string(index=False))

# %% 8. Multicolinéarité (VIF)
vars_candidats = ['utilisation_info', 'realisme_percu', 'freq_reception',
                  'sex', 'agep', 'educ', 'alpha', 'smetri_bin', 'group', 'vulg',
                  'acred', 'crefor', 'supdis', 'parpv', 'parna', 'tailm', 'nactif',
                  'poscel', 'posradio', 'postele', 'zone']
vars_candidats = [v for v in vars_candidats if v not in EXCLUSIONS]

df_cand = df[vars_candidats].dropna().copy()
print("N =", len(df_cand))

X_vif = df_cand.drop(columns=['utilisation_info']).copy()
X_vif = pd.get_dummies(X_vif, columns=['zone'], drop_first=True).astype(float)
X_vif_c = sm.add_constant(X_vif)

vif = pd.DataFrame()
vif['Variable'] = X_vif.columns
vif['VIF'] = [variance_inflation_factor(X_vif_c.values, i) for i in range(1, X_vif_c.shape[1])]
vif['Statut'] = vif['VIF'].apply(
    lambda v: 'PROBLEME SERIEUX' if v > 10 else ('Colinearite moderee' if v > 5 else 'OK'))
print(vif.sort_values('VIF', ascending=False).to_string(index=False))

# %% 9. Modèle 1 (régression logistique)
df_mod = df_cand.copy()
df_mod['freq_reception'] = df_mod['freq_reception'].map(
    {1: 'Quotidien', 2: 'Hebdomadaire', 3: 'Mensuel', 4: 'Saisonnier'})
df_mod = pd.get_dummies(df_mod, columns=['freq_reception', 'zone'], drop_first=False).astype(float)
df_mod = df_mod.drop(columns=[c for c in df_mod.columns
                              if c in ['freq_reception_Quotidien', 'zone_Sud']])

Y = df_mod['utilisation_info']
X = df_mod.drop('utilisation_info', axis=1)
Xc = sm.add_constant(X)

modele1 = sm.Logit(Y, Xc).fit(disp=0)
print(modele1.summary())
print("Pseudo R2 Modele 1:", modele1.prsquared)
print("AUC Modele 1:", roc_auc_score(Y, modele1.predict()))

# %% 10. Observations influentes
infl = modele1.get_influence()
resid_std = infl.resid_studentized
leverage = infl.hat_matrix_diag
cooks = infl.cooks_distance[0]
dfbetas = infl.dfbetas

n, k = Xc.shape
seuil_lev = 2 * k / n
seuil_cook = 4 / n
seuil_dfbeta = 2 / np.sqrt(n)

influentes = (
    (np.abs(resid_std) > 2) &
    (leverage > seuil_lev) &
    (cooks > seuil_cook) &
    (np.abs(dfbetas).max(axis=1) > seuil_dfbeta)
)
print(f"Seuils: levier>{seuil_lev:.4f}, Cook>{seuil_cook:.4f}, dfbeta>{seuil_dfbeta:.4f}")
print("Nb observations influentes:", influentes.sum())

# %% 11. Modèle 2 (sans observations influentes)
df_mod2 = df_mod[~influentes].copy()
Y2 = df_mod2['utilisation_info']
X2 = df_mod2.drop('utilisation_info', axis=1)
X2c = sm.add_constant(X2)

modele2 = sm.Logit(Y2, X2c).fit(disp=0)
print(modele2.summary())

pred2 = modele2.predict()
pred2_class = (pred2 > 0.5).astype(int)
cm2 = confusion_matrix(Y2, pred2_class)
print("\nMatrice de confusion:\n", cm2)
print("Taux classification:", round((cm2[0, 0] + cm2[1, 1]) / cm2.sum() * 100, 1), "%")
print("AUC Modele 2:", roc_auc_score(Y2, pred2))
print("Pseudo R2 Modele 2:", modele2.prsquared)

# %% 12. Odds ratios (Modèle 2)
labels = {
    'const': 'Constante',
    'realisme_percu': 'Réalisme perçu (Réaliste vs Non réaliste)',
    'sex': 'Sexe (Féminin vs Masculin)',
    'agep': 'Âge du chef de ménage (années)',
    'educ': "Niveau d'instruction (années)",
    'alpha': 'Alphabétisation (Alphabétisé vs Non)',
    'smetri_bin': 'Situation matrimoniale (Marié vs Non marié)',
    'group': 'Appartenance groupement (Oui vs Non)',
    'vulg': 'Contact vulgarisation (Oui vs Non)',
    'acred': 'Crédit formel (Oui vs Non)',
    'crefor': 'Crédit informel (Oui vs Non)',
    'supdis': 'Superficie disponible (ha)',
    'parpv': 'Part revenu végétal',
    'parna': 'Part revenu non agricole',
    'tailm': 'Taille du ménage',
    'nactif': "Nombre d'actifs agricoles",
    'poscel': 'Téléphone portable (Oui vs Non)',
    'posradio': 'Radio (Oui vs Non)',
    'postele': 'Télévision (Oui vs Non)',
    'freq_reception_Hebdomadaire': 'Fréquence Hebdomadaire (vs Quotidien)',
    'freq_reception_Mensuel': 'Fréquence Mensuelle (vs Quotidien)',
    'freq_reception_Saisonnier': 'Fréquence Saisonnière (vs Quotidien)',
    'zone_Centre': 'Zone Centre (vs Sud)',
    'zone_Nord': 'Zone Nord (vs Sud)',
}

rows = []
for var in modele2.params.index:
    coef = modele2.params[var]
    se = modele2.bse[var]
    p = modele2.pvalues[var]
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'ns'
    rows.append(['Const' if var == 'const' else var, labels.get(var, var),
                 round(coef, 4), round(se, 4), round(np.exp(coef), 4), f"{p:.4f} {sig}"])

tab7 = pd.DataFrame(rows, columns=[
    'Variable', 'Description', 'Coefficient', 'Erreur standard', 'Odds Ratio', 'p-value'])
print(tab7.to_string(index=False))
