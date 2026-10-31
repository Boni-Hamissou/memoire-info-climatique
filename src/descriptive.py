"""Statistiques descriptives et tests bivariés (Tableaux 2 et 3, Figure 2)."""
import pandas as pd
from scipy import stats

from .config import SOURCES_LABELS, CAT_MAP, VARS_QUALI, VARS_QUANTI


def signif(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else 'ns'


def tableau2_sources(df):
    """Niveau d'accès par source."""
    rows = []
    for suffix, (label, cat) in SOURCES_LABELS.items():
        n = int((df['s' + suffix] == 1).sum())
        rows.append({'Source': label, 'Catégorie': cat, 'N avec accès': n,
                     '% accès': round(n / len(df) * 100, 1)})
    tab = pd.DataFrame(rows).sort_values('% accès', ascending=False).reset_index(drop=True)
    tab['Rang'] = tab.index + 1
    return tab


def tableau2_categories(df):
    """Accès par catégorie : UNION des sources (any), et non somme des N par source."""
    rows = []
    for cat, cols in CAT_MAP.items():
        n = int((df[cols] == 1).any(axis=1).sum())
        rows.append({'Catégorie': cat, 'N avec accès': n, '% accès': round(n / len(df) * 100, 1)})
    return pd.DataFrame(rows)


def bloc_quali(df, var, order, mod_labels, label):
    """Khi-deux entre `var` et utilisation_info ; lignes du Tableau 3."""
    tab = pd.crosstab(df[var], df['utilisation_info'])
    chi2, p, _, _ = stats.chi2_contingency(tab)
    lignes = []
    for i, k in enumerate(order):
        if k not in tab.index:
            continue
        n_mod = int(tab.loc[k].sum())
        n_oui = int(tab.loc[k, 1]) if 1 in tab.columns else 0
        n_non = int(tab.loc[k, 0]) if 0 in tab.columns else 0
        lignes.append({
            'Caractéristiques': label if not lignes else '',
            'Modalité': mod_labels[k],
            'Oui (%)': round(n_oui / n_mod * 100, 1),
            'Non (%)': round(n_non / n_mod * 100, 1),
            'Ensemble (%)': round(n_mod / len(df) * 100, 1),
            'Test': 'Chi²' if not lignes else '',
            'Statistique (χ²)': round(chi2, 3) if not lignes else None,
            'p-value': f"{p:.4f} {signif(p)}" if not lignes else '',
        })
    return lignes


def tableau3_quali(df):
    rows = []
    for variables in VARS_QUALI.values():
        for label, var, order, mods in variables:
            rows += bloc_quali(df, var, order, mods, label)
    return pd.DataFrame(rows)


def tableau3_quanti(df):
    """Test t de Welch (utilisateurs vs non-utilisateurs)."""
    rows = []
    for var, label in VARS_QUANTI:
        g0 = df.loc[df['utilisation_info'] == 0, var].dropna()
        g1 = df.loc[df['utilisation_info'] == 1, var].dropna()
        g = df[var].dropna()
        t, p = stats.ttest_ind(g1, g0, equal_var=False)
        rows.append([label, f"{g1.mean():.2f} ({g1.std():.2f})", f"{g0.mean():.2f} ({g0.std():.2f})",
                     f"{g.mean():.2f} ({g.std():.2f})", 't Welch', round(t, 3), f"{p:.4f} {signif(p)}"])
    return pd.DataFrame(rows, columns=['Variable', 'Oui (moy±ET)', 'Non (moy±ET)',
                                       'Ensemble (moy±ET)', 'Test', 'Statistique (t)', 'p-value'])


def figure2_acces(df, chemin_sortie):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    tab2, tab_cat = tableau2_sources(df), tableau2_categories(df).sort_values('% accès', ascending=False)
    couleurs = {'Formel/institutionnel': '#4472C4', 'Médias': '#2E75B6', 'Informel/social': '#ED7D31'}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].barh(tab2['Source'][::-1], tab2['% accès'][::-1], color=[couleurs[c] for c in tab2['Catégorie']][::-1])
    axes[0].set_xlabel("Taux d'accès (%)")
    for i, v in enumerate(tab2['% accès'][::-1]):
        axes[0].text(v + 1, i, f'{v}%', va='center', fontsize=9)

    axes[1].bar(tab_cat['Catégorie'], tab_cat['% accès'], color=[couleurs[c] for c in tab_cat['Catégorie']])
    axes[1].set_ylabel("Taux d'accès (%)")
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(tab_cat['% accès']):
        axes[1].text(i, v + 2, f'{v}%', ha='center', fontsize=10, fontweight='bold')

    fig.legend(handles=[mpatches.Patch(color=c, label=l) for l, c in couleurs.items()],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig(chemin_sortie, dpi=150, bbox_inches='tight')
    plt.close(fig)
