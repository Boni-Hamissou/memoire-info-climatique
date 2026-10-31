# Utilisation de l'information climatique par les producteurs agricoles

Code d'analyse du mémoire : statistiques descriptives, tests bivariés (χ², t de Welch) et régression logistique
de l'utilisation de l'information climatique (base producteurs IDID 2013, N = 360).

## Structure

```
├── run_all.py              # exécute toute l'analyse
├── src/
│   ├── config.py           # chemins, listes de variables, libellés
│   ├── preprocessing.py    # chargement .sav + variables dérivées
│   ├── descriptive.py      # Tableaux 2 et 3, Figure 2
│   └── modeling.py         # VIF, logit, observations influentes, odds ratios
├── notebooks/Memoire2.ipynb        # notebook d'origine (inchangé)
├── extraction/
│   ├── memoire_code_complet.py     # extraction intégrale des 17 cellules, sans modification
│   └── ORDRE_EXECUTION.md          # ordre d'exécution réel dans Jupyter
├── data/                   # y placer le .sav (non versionné)
└── outputs/                # tableaux CSV et figure générés
```

## Utilisation

```bash
pip install -r requirements.txt
python run_all.py            # ou : python run_all.py "chemin/vers/base.sav"
```

## Correspondance notebook → modules

| Notebook (cellule) | Module |
|---|---|
| 0, 1, 7 | `preprocessing.construire_variables` |
| 2, 6, 8 | `descriptive.tableau2_*`, `figure2_acces` |
| 3, 4, 9, 10, 11 | `descriptive.tableau3_quali`, `tableau3_quanti` |
| 12 | `modeling.calcul_vif` |
| 13, 14, 15 | `modeling.ajuster_logit`, `observations_influentes`, `performance` |
| 5 | `modeling.tableau_odds_ratios` |

## Remarques de reproductibilité

- La cellule 2 du notebook additionne les N par source pour les catégories (Médias 88,9 %, Formel 78,6 %) :
  c'est un cumul qui compte plusieurs fois les mêmes producteurs. La cellule 6 le corrige par une union
  (Médias 57,2 %, Formel 46,1 %) ; seule cette version est reprise dans `src/`.
- Les cellules du notebook ont été exécutées dans le désordre (voir `extraction/ORDRE_EXECUTION.md`).
