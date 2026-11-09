# Utilisation de l'information climatique par les producteurs agricoles

Code d'analyse du mémoire : statistiques descriptives, tests bivariés (χ², t de Welch) et régression
logistique de l'utilisation de l'information climatique par les producteurs agricoles
(base producteurs IDID 2013, N = 360).

## Problématique et données

L'analyse étudie les facteurs associés à l'utilisation de l'information climatique : accès aux sources
d'information (10 sources, formelles, informelles et médias), réalisme perçu, fréquence de réception et
caractéristiques socio-économiques des producteurs.

Les données (fichier SPSS `.sav`) ne sont pas versionnées. Placez le fichier dans `data/`
(voir `data/README.md`) ou indiquez son chemin (voir ci-dessous).

## Méthodes

- Statistiques descriptives : accès par source et par catégorie, caractéristiques de l'échantillon
- Tests bivariés : χ² (variables qualitatives) et t de Welch (variables quantitatives)
- Multicolinéarité : facteur d'inflation de la variance (VIF)
- Régression logistique de l'utilisation, avec détection des observations influentes (résidus studentisés,
  levier, distance de Cook, DFBETAS) et ré-estimation sans ces observations
- Odds ratios, matrice de confusion et AUC

## Structure

```
├── run_all.py              # exécute toute l'analyse
├── src/
│   ├── config.py           # chemins, listes de variables, libellés
│   ├── preprocessing.py    # chargement .sav + variables dérivées
│   ├── descriptive.py      # Tableaux 2 et 3, Figure 2
│   └── modeling.py         # VIF, logit, observations influentes, odds ratios
├── data/                   # y placer le .sav (non versionné)
└── outputs/                # tableaux CSV et figure générés
```

## Reproduction

```bash
pip install -r requirements.txt
python run_all.py                      # base dans data/
python run_all.py "chemin/vers/base.sav"
```

Le chemin de la base peut aussi être fourni par la variable d'environnement `MEMOIRE_SAV`.
Les tableaux et la figure sont écrits dans `outputs/`.

## Notes méthodologiques

- Les catégories de sources (formelles, médias, informelles) sont comptées en union : un producteur
  est compté une seule fois par catégorie, quel que soit le nombre de sources auxquelles il a accès.
- Les questions sur l'utilisation, le réalisme perçu et la fréquence de réception ne sont posées qu'aux
  producteurs ayant accès à la source. Les valeurs manquantes correspondantes sont codées 0 (utilisation,
  réalisme), et la fréquence manquante est remplacée par la médiane.
- L'accessibilité est présentée dans les analyses descriptives mais exclue du modèle logit, en raison de
  la quasi-séparation (aucun producteur sans accès n'utilise l'information).
