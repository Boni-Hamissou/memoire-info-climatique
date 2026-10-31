"""Exécute l'ensemble de l'analyse et écrit les résultats dans outputs/.

Usage :
    python run_all.py                     # lit data/Base Producteurs IDID 2013 V3.sav
    python run_all.py chemin/vers/base.sav
"""
import sys
from src.config import DOSSIER_SORTIES, FICHIER_SAV
from src.preprocessing import charger_base, construire_variables
from src import descriptive as d, modeling as m


def main(chemin=FICHIER_SAV):
    DOSSIER_SORTIES.mkdir(exist_ok=True)
    df, _ = charger_base(chemin)
    df = construire_variables(df)
    n = len(df)
    print(f"N = {n} | utilisation_info : {df['utilisation_info'].value_counts().to_dict()}")

    # --- Descriptif ---
    tab2, tab2c = d.tableau2_sources(df), d.tableau2_categories(df)
    t3q, t3n = d.tableau3_quali(df), d.tableau3_quanti(df)
    print(tab2.to_string(index=False)); print(tab2c.to_string(index=False))
    tab2.to_csv(DOSSIER_SORTIES / "tableau2_sources.csv", index=False)
    tab2c.to_csv(DOSSIER_SORTIES / "tableau2_categories.csv", index=False)
    t3q.to_csv(DOSSIER_SORTIES / "tableau3_qualitatif.csv", index=False)
    t3n.to_csv(DOSSIER_SORTIES / "tableau3_quantitatif.csv", index=False)
    d.figure2_acces(df, DOSSIER_SORTIES / "Figure2_Acces_Sources.png")

    # --- Modélisation ---
    df_cand = m.preparer_candidats(df)
    vif = m.calcul_vif(df_cand)
    print("\nVIF :\n", vif.to_string(index=False))
    vif.to_csv(DOSSIER_SORTIES / "vif.csv", index=False)

    df_mod = m.matrice_modele(df_cand)
    modele1, Y1, X1c = m.ajuster_logit(df_mod)
    print(modele1.summary())
    print("Pseudo R2 M1:", modele1.prsquared)

    infl, seuils = m.observations_influentes(modele1, X1c)
    print("Seuils:", seuils, "| observations influentes:", int(infl.sum()))

    modele2, Y2, _ = m.ajuster_logit(df_mod[~infl])
    print(modele2.summary())
    perf = m.performance(modele2, Y2)
    print("Matrice de confusion:\n", perf['matrice_confusion'])
    print("Classification:", perf['taux_classification'], "% | AUC:", perf['auc'], "| Pseudo R2:", perf['pseudo_r2'])

    tab7 = m.tableau_odds_ratios(modele2)
    print(tab7.to_string(index=False))
    tab7.to_csv(DOSSIER_SORTIES / "tableau7_odds_ratios.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else FICHIER_SAV)
