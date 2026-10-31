"""Configuration : chemins, listes de variables et libellés."""
import os
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
# Chemin du fichier SPSS : variable d'environnement ou data/ par défaut
FICHIER_SAV = Path(os.environ.get("MEMOIRE_SAV", RACINE / "data" / "Base Producteurs IDID 2013 V3.sav"))
DOSSIER_SORTIES = RACINE / "outputs"

# 10 sources : les 8 d'origine + Internet + Journaux
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

# Regroupement des colonnes d'accès ('s' + source) par catégorie
CAT_MAP = {
    'Formel/institutionnel': ['scecp', 'song', 'scoop', 'sfour'],
    'Médias':                ['sradio', 'stele', 'sfone', 'sint', 'sjour'],
    'Informel/social':       ['svois'],
}

ZONE_MAP = {1: 'Nord', 5: 'Nord', 3: 'Centre', 2: 'Sud', 4: 'Sud', 6: 'Sud'}
FREQ_LABELS = {1: 'Quotidien', 2: 'Hebdomadaire', 3: 'Mensuel', 4: 'Saisonnier'}

# Tableau 3 : variables qualitatives (section, libellé, variable, ordre, modalités)
OUI_NON = {0: 'Non', 1: 'Oui'}
VARS_QUALI = {
    "Caractéristiques de l'information climatique": [
        ('Réalisme perçu', 'realisme_percu', [0, 1], {0: 'Non réaliste', 1: 'Réaliste'}),
        ('Accessibilité de la source', 'accessibilite', [0, 1], {0: 'Sans accès', 1: 'Avec accès'}),
        ('Fréquence de réception', 'freq_reception', [1, 2, 3, 4], FREQ_LABELS),
    ],
    "Caractéristiques socio-économiques du producteur": [
        ('Sexe', 'sex', [0, 1], {0: 'Masculin', 1: 'Féminin'}),
        ('Alphabétisation', 'alpha', [0, 1], OUI_NON),
        ("Possession d'un téléphone", 'poscel', [0, 1], OUI_NON),
        ("Possession d'une radio", 'posradio', [0, 1], OUI_NON),
        ("Possession d'une télévision", 'postele', [0, 1], OUI_NON),
        ('Accès au crédit formel', 'acred', [0, 1], OUI_NON),
        ('Accès au crédit informel', 'crefor', [0, 1], OUI_NON),
        ('Contact avec la vulgarisation', 'vulg', [0, 1], OUI_NON),
        ("Membre d'un groupement/coopérative", 'group', [0, 1], OUI_NON),
        ('A des proches dans le village', 'proch', [0, 1], OUI_NON),
        ('Situation matrimoniale (marié)', 'smetri_bin', [0, 1], OUI_NON),
        ('Connaissance du changement climatique', 'infocc', [0, 1], OUI_NON),
    ],
}

VARS_QUANTI = [
    ('agep',   'Âge (années)'),
    ('educ',   "Niveau d'instruction (années)"),
    ('supdis', 'Superficie disponible (ha)'),
    ('tailm',  'Taille du ménage'),
    ('nactif', "Nombre d'actifs agricoles"),
    ('parna',  'Part revenu non agricole (Sur 10)'),
    ('parpv',  'Part revenu végétal (Sur 10)'),
]

# Variables candidates du modèle logit
VARS_CANDIDATES = ['utilisation_info', 'realisme_percu', 'freq_reception',
    'sex', 'agep', 'educ', 'alpha', 'smetri_bin', 'group', 'vulg',
    'acred', 'crefor', 'supdis', 'parpv', 'parna', 'tailm', 'nactif',
    'poscel', 'posradio', 'postele', 'zone']

LABELS_MODELE = {
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
