"""Où trouver chaque table PPMI. On cherche par motif : LONI ajoute la date de téléchargement au nom."""

REQUIRED = {
    "status": "Participant_Status_*.csv",
    "demographics": "Demographics_*.csv",
    "age": "Age_at_visit_*.csv",
    "socio": "Socio-Economics_*.csv",
    "family": "Family_History_*.csv",
    "updrs1": "MDS-UPDRS_Part_I_[0-9]*.csv",
    "updrs1p": "MDS-UPDRS_Part_I_Patient_Questionnaire_*.csv",
    "updrs2": "MDS_UPDRS_Part_II__Patient_Questionnaire_*.csv",
    "updrs3": "MDS-UPDRS_Part_III_*.csv",
    "updrs4": "MDS-UPDRS_Part_IV__Motor_Complications_*.csv",
    "schwab": "Modified_Schwab___England_*.csv",
    "datscan": "Xing_Core_Lab_-_Quant_SBR_*.csv",
    "mri_aseg": "FS7_ASEG_VOL_*.csv",
    "genetics": "iu_genetic_consensus_*.csv",
    "prs": "Polygenic_Risk_Scores_*.csv",
}

# Tables complémentaires : utilisées si présentes, jamais bloquantes.
OPTIONAL = {
    "biospecimen": "Current_Biospecimen_Analysis_Results_[0-9]*.csv",
    "pd_dx_history": "PD_Diagnosis_History_*.csv",
    "ledd": "LEDD_Concomitant_Medication_Log_*.csv",
    "moca": "Montreal_Cognitive_Assessment*.csv",
    "ess": "Epworth_Sleepiness_Scale_*.csv",
    "gds": "Geriatric_Depression_Scale*.csv",
    "scopa": "SCOPA-AUT_*.csv",
    "stai": "State-Trait_Anxiety_Inventory*.csv",
}

ALL = {**REQUIRED, **OPTIONAL}
IGNORED_DIRS = {"_to_delete", "_archive"}
