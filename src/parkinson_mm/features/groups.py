"""Groupes de variables. Chaque modalité est un bloc que l'on peut ajouter ou retirer (ablation M1 -> M4)."""

# Variables mesurées à chaque visite (forment les séquences des modèles récurrents)
STEP_CLIN = ["np3", "nhy", "updrs1", "updrs2", "updrs4", "schwab", "tremor_score", "pigd_score", "rigid3",
             "brady3", "asym3", "on_treatment", "med_off", "med_on", "ledd", "moca", "ess", "gds", "scopa_aut",
             "stai", "disease_dur_years", "time_years", "age"]
# Résumé de l'historique et conditions de la visite à prédire (connues à l'avance : protocole)
HISTORY = ["np3_prev", "np3_slope_hist", "np3_mean_hist", "n_prior", "gap_years", "target_off", "target_on",
           "target_untreated"]
DEMO = ["male", "age_bl", "EDUCYRS", "ANYFAMPD"]

DAT = ["sbr_striatum", "sbr_putamen", "sbr_putamen_min", "sbr_caudate", "sbr_put_caud_ratio", "sbr_put_asym",
       "sbr_age_months"]
MRI = ["mri_putamen", "mri_caudate", "mri_pallidum", "mri_thalamus", "mri_hippocampus", "mri_lat_ventricles",
       "mri_brainstem", "mri_subcort_gray", "mri_cortex"]
GEN = ["lrrk2", "gba", "apoe4", "PATHVAR_COUNT", "prs"]
BIO = ["csf_abeta42", "csf_ab42_40", "csf_ptau181", "csf_asyn", "plasma_nfl", "plasma_gfap"]

MODALITIES = {"bio": BIO, "dat": DAT, "mri": MRI, "gen": GEN}
