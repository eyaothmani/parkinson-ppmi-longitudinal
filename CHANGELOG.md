# Changelog

Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/) — versions : [SemVer](https://semver.org/lang/fr/).
Une version par sprint de 2 semaines ; les sprints portent aussi un tag `sprint-N`.

## [Non publié]

## [1.5.0] - Assistant documentaire (RAG), API et tableau de bord
### Ajouté
- **API restaurée** (retirée en ADR-0026) : `/health`, `/state`, `/predict`, `/explain`, `/factors`, clé facultative, limite de débit, journaux sans donnée patient (code et tests de `26345d0^`, inchangés, passent sur le code actuel). Nouvelles routes `GET /summary` (résultats agrégés de `reports/metrics/`, liste blanche) et `POST /rag/ask` (assistant chargé à la première question ; si l'API du LLM échoue, réponse extractive). `make serve`, service `api` de docker compose, `docker/Dockerfile.api`, extra `api`.
- **Tableau de bord** servi par l'API sur `/app/` (`src/parkinson_mm/serving/web/`) : HTML, CSS et JavaScript sans framework, graphiques SVG faits main. Vue d'ensemble (éventail des trajectoires tracé avec la pente moyenne ± un écart-type, trois objectifs, erreurs face aux deux références), score actuel et score futur sur un patient fictif saisi à la main, facteurs (graphique en forêt, filtre FDR, vue tableau), assistant. Identité tulipe rouge, symbole de la maladie de Parkinson (interface rouge, logo tulipe) ; rouge = vers un score plus élevé, bleu = vers un score plus bas, ocre = sans traitement ; palette validée pour le daltonisme et le contraste, en clair et en sombre ; identité par la couleur ET la forme ; infobulles au clavier ; mobile.
- **Assistant documentaire** (`src/parkinson_mm/rag/`, `rag/README.md`) : répond sur 4 articles publiés et le README du projet en citant le document et la page ou la section ; « Je ne trouve pas cela dans le corpus » plutôt qu'inventer ; refus des questions médicales personnelles avant toute recherche. Chaîne : manifeste → extraction (pypdf, bibliographie retirée) → passages (LangChain, 1 000 caractères) → embeddings locaux `multilingual-e5-small` → index FAISS exact → LLM via une API compatible OpenAI (par défaut Groq, `openai/gpt-oss-120b` avec un effort de raisonnement `low`, offre gratuite ; changer de fournisseur = 3 lignes de `params.yaml`) → contrôle des citations (réponse rejetée si une citation manque ou n'existe pas). Sans clé `GROQ_API_KEY` : passages cités (mode extractif). Sans LangGraph : la chaîne est linéaire.
- Garde-fous testés : le manifeste refuse `data/`, `reports/tables/`, `models/`, `mlruns/`, `mlflow.db` et tout tableau avec une colonne PATNO ; le paquet `rag` n'importe rien du pipeline PPMI (test d'imports).
- Étapes DVC `rag_index` (ne dépend que des modules qui construisent l'index : changer de LLM ne recalcule pas les embeddings) et `rag_eval` ; `scripts/build_rag_index.py`, `scripts/eval_rag.py [--llm]`, `scripts/ask_rag.py` ; section `rag` de `params.yaml` ; experiment MLflow `extension_rag` (catalogue) ; 22 questions de référence écrites avant la mesure (`rag/eval/questions.yaml`).
- Consigne v2 de l'assistant après relecture de la v1 (12 réponses justes sur 16, 1 fausse mais citée, 3 manquées) : reprendre les conclusions telles qu'écrites, donner les intervalles, répondre dès qu'un passage répond ; effort `medium`. Jeu de contrôle de 9 questions nouvelles (`rag/eval/questions_controle.yaml`), écrit avant tout essai de la v2 ; `eval_rag.py --questions`. Résultat relu à la main : la v2 n'est pas meilleure dans l'ensemble (10 justes, 4 avec une phrase fausse ou trompeuse, 1 fausse mais citée, 1 manquée ; contrôle : 4 justes sur 7, aucune fausse, 3 manquées) ; gardée sans nouveau réglage.
- Extra `rag` (`pyproject.toml`, `uv.lock`) : langchain-core, langchain-text-splitters, langchain-openai, sentence-transformers, faiss-cpu, pypdf. Aucune bibliothèque numérique existante ne change de version (seul `jsonpointer`, outil de Jupyter, passe de 3.2.0 à 3.12).
### Modifié
- CI et CD installent les extras `rag` et `api` ; l'audit de sécurité couvre toutes les extras.
- README : la conclusion H1 dit que l'écart GRU-D − MLP à 24 mois est non significatif (−0,06 pt [−0,19 ; +0,05]) au lieu de « ne bat pas », qui se lisait comme « fait moins bien ».

## [1.4.0] - Analyses de sensibilité (validation croisée seulement)
### Ajouté
- **4 étapes DVC en CV** (`src/parkinson_mm/analysis/`, `make sensibilite`), runs tracés sur `5077739` ; le test n'est pas relu et aucune étape existante n'est réentraînée.
  - `reference_horizons` : persistance ajustée à l'état à 12, 24 et 36 mois. Gain des modèles face à elle : 0,48–0,55 / 0,63–0,79 / 0,66–0,75 pt (IC 99,4 %, Bonferroni 9 comparaisons).
  - `etat_cv` : score d'état de l'objectif 1 refait pli par pli. MAE 6,46 ; −0,22 pt [−0,37 ; −0,06] face à la persistance ajustée, +0,40 pt [0,31 ; 0,50] face au GRU (IC 99,2 %) ; couverture 80 % : 82,5 %.
  - `sensibilite` : LightGBM, MLP et GRU **sans l'état de l'examen cible** (MAE 6,31–6,35 ; −0,71 à −0,75 pt face à la persistance brute) ; LightGBM et MLP **sans historique** (RQ1 : +0,24 et +0,26 pt face aux modèles complets). 8 comparaisons fixées dans `params.yaml`, IC 99,4 %. Une variable retirée est mise à manquant partout : mêmes plis, mêmes graines.
  - `stats_sensibilite` : temps quadratique préféré (LRT p < 10⁻¹⁵) mais faible courbure (2,34 pt/an la 1re année, 1,84 à 5 ans) ; H&Y ≥ 2 : −0,12 [−0,46 ; +0,22] une fois ajusté sur le score initial ; GBA toujours non significatif, y compris sur les examens non traités ou OFF ; de novo sur 5 ans : 2,35 [2,14 ; 2,57] pt/an, proche des 2,4 de Holden et al. 2018.
- README : sections « Analyses de sensibilité », « Limites » et « Décisions principales » (à la place des ADR retirés).
### Modifié
- README : la persistance ajustée apprend aussi la progression moyenne ; elle ne mesure pas l'effet ON/OFF seul.
- `params.yaml` : commentaire des critères de promotion corrigé (le critère 3 compare la MAE de CV ; le critère 1 est la seule règle qui lit le test).
- `analysis/reference.py` : la variation de repli peut être prise au même horizon (`within`) ; résultats de `reference_etat` inchangés.

## [1.3.0] - Validation 3 : référence juste, traçabilité, CI/CD
### Ajouté
- **Référence juste de l'objectif 2** (`src/parkinson_mm/analysis/`, étape DVC `reference_etat`, `scripts/run_reference_etat.py`) : persistance ajustée à l'état de l'examen cible (variation médiane par transition d'état, apprise sur les autres plis), comparée aux 4 modèles en validation croisée seulement (le test n'est pas relu). Tableaux `reports/tables/reference_etat/` (résumé CV, comparaisons appariées, MAE par transition d'état), run « Persistance ajustée à l'état » dans l'experiment `objectif2_score_futur`. Le dossier `analysis/` n'est une dépendance d'aucune étape d'entraînement ou de test : rien n'est réentraîné.
- `tests/unit/test_notebooks.py` : un notebook n'a des sorties que si `keep_output` est déclaré, et jamais un tableau par patient (colonne ou index PATNO).
- **CD** (`.github/workflows/cd.yml`) : sur un tag `vX.Y.Z`, les tests repassent, l'image d'entraînement est construite, vérifiée (import du paquet) et publiée sur GHCR, puis une release GitHub est créée avec les notes du CHANGELOG. L'image ne contient ni données PPMI, ni modèles, ni `mlflow.db`.
- **Notebook `03_sprint6_robustesse`** : la robustesse du modèle retenu expliquée pas à pas, exécuté ; ses sorties (agrégées seulement) sont gardées malgré `nbstripout` (métadonnée `keep_output`).
- **`rag/` : début de l'assistant documentaire.** Corpus de 4 articles (PPMI, deep learning multimodal et multitâche, voix mPower) avec leurs références dans `rag/README.md` ; PDF suivis par DVC, jamais par Git.
- **Sprint 6 — robustesse du modèle retenu** (validation croisée seulement, le test n'est pas relu) : MAE par sous-groupe (stade H&Y, état de l'examen, durée de maladie, génétique, sexe, âge, période d'inclusion) avec IC bootstrap patient, et couverture de l'intervalle à 80 % par prédiction conforme pli par pli. Étape DVC `robustesse`, `scripts/run_robustesse.py`, experiment MLflow `objectif2_robustesse`, figures dans `reports/figures/robustesse/`.
- **Modèle hybride** (ADR-0024, validation croisée seulement, protocole écrit avant le calcul) : le MLP et le GRU-D reçoivent le niveau et la vitesse du modèle mixte de l'objectif 1, réajusté à chaque pli sur ses seuls patients d'entraînement ; référence « Modèle mixte seul ». H-HYB **non confirmée** à 36 mois (MLP −0,01 [−0,04 ; +0,03], GRU-D +0,02 [−0,01 ; +0,05], IC 97,5 %). Les réseaux battent le modèle mixte seul de 0,4 à 0,7 point selon l'horizon. Étape DVC `hybride`, `scripts/run_hybride.py`, experiment `objectif2_hybride`.
- **Dashboard d'exploration scientifique** (ADR-0023, révise l'ADR-0014) : Streamlit local (127.0.0.1, statistiques d'usage coupées), en lecture seule. Phase 1 : Accueil et « Cohorte & données » : du fichier PPMI au jeu modélisé, Tableau 1 dev / test avec différences standardisées, suivi et attrition, MDS-UPDRS III à l'inclusion, évolution moyenne par état de mesure (IC 95 %), trajectoires sous identifiants anonymes, manquants par variable et par modalité. `make dashboard`, extra `ui` ; palette commune déplacée dans `utils/palette.py`.
- **Organisation MLflow** (ADR-0022) : un experiment = des runs qu'on compare entre eux. Runs TEST dans `<question>_test` (`evaluation` renommé `objectif2_score_futur_test`, nouvel `objectif2_multi_horizon_test`), chaîne M1 → M4 dans `objectif2_multimodal_chaine` (M1 compris) ; description et tags `objectif`, `contenu`, `reference` de chaque experiment depuis `docs/mlflow_experiments.yaml`. `tools/organize_mlflow.py` range la base existante (sauvegarde d'abord, idempotent) sans changer identifiants de runs, métriques, artefacts ni registry.
- **Modalités aux horizons longs** (ADR-0021, validation croisée seulement, protocole écrit avant le calcul) : MLP, GRU-D et LightGBM sans / avec 7 variables d'imagerie, de biologie et de génétique. H-MM **non confirmée** à 36 mois (MLP −0,05 [−0,13 ; +0,03], GRU-D −0,06 [−0,14 ; +0,02], IC 97,5 %). Amendement exploratoire : chaîne M1 → M4 du GRU-D, aucun gain mesurable. Étape DVC `multimodal`, `scripts/run_multimodal.py [--chaine]`.
- **Suivi MLflow en ligne sur DagsHub, par copie filtrée** (ADR-0020) : `tools/publish_dagshub.py` recopie les runs actifs (paramètres, métriques, tags, datasets déclarés, lien parent → pli, dates) vers un dépôt DagsHub privé. Artefacts par **liste blanche** de 4 tableaux agrégés, refusés s'ils contiennent un identifiant ou dépassent 200 ko ; prédictions par patient, modèles et registry restent locaux. Relançable sans doublon (`source_run_id`), `--dry-run` sans connexion, `--sans-plis` et `--prefixe colab/` pour les runs de Colab ; section 6 bis du carnet Colab ; guide `docs/dagshub.md`.
### Modifié
- README : liens vers `docs/` et `colab/` retirés ; traçabilité corrigée (commit des runs de chaque objectif) ; objectif 2 présenté comme une prédiction conditionnelle à l'état ON/OFF de l'examen cible ; « test utilisé une seule fois » remplacé par « le test ne sert jamais à choisir un modèle ».
- Notebook `01_EDA` : une seule version (celle du 07/11, avec Wilcoxon et ICC), section 5.1 « Distribution à l'inclusion » rétablie ; `01_EDA` et `02_nettoyage` gardent leurs sorties agrégées (`keep_output`).
- `docker/docker-compose.yml` : `.env` devient facultatif ; sans lui, `docker compose run train` échouait (« env file not found »).
- Notebooks `01_EDA` et `02_nettoyage` réécrits : plus courts, chaque section suit « observation → décision → conséquence pour les modèles ». `02_nettoyage` lit les sorties du pipeline sans rien recalculer et explique la séparation par patient (SMD), la construction des variables sans fuite, la sélection de variables a priori et le prétraitement appris sur le train. `01_EDA_court` supprimé (remplacé par `01_EDA`).
- `training/tracking.py` refuse une URI de suivi `http(s)://` : le pipeline écrit toujours dans la base locale.
- Promotion (ADR-0004) : le critère 3 compare la MAE de validation croisée (tag `cv_MAE`) à celle du champion, au lieu de la MAE de test. Modèles mixtes : avertissement et colonne `converge` en cas de non-convergence. Nouvelle référence « persistance corrigée (condition) » ; notebook 07 (analyses de sensibilité).

### Corrigé
- **Résultats relancés depuis `67f17af`** (calcul dans un environnement distant, données traitées seulement, effacées ensuite) : objectif 3 et robustesse identiques aux runs précédents (pente 1,89 pt/an, 18 / 31 facteurs, 31 modèles convergés ; couverture 80,1 %). Référence ajustée (CV) : 6,67 contre 7,06 pour la persistance brute et 6,05 pour le GRU ; gain réel des modèles −0,62 pt [−0,75 ; −0,50].
- **Traçabilité** : « +modifs » ne signale plus que du code non commité ; les résultats que les étapes écrivent elles-mêmes (`reports/metrics/`, `dvc.lock`) ne comptent plus. Avant, un run lancé par `dvc repro` était toujours marqué « +modifs » (DVC efface la sortie avant de relancer l'étape). Tag `sprint-4-origine` sur `c681fbf` : le commit des runs de l'objectif 1 reste dans l'historique.
- **Environnement** : `mlflow.db` est au schéma de MLflow 3.18, que le `uv.lock` (3.17.0) ne pouvait plus ouvrir. MLflow passe à 3.18.0 (`pyproject` : `mlflow>=3.18`) ; le service `mlflow` de docker compose est construit depuis l'image du projet (même version que `uv.lock`). Paquets non numériques mis à jour pour les vulnérabilités connues (pillow, filelock, pyasn1, certifi, setuptools, gitpython, opentelemetry) : pip-audit sans alerte sous Python ≥ 3.11. numpy, pandas, scipy, statsmodels, scikit-learn, LightGBM et torch sont inchangés.
### Retiré
- `docs/` (sauf `docs/mlflow_experiments.yaml`, lu par le code), `colab/`, `CITATION.cff` et la copie `notebooks/01_EDA (1).ipynb`.
- **Simplification de la version encadrant** (ADR-0026) : API FastAPI (+ `Dockerfile.api`, `cd.yml`, extra `api`), dashboard Streamlit (+ extra `ui`), notebooks 03 à 07, analyses complémentaires multimodal (ADR-0021) et hybride (ADR-0024) avec leurs étapes DVC, persistance corrigée de l'audit, article LaTeX, modèles de tickets GitHub. Tout est rangé dans `archive/retires-2026-11-06/` ; aucun résultat des objectifs 1 à 3 ni du sprint 5 ne change.
- Publication DagsHub (ADR-0020 passé au statut « retiré ») : `publication/`, `tools/publish_dagshub.py`, `docs/dagshub.md`, section 6 bis du carnet Colab et leurs tests. Non demandée par le sujet ; le suivi MLflow reste local et le pipeline refuse toujours un serveur distant.
- Outils utilisés une seule fois sur la base MLflow existante : `tools/backfill_datasets.py` (ADR-0019) et `tools/organize_mlflow.py` avec `training/experiment_moves.py` (ADR-0022). Leur travail est fait ; le code reste dans l'historique Git.

## [1.2.0] - Sprint 5 : deep learning central, GRU-D multi-horizon
### Ajouté
- **GRU-D multi-horizon** (ADR-0018) : MDS-UPDRS III à 12, 24 et 36 mois (mesure la plus proche, ±3 mois ; 9 160 / 6 866 / 5 217 paires), masques et temps écoulé par variable, quantiles 10/50/90 %, multitâche (UPDRS II, MoCA). Comparé à la persistance, à LightGBM (un par horizon) et au MLP dernière visite ; réglage à budget égal (recherche aléatoire, 20 essais, plis 0 et 1) ; ablations sur la validation croisée ; test unique sur 5 graines après le tag `gel-multi-horizon`.
- Résultat test à 24 mois : GRU-D 7,37 [6,93 ; 7,82], MLP 7,44, LightGBM 7,56, persistance 8,94. H1 (GRU-D < MLP) **non confirmée** (−0,06 [−0,19 ; +0,05]) ; H4 confirmée (couverture 78,4 %) ; H5 : ρ = 0,18 [0,07 ; 0,30] avec la pente du modèle mixte.
- Étapes DVC `horizons`, `train_horizons`, `evaluate_horizons` ; `scripts/run_horizons.py` ; experiment MLflow `objectif2_multi_horizon` ; registered model `parkinson-multi-horizon-updrs3` (une version et un alias par modèle, MLP `champion`) ; contrat de données `horizons` ; notebook 06.
- **Colonne Dataset de MLflow remplie** (ADR-0019) : chaque run déclare ses datasets (`training`, `validation`, `test`) avec empreinte, schéma et md5 DVC, sans copier de données ; `tools/backfill_datasets.py` a complété les 30 runs existants sans toucher à leurs métriques.
- Registry : les **quatre modèles comparés** y sont enregistrés (versions 2 à 5), chacun avec un run `TEST – <modèle>` et un alias propre (`persistance`, `lightgbm`, `mlp`, `gru`) ; seul le modèle retenu peut être `champion` ; réévaluer avec les mêmes données et la même configuration ne crée aucune version (ADR-0017).
- Article (`docs/paper`) et documents des sprints 0 et 3 réécrits pour les quatre modèles ; l'article garde un paragraphe sur la première version à 13 modèles (ADR-0015).
### Modifié
- `make_tabular` accepte des réglages (aucun pour l'objectif 2) ; `training/tracking.py` séparé en ouverture de run (`tracking.py`) et écriture dans un run (`run_log.py`).
- `dvc.lock` : empreintes du code mises à jour ; sorties des objectifs 1 à 3 inchangées (relancées à l'identique).

## [1.1.0] - Version encadrant : quatre modèles, limites corrigées
### Ajouté
- Notebook 05 réécrit pour les 4 modèles : comparaisons appariées, intervalle par horizon, importance par modèle, explication locale ; il n'entraîne plus jamais rien.
- Comparaisons appariées entre modèles appris, en CV et sur le test, IC corrigé de Bonferroni (`*_comparaisons_appariees.csv`).
- Importance par permutation pour chacun des 4 modèles (`importance_par_modele.csv`).
- Intervalle de prédiction à 80 % par horizon (0-6, 6-12, 12-24 mois) et couverture du test par horizon (`test_couverture_horizon.csv`).
- Explication locale par groupe de variables et route `POST /explain` (ADR-0016).
- Registry : version 2 = GRU trajectoire (M1), champion ; version 1 (fusion M4) retirée avec un tag `statut`.
### Modifié
- Objectif 2 : **4 modèles comparés** sur les mêmes variables cliniques (persistance, LightGBM, MLP, GRU), ADR-0015. Un seul experiment MLflow `parkinson-mm/objectif2_score_futur` (4 runs, 20 plis) ; les 54 runs des 9 autres expériences sont archivés (supprimés de l'interface, restaurables).
### Retiré
- Notebook 00 (environnement), remplacé par `hardware_report()` ; notebook 01 allégé (détail DaTSCAN / IRM / génétique / modalités complémentaires / capteurs remplacé par une synthèse des décisions D11–D22).
- Expériences tendance linéaire, Ridge, Transformer, ablations M2–M4 et fusions précoce / intermédiaire / tardive (consultables au tag `v1.0.0`).

## [1.0.0] - Sprint 4 (semaines 7-8) : valorisation et article
### Ajouté
- Article LaTeX rédigé en entier (`docs/paper/main.tex`) : introduction, données et méthodes, résultats des trois objectifs, RQ1 et RQ2, discussion, limites, remerciements PPMI et références.
- Section sur la RQ3 (signaux smartphone, mPower non retenu) et sur le choix d'infrastructure (Spark non nécessaire au volume PPMI).
- Trois tableaux générés en plus par `scripts/export_paper_tables.py` : journal de nettoyage, chaîne M1 → M4, analyses de sensibilité.
- Figures de l'article lues dans `reports/figures/` lorsqu'elles existent (l'article compile aussi sans elles).
- API FastAPI des trois objectifs : `POST /state` (objectif 1), `POST /predict` (objectif 2), `GET /factors` (objectif 3), plus `/health`, `/model-info`, `/metrics` ; clé d'API, limitation de débit, schémas stricts, avertissement non diagnostique.
- Monitoring léger : latence, erreurs, dérive des entrées (PSI).
- Image Docker de l'API, `docker-compose` complet, CD sur tag `vX.Y.Z`, scan d'image Trivy.
- ADR-0009, 0011, 0014 (interface : API et Swagger, sans dashboard).

## [0.3.0] - Sprint 3 (semaines 5-6) : objectif 2, score futur
### Ajouté
- Treize expériences déclarées dans `params.yaml` : persistance, tendance linéaire, Ridge, LightGBM, MLP, GRU, Transformer, fusions précoce / intermédiaire / tardive.
- Validation croisée groupée par patient (5 plis), un run MLflow par modèle et un run imbriqué par pli.
- Évaluation unique sur le test, IC bootstrap par patient, prédicteur empaqueté (pyfunc), Model Registry avec promotion objective `candidate` / `champion`.
- Importance par permutation de groupes de variables (modalités).
- Notebook 05, model card, guide de reproductibilité, article LaTeX généré depuis `reports/`, image Docker d'entraînement.
- ADR-0004, 0007, 0012.

## [0.2.0] - Sprint 2 (semaines 3-4) : objectifs 3 et 1
### Ajouté
- Modèles linéaires à effets mixtes, criblage pré-spécifié de 31 facteurs avec correction de Benjamini-Hochberg, chaîne emboîtée M1 → M4 (objectif 3).
- Score d'état actuel : niveau sans traitement + vitesse par BLUP filtré, validé sur les patients de test contre la persistance (objectif 1).
- Suivi MLflow (backend SQLite) avec tags de traçabilité : `objectif3_facteurs`, `objectif1_score_actuel`.
- Notebooks 03 et 04, `docs/objectifs.md`, model card du score d'état, ADR-0003.

## [0.1.0] - Sprint 1 (semaines 1-2) : données
### Ajouté
- Package `parkinson_mm` en couches, fichiers ≤ 60 lignes ; `params.yaml` seule source des paramètres ; `dvc.yaml` (ingest, validate, split, features).
- Lecture des CSV PPMI, nettoyage journalisé (22 décisions D1–D22), contrats de données (brut, nettoyé, échantillons).
- Découpage au niveau patient avant toute feature ; échantillons « visite i → visite i+1 » sans fuite.
- Jeu synthétique au schéma PPMI pour la CI ; tests unitaires et d'intégration ; CI (lint, tests), pre-commit, gitleaks.
- Notebooks 00 (environnement GPU/CPU), 01 (EDA), 02 (nettoyage) ; data card, architecture, ADR-0001, 0002, 0005, 0006, 0008, 0010.
