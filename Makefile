# Commandes du projet. Sous Windows : utiliser Git Bash, WSL, ou lancer directement les commandes python indiquées.
PY ?= python
.PHONY: setup data synthetic stats state train train-fast eval explain robustesse reference sensibilite all test test-unit test-integration lint docker mlflow serve

setup:            ## environnement de développement + hooks
	$(PY) -m pip install -e ".[notebooks,dev]" && pre-commit install

synthetic:        ## jeu synthétique au schéma PPMI (CI / démo)
	$(PY) scripts/make_synthetic.py --out data/synthetic

data:             ## ingest -> validate -> split -> features
	$(PY) scripts/prepare_data.py

stats:            ## objectif 3 : facteurs associés à la vitesse d'évolution
	$(PY) scripts/run_stats.py

state:            ## objectif 1 : score d'état actuel (modèle mixte)
	$(PY) scripts/run_state.py

train:
	$(PY) scripts/train.py

train-fast:
	$(PY) scripts/train.py --fast

eval:
	$(PY) scripts/evaluate.py && $(PY) scripts/register.py

explain:
	$(PY) scripts/explain.py

robustesse:       ## sprint 6 : erreurs par sous-groupe + intervalle à 80 % (quelques secondes, CV seulement)
	$(PY) scripts/run_robustesse.py

reference:        ## objectif 2 : persistance ajustée à l'état, la référence juste (quelques secondes, CV seulement)
	$(PY) scripts/run_reference_etat.py

sensibilite:      ## phase 3 : référence ajustée multi-horizon, objectif 1 en CV, variantes des modèles, stats
	$(PY) scripts/run_reference_horizons.py && $(PY) scripts/run_etat_cv.py
	$(PY) scripts/run_sensibilite.py && $(PY) scripts/run_stats_sensibilite.py

all:              ## pipeline complet, reproductible
	dvc repro

test:
	$(PY) -m pytest --cov --cov-report=term-missing

test-unit:
	$(PY) -m pytest tests/unit -q

test-integration:
	$(PY) -m pytest tests/integration -q

lint:
	ruff check src scripts tests tools && $(PY) tools/check_file_length.py

mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

serve:            ## API + tableau de bord : http://localhost:8000/ (Swagger : /docs)
	uvicorn parkinson_mm.serving.api:app --host 127.0.0.1 --port 8000

docker:
	docker compose -f docker/docker-compose.yml up --build
