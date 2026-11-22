# Analyse longitudinale multimodale de la maladie de Parkinson (cohorte PPMI)

Étude observationnelle rétrospective sur les données de la cohorte **PPMI** : comment évolue la sévérité motrice
(**MDS-UPDRS III**, de 0 à 132 points), et quels facteurs cliniques, biologiques, d'imagerie et génétiques sont
**associés** à cette évolution ?

> **Projet de recherche, pas un outil de diagnostic.** Les résultats sont des associations statistiques, pas des
> relations causales. Ils ne doivent guider aucune décision clinique individuelle.

## Objectifs et résultats

Cohorte : 1 741 patients Parkinson, 13 206 visites, séparés par patient (développement 1 209 / test 303).

| Objectif | Méthode | Résultat principal |
|---|---|---|
| **1 · Score actuel** | modèle mixte (niveau et vitesse propres à chaque patient) | −0,85 pt d'erreur face à la persistance (test) ; intervalle à 80 % bien calibré (couverture 82,5 %) |
| **2 · Score à la visite suivante** | 4 modèles comparés : persistance, LightGBM, MLP, GRU | MAE 6,09 (GRU) contre 7,30 pour la persistance (test) ; gain réel de 0,62 pt face à la persistance ajustée à l'état ON/OFF (validation croisée) |
| **3 · Facteurs de progression** | un modèle mixte par facteur (temps × facteur), correction FDR | +1,89 pt/an en moyenne ; 18 facteurs sur 31 associés à la vitesse ; 2,35 pt/an chez les patients non traités à l'inclusion, proche des 2,4 publiés (Holden et al. 2018) |

* **Une référence juste** : les modèles connaissent l'état ON/OFF de l'examen à prédire, la persistance brute non.
  La comparaison principale se fait donc contre une persistance ajustée à cet état.
* **Deep learning** : GRU, MLP et LightGBM font jeu égal. À 24 mois, le GRU-D n'est pas significativement meilleur
  que le MLP (−0,06 pt [−0,19 ; +0,05]).
* **Forme de la progression** : un modèle quadratique est légèrement préféré (2,34 pt/an la 1re année, 1,84 à 5 ans) ;
  la pente linéaire reste un bon résumé moyen.

**Limites** : pas de validation externe (une seule cohorte de recherche) ; données manquantes supposées MAR dans les
modèles mixtes ; prédiction conditionnelle à l'état ON/OFF de l'examen cible.

## Données

Les données PPMI sont soumises à un Data Use Agreement : elles ne sont **pas** dans ce dépôt. L'accès se demande sur
[ppmi-info.org](https://www.ppmi-info.org). Le dossier `data/synthetic/` contient un jeu **fictif** au même schéma,
utilisé par les tests et la CI.

## Stack

Python 3.10 · pandas · statsmodels · scikit-learn · LightGBM · PyTorch · MLflow · DVC · FastAPI ·
LangChain + FAISS · Docker · GitHub Actions

## Démarrage rapide

```bash
pip install -e ".[api,rag,notebooks,dev]"
python scripts/make_synthetic.py --out data/synthetic   # jeu fictif, aucune donnée réelle
pytest
```

Avec les données PPMI dans `data/raw/` :

```bash
dvc repro explain   # pipeline des objectifs 1 à 3
make serve          # API + tableau de bord : http://localhost:8000 (Swagger : /docs)
make mlflow         # résultats tracés : http://localhost:5000
```

## Assistant documentaire (RAG)

Extension séparée de l'analyse : il répond aux questions sur 4 articles publiés et sur ce README, en citant ses
sources, et refuse les questions médicales personnelles. Il ne lit aucune donnée patient.
Détails : [`rag/README.md`](rag/README.md).

## Organisation

```text
src/parkinson_mm/   code : data, features, stats, models, training, evaluation, serving, rag
scripts/            points d'entrée (une étape = un script)
notebooks/          01 EDA · 02 nettoyage · 03 robustesse
tests/              tests unitaires et d'intégration (données synthétiques)
params.yaml         paramètres du projet
dvc.yaml            pipeline reproductible
docker/ .github/    Docker, CI/CD
```

## Licence et remerciements

Code sous licence MIT. *Data used in the preparation of this article were obtained from the Parkinson's Progression
Markers Initiative (PPMI) database (www.ppmi-info.org/access-data-specimens/download-data).*

**Auteure** : Eya Othmani
