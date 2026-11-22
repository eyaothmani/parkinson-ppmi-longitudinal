# Assistant documentaire (RAG) — extension séparée de l'analyse PPMI

Un assistant qui répond aux questions sur un petit corpus (4 articles publiés + le README du projet) **en citant
chaque source** (document et page ou section), et qui préfère répondre « Je ne trouve pas cela dans le corpus »
plutôt que d'inventer.

> **Outil de recherche documentaire, pas un outil médical.** Il ne lit aucune donnée patient, ne pose aucun
> diagnostic et ne recommande aucun traitement : une question médicale personnelle est refusée avant toute recherche.

## Comment il répond

```text
question ─► garde-fou ─► recherche des 5 passages les plus proches ─► LLM (API) ─► vérification des citations ─► réponse + sources
              │ question médicale personnelle → refus           │ pas de clé d'API → passages cités (mode extractif)
                                                                 │ citation absente ou inventée → réponse rejetée, passages cités
```

| Étape | Choix | Pourquoi |
|---|---|---|
| Corpus | `rag/manifest.yaml` : la seule liste des sources | le code refuse `data/`, `reports/tables/`, `models/`, `mlruns/`, `mlflow.db` et tout tableau avec une colonne PATNO |
| Extraction | pypdf, page par page ; Markdown section par section ; bibliographie retirée | la page ou la section est ce qu'on cite ; les références attirent la recherche sans répondre |
| Découpage | LangChain `RecursiveCharacterTextSplitter`, 1 000 caractères, recouvrement 150 | coupe aux paragraphes puis aux phrases ; ≈ 250 tokens, sous la limite du modèle (512) |
| Embeddings | `intfloat/multilingual-e5-small`, local, licence MIT, ≈ 470 Mo, 2 threads | questions en français, articles en anglais ; aucun texte n'est envoyé pour calculer les vecteurs |
| Index | FAISS exact (cosinus), passages en JSON lisible | ≈ 400 passages : la recherche exacte suffit ; pas de pickle à recharger |
| Génération | Groq `openai/gpt-oss-120b` via une API compatible OpenAI (LangChain), température 0, effort de raisonnement `medium` (`low` en v1) | offre gratuite sans carte bancaire ; remplaçant recommandé par Groq de Llama 3.3 70B (arrêté le 16/08/2026) ; autre fournisseur = changer `base_url`, `key_env` et `model` |
| Contrôle | chaque phrase doit citer un extrait [n] qui existe, sinon la réponse est rejetée | jamais de réponse sans source |

**Pourquoi pas LangGraph ?** La chaîne est linéaire (une recherche, une génération, un contrôle) : un graphe
n'apporterait rien ici. **Pourquoi pas de seuil de score ?** Avec ce modèle, les scores se situent en général entre
0,7 et 1,0 et seul leur ordre compte (fiche du modèle) : c'est le LLM, guidé par la consigne, qui dit « introuvable ».

**Ce qui part vers l'API** : la question et les 5 extraits retrouvés (articles publiés ou README). Jamais de donnée
PPMI. L'offre gratuite de Mistral n'active plus les clés d'API (constaté le 21/11/2026) : d'où Groq. L'offre gratuite
est plafonnée (requêtes et tokens par minute) ; au-delà, l'API répond 429 et l'assistant réessaie après une pause.

## Utilisation

```bash
pip install -e ".[rag]"                              # ou : uv sync --extra rag
dvc repro rag_eval                                   # index (≈ 400 passages) puis évaluation sans LLM
python scripts/ask_rag.py "Combien de patients non traités PPMI a-t-elle inclus ?"
```

Pour les réponses rédigées : créer une clé sur console.groq.com (« API Keys »), l'écrire dans `.env`
(`GROQ_API_KEY=...`, fichier jamais versionné), puis `python scripts/eval_rag.py --llm` (≈ 5 min : une pause de 15 s
entre deux questions, pour rester sous la limite de tokens par minute). Sans clé, l'assistant donne les passages cités.
Le README du projet fait partie du corpus : après l'avoir modifié, `dvc repro rag_index` reconstruit l'index.

## Évaluation

22 questions écrites **avant** toute mesure (`rag/eval/questions.yaml`), réponses vérifiées à la main dans les textes :
16 sur le corpus (10 articles, 6 projet), 3 hors corpus (réponse attendue : introuvable), 3 médicales personnelles
(refus attendu). Les réglages de `params.yaml` sont fixés avant et ne sont pas ajustés après : 22 questions sont trop
peu pour régler puis mesurer. Runs dans l'experiment MLflow `extension_rag` ; détail par question en artefact.

| Métrique | Sens |
|---|---|
| `hit_at_5`, `mrr` | le document attendu est-il parmi les 5 passages, et à quel rang ? |
| `refus_corrects`, `faux_refus` | questions médicales refusées ; questions légitimes refusées à tort |
| `reponses_sourcees`, `reponses_rejetees` | avec le LLM : réponses citées et valides ; réponses rejetées par le contrôle |
| `introuvable_hors_corpus` | avec le LLM : « introuvable » sur les 3 questions hors corpus |

### Relecture de la consigne v1 (21/11/2026, `openai/gpt-oss-120b`, effort `low`)

Métriques automatiques : 16 recherches sur 16 trouvent le bon document dans les 5 passages (MRR 0,97), 3 refus sur 3,
3 « introuvable » sur 3 hors corpus, 13 réponses sourcées sur 16, 1 rejetée par le contrôle, 2 « introuvable » à tort.
Relecture à la main des 16 réponses sur le corpus : **12 justes et sourcées**, **1 fausse mais citée** (q13 : « le
GRU-D fait mieux que le MLP », alors que le passage cité dit l'inverse et que l'hypothèse n'est pas confirmée),
**3 manquées** sans erreur (q02 et q08 « introuvable » alors que le 1er passage contient la réponse ; q15 rejetée).
Le contrôle automatique ne voit pas q13 : une citation qui existe ne garantit pas une phrase fidèle.

### Consigne v2 et jeu de contrôle

La v2 demande de reprendre les conclusions telles qu'elles sont écrites (« non significatif », « non confirmée »),
de donner les intervalles et de répondre dès qu'un passage contient la réponse ; l'effort de raisonnement passe à
`medium`. Comme elle a été écrite en regardant les erreurs de la v1, ses résultats sur les 22 questions sont
optimistes : 9 questions nouvelles (`rag/eval/questions_controle.yaml`), écrites avant tout essai de la v2, disent si
l'amélioration tient. `python scripts/eval_rag.py --llm --questions rag/eval/questions_controle.yaml`.

### Résultats de la v2 (21/11/2026, `openai/gpt-oss-120b`, effort `medium`)

Relecture à la main de chaque réponse dans les textes (16 questions sur le corpus) :

| | v1 (effort `low`) | v2 (effort `medium`) |
|---|---|---|
| Recherche : bon document dans les 5 passages (MRR) | 16 / 16 (0,97) | 16 / 16 (0,97) |
| Réponses sourcées (métrique automatique) | 13 / 16 | 15 / 16 |
| Justes | 12 | 10 |
| Conclusion juste, mais une phrase fausse ou trompeuse | 0 | 4 (q03, q11, q12, q13) |
| Fausses mais citées | 1 (q13) | 1 (q02) |
| Manquées (« introuvable » ou rejetée) | 3 (q02, q08, q15) | 1 (q08) |
| Hors corpus : « introuvable » ; questions médicales : refus | 3 / 3 ; 3 / 3 | 3 / 3 ; 3 / 3 |

Jeu de contrôle (v2 seulement) : la recherche trouve le bon document pour les 7 questions sur le corpus, mais moins
haut (MRR 0,79) ; **4 réponses justes, aucune fausse, 3 manquées** (c02, c03, c04 : « introuvable » alors que la
réponse était dans un extrait, au rang 3, 1 et 5) ; hors corpus 1 / 1, refus 1 / 1.

**Lecture.** La v2 répond plus souvent et corrige q13 et q15, mais elle n'est pas meilleure dans l'ensemble. Elle se
trompe encore avec une citation valide (q02 : « Hoehn et Yahr ≥ 2 exigé », alors que l'article dit stade < 3 et
diagnostic depuis moins de 2 ans) et ajoute des phrases secondaires fausses : q03 dit SWEDD « non défini » et donne
62 (sous-groupe imagé) au lieu de 64 inclus ; q11 interprète mal les « 40 % » ; q12 mélange les chiffres de deux
comparaisons ; q13 oppose 7,37 et 7,44 sans dire que l'écart n'est pas significatif. Sur des questions qu'elle n'a
jamais vues, elle ne se trompe pas mais en manque 3 sur 7. Sur 16 + 7 questions, ces écarts restent de l'ordre du
bruit. **Décision** : la v2 reste en place (c'est elle que sert l'API), sans nouveau réglage : régler encore sur les
mêmes questions ôterait toute valeur à la mesure. Une réponse se lit avec ses sources.

Après ces mesures, la phrase du README sur H1 a été précisée : « le GRU-D ne bat pas le MLP » se lisait comme « il
fait moins bien », alors que son erreur est à peine plus basse et que l'écart n'est pas significatif. C'est une
correction du document (pour tous ses lecteurs), pas de l'assistant ; les métriques ci-dessus n'ont pas été relancées.

## Les articles du corpus (`rag/corpus/`)

| Fichier | Référence | Accès | Pourquoi il sert au projet |
|---|---|---|---|
| `2018_marek_ppmi_cohorte_biomarqueurs.pdf` | Marek K. et al. *The Parkinson's Progression Markers Initiative (PPMI) — establishing a PD biomarker cohort.* Annals of Clinical and Translational Neurology, 5(12), 2018. DOI 10.1002/acn3.644 | libre (CC BY-NC-ND 4.0) | décrit la cohorte PPMI : inclusion, visites, biomarqueurs |
| `2024_dentamaro_multimodal_xai_ppmi.pdf` | Dentamaro V., Impedovo D., Musti L., Pirlo G., Taurisano P. *Enhancing early Parkinson's disease detection through multimodal deep learning and explainable AI: insights from the PPMI database.* Scientific Reports, 14:20941, 2024. DOI 10.1038/s41598-024-70165-4 | libre | deep learning multimodal (IRM 3D + clinique) et explicabilité sur PPMI |
| `2025_junaid_multitache_progression_depression.pdf` | Junaid M., Ghergherehchi M., Lee S. *Multitask Deep Learning for Predicting Parkinson's Progression and Depression From Multimodal Time Series Data.* IEEE Access, 13, 147818, 2025. DOI 10.1109/ACCESS.2025.3593254 | libre (CC BY 4.0) | réseau récurrent multitâche sur des séries temporelles PPMI : proche de notre GRU-D multitâche |
| `2026_johnson_voix_mpower_trajectoires.pdf` | Johnson D. R., Logeswari G. *Longitudinal voice biomarker trajectory modelling for Parkinson's disease severity: domain-adaptive transfer learning on mPower real-world smartphone data.* Frontiers in Digital Health, 8:1864460, 2026. DOI 10.3389/fdgth.2026.1864460 | libre (CC BY 4.0) | trajectoires de sévérité à partir de la voix (mPower) : contexte du choix « PPMI seulement » |

Les PDF ne vont **jamais dans Git** (droits des éditeurs) : `rag/corpus/` est suivi par **DVC** (`rag/corpus.dvc`,
remote S3 privé). Ajouter un article : déposer le PDF dans `rag/corpus/` (nom `année_auteur_sujet.pdf`), ajouter une
entrée dans `rag/manifest.yaml` et une ligne ci-dessus, puis `dvc add rag/corpus`, `dvc push` et `dvc repro rag_eval`.

## Limites

* Petit corpus (4 articles) : l'assistant ne connaît que ces textes ; il ne remplace pas une recherche bibliographique.
* 22 questions : une estimation grossière, sans jeu de test séparé ; le garde-fou médical est une règle de mots-clés,
  doublée par la consigne du LLM, pas une garantie.
* Le contrôle vérifie qu'une citation existe, pas que la phrase dit exactement ce que dit l'extrait (q13 en v1, q02
  en v2) : les réponses restent à relire avec leurs sources.
* Sensible à la formulation : q01 (« patients atteints de Parkinson non traités ») reçoit 423, mais « patients non
  traités » reçoit « introuvable », en v1 comme en v2. Les suggestions du tableau de bord reprennent donc les
  formulations évaluées (q01, q13, q22), pas des variantes non mesurées.
