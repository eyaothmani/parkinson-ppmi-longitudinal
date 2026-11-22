"""Assistant documentaire : évaluation sur rag/eval/questions.yaml, tracée dans MLflow (experiment extension_rag).
Sans option : recherche et garde-fou seulement (déterministe, étape DVC rag_eval).
--llm : réponses générées par l'API (clé GROQ_API_KEY) ; métriques dans reports/metrics/rag_llm.json.
--questions rag/eval/questions_controle.yaml : un autre jeu (métriques rag[_llm]_questions_controle.json)."""
from _cli import main

from parkinson_mm.rag import eval_rag

if __name__ == "__main__":
    main(__doc__, [lambda cfg, args: print(eval_rag(cfg, use_llm=args.llm, questions=args.questions))],
         [(("--llm",), {"action": "store_true", "help": "générer les réponses avec le LLM (API)"}),
          (("--questions",), {"default": None, "help": "jeu de questions (défaut : rag.questions de params.yaml)"})])
