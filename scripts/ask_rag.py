"""Poser une question à l'assistant documentaire. Exemple :
python scripts/ask_rag.py "Combien de patients de novo PPMI a-t-elle inclus ?"
Outil de recherche documentaire : il ne pose aucun diagnostic et ne recommande aucun traitement."""
from _cli import main

from parkinson_mm.rag import ask
from parkinson_mm.rag.index import Index
from parkinson_mm.rag.llm import make_llm
from parkinson_mm.rag.stage import embedder_for


def answer(cfg, args) -> None:
    embedder = embedder_for(cfg)
    llm = None if args.sans_llm else make_llm(cfg.rag.llm, cfg.root)
    result = ask(args.question, Index(cfg.root / cfg.rag.index_dir, embedder.name), embedder, cfg.rag.k, llm)
    print(f"\n[{result.status}] {result.text}\n")
    for source in result.sources:
        print(f"  [{source['n']}] {source['citation']}, {source['where']}")


if __name__ == "__main__":
    main(__doc__, [answer], [(("question",), {"help": "la question, entre guillemets"}),
                             (("--sans-llm",), {"action": "store_true", "help": "passages cités seulement"})])
