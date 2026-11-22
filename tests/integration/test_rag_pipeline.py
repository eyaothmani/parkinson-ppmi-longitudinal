"""Assistant documentaire de bout en bout (embedder et LLM factices) : index, évaluation, runs MLflow."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import yaml
from mlflow import MlflowClient

from parkinson_mm.rag import NOT_FOUND, build_rag_index, eval_rag
from parkinson_mm.rag.embed import HashEmbedder
from parkinson_mm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
DOC = "## Cohorte\nPPMI enrolled 423 untreated Parkinson subjects.\n## Voix\nmPower voice predicts part II.\n"
QUESTIONS = [{"id": "a", "type": "corpus", "document": "doc", "question": "untreated Parkinson subjects enrolled"},
             {"id": "b", "type": "hors_corpus", "question": "football world cup"},
             {"id": "c", "type": "medical", "question": "Quelle dose de lévodopa dois-je prendre ?"}]


class RuleLLM:
    """Répond « introuvable » au football, sinon une phrase qui cite l'extrait 1."""
    def invoke(self, messages):
        return SimpleNamespace(content=NOT_FOUND if "football" in messages[-1].content else "Réponse [1].")


def test_index_then_evaluation_with_and_without_llm(tmp_path, monkeypatch):
    shutil.copy(ROOT / "params.yaml", tmp_path / "params.yaml")
    (tmp_path / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "mlflow_experiments.yaml", tmp_path / "docs")
    (tmp_path / "rag" / "eval").mkdir(parents=True)
    (tmp_path / "doc.md").write_text(DOC, encoding="utf-8")
    manifest = {"documents": [{"id": "doc", "fichier": "doc.md", "citation": "Doc 2024"}]}
    (tmp_path / "rag" / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (tmp_path / "rag" / "eval" / "questions.yaml").write_text(yaml.safe_dump({"questions": QUESTIONS}))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    cfg = load_config(tmp_path / "params.yaml").make_dirs()
    cfg["rag"]["pause_s"] = 0

    info = build_rag_index(cfg, HashEmbedder())
    assert info["n_documents"] == 1 and info["corpus_version"] == "non-suivi"
    passages = (tmp_path / "rag" / "index" / "passages.jsonl").read_text(encoding="utf-8")
    assert "PATNO" not in passages and "423" in passages
    plain = eval_rag(cfg, embedder=HashEmbedder())
    assert plain["hit_at_5"] == 1.0 and plain["refus_corrects"] == 1.0 and plain["faux_refus"] == 0.0
    shutil.copy(tmp_path / "rag" / "eval" / "questions.yaml", tmp_path / "rag" / "eval" / "controle.yaml")
    assert eval_rag(cfg, embedder=HashEmbedder(), questions="rag/eval/controle.yaml") == plain
    assert (cfg.path("metrics") / "rag_controle.json").exists()       # un autre jeu n'écrase pas rag.json
    with_llm = eval_rag(cfg, embedder=HashEmbedder(), llm=RuleLLM())
    assert with_llm["reponses_sourcees"] == 1.0 and with_llm["introuvable_hors_corpus"] == 1.0
    assert json.loads((cfg.path("metrics") / "rag_llm.json").read_text())["reponses_rejetees"] == 0

    client = MlflowClient()
    experiment = client.get_experiment_by_name(f"{cfg.tracking.experiment_prefix}/{cfg.rag.question}")
    runs = client.search_runs([experiment.experiment_id], "tags.stage = 'rag_eval'")
    assert {r.info.run_name for r in runs} == {"Recherche seule", "Recherche seule, controle",
                                               "Recherche + LLM (openai/gpt-oss-120b, consigne v2)"}
    assert all(r.data.tags["data_version"] == "non-suivi" for r in runs)
