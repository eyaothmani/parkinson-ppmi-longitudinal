"""Assistant documentaire : préfixes du modèle e5, LLM via API (sans réseau), séparation avec le pipeline PPMI."""
import ast
from pathlib import Path

import numpy as np

import parkinson_mm.rag as rag
from parkinson_mm.rag import embed
from parkinson_mm.rag.llm import make_llm

ALLOWED = {"..utils.io", "..utils.log", "..training.tracking"}       # seuls emprunts au reste du paquet


class FakeSentenceTransformer:
    def __init__(self, name, device):
        self.seen = []

    def encode(self, texts, **kwargs):
        self.seen += texts
        return np.ones((len(texts), 4)) / 2


def test_e5_prefixes(monkeypatch):
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    model = embed.E5Embedder("intfloat/multilingual-e5-small", threads=1)
    assert model.passages(["texte"]).shape == (1, 4) and model.query("question").dtype == np.float32
    assert model.model.seen == ["passage: texte", "query: question"]


def test_llm_needs_a_key(monkeypatch, tmp_path):
    key = "CLE_RAG_TEST"
    settings = {"base_url": "https://api.groq.com/openai/v1", "key_env": key, "model": "openai/gpt-oss-120b",
                "reasoning_effort": "low", "temperature": 0, "max_tokens": 50}
    monkeypatch.setenv(key, "provisoire")
    monkeypatch.delenv(key)                          # absente pendant le test, et nettoyée après (même via .env)
    assert make_llm(settings, tmp_path) is None
    (tmp_path / ".env").write_text(f"{key}=cle-factice\n")
    llm = make_llm(settings, tmp_path)               # construit le client, sans appeler l'API
    assert llm.model_name == "openai/gpt-oss-120b" and llm.openai_api_base == settings["base_url"]
    assert llm.reasoning_effort == "low"
    assert make_llm({**settings, "reasoning_effort": None}, tmp_path).reasoning_effort is None


def test_rag_package_never_imports_the_ppmi_pipeline():
    for path in Path(rag.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level >= 2:
                assert f"..{node.module}" in ALLOWED, f"{path.name} importe ..{node.module}"
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                assert not node.module.startswith("parkinson_mm"), path.name
