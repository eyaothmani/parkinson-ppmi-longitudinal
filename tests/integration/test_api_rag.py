"""API : POST /rag/ask avec un petit index (embedder haché) et des LLM factices, dont un qui tombe en panne."""
from types import SimpleNamespace

import pytest

from parkinson_mm.rag import NOT_FOUND
from parkinson_mm.rag.embed import HashEmbedder
from parkinson_mm.rag.index import Index, build_index
from parkinson_mm.serving.state import state

CHUNKS = [{"id": 0, "doc_id": "marek", "citation": "Marek 2018", "where": "p. 3",
           "text": "PPMI enrolled 423 untreated Parkinson subjects"}]


class BrokenLLM:
    def invoke(self, messages):
        raise ConnectionError("API du LLM injoignable")


@pytest.fixture
def rag(tmp_path, monkeypatch):
    build_index(CHUNKS, HashEmbedder(), tmp_path, {"corpus_version": "test"})
    assistant = {"index": Index(tmp_path, HashEmbedder.name), "embedder": HashEmbedder(), "llm": None, "k": 3}
    monkeypatch.setitem(state, "rag", assistant)
    return assistant


def replying(text):
    return SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=text))


@pytest.mark.parametrize("llm, status, generative", [
    (None, "extractif", False), (replying("423 patients [1]."), "generee", True),
    (replying(NOT_FOUND), "introuvable", True), (BrokenLLM(), "extractif", False)])
def test_rag_answers_with_sources_or_falls_back(client, key, rag, llm, status, generative):
    rag["llm"] = llm
    body = client.post("/rag/ask", json={"question": "untreated Parkinson subjects enrolled"}, headers=key).json()
    assert body["status"] == status and body["generative"] is generative
    assert status == "introuvable" or body["sources"][0]["citation"] == "Marek 2018"


def test_rag_refuses_personal_medical_questions_and_short_input(client, key, rag):
    body = client.post("/rag/ask", json={"question": "Quelle dose de lévodopa dois-je prendre ?"}, headers=key).json()
    assert body["status"] == "refus" and body["sources"] == []
    assert client.post("/rag/ask", json={"question": "a"}, headers=key).status_code == 422


def test_rag_without_index_is_unavailable(client, key, monkeypatch, tmp_path):
    monkeypatch.setitem(state, "rag", None)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    assert client.post("/rag/ask", json={"question": "Que mesure PPMI ?"}, headers=key).status_code == 503
