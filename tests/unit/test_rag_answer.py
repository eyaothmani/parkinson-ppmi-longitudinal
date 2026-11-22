"""Assistant documentaire : garde-fou, index FAISS, réponse sourcée et vérification des citations (LLM factice)."""
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from parkinson_mm.rag import NOT_FOUND, ask
from parkinson_mm.rag.embed import HashEmbedder
from parkinson_mm.rag.guard import is_personal_medical
from parkinson_mm.rag.index import Index, build_index

CHUNKS = [{"id": 0, "doc_id": "marek", "citation": "Marek 2018", "where": "p. 1",
           "text": "PPMI enrolled 423 untreated Parkinson subjects at 24 sites"},
          {"id": 1, "doc_id": "johnson", "citation": "Johnson 2026", "where": "p. 4",
           "text": "voice recordings from mPower predict MDS-UPDRS part II"}]


@pytest.fixture()
def index(tmp_path):
    build_index(CHUNKS, HashEmbedder(), tmp_path, {"corpus_version": "test"})
    return Index(tmp_path, HashEmbedder.name)


@pytest.mark.parametrize("question, refused", [
    ("Quel traitement mon père devrait-il prendre ?", True), ("J'ai des tremblements, ai-je Parkinson ?", True),
    ("Quelle dose de lévodopa dois-je prendre ?", True), ("Should I take levodopa?", True),
    ("Quels traitements étaient interdits à l'inclusion dans PPMI ?", False),
    ("Peux-tu me dire comment le traitement modifie le score ?", False), ("Combien de patients PPMI ?", False)])
def test_guard_refuses_personal_medical_questions_only(question, refused):
    assert is_personal_medical(question) is refused


def test_index_search_and_model_check(index, tmp_path):
    hits = index.search(HashEmbedder().query("untreated Parkinson subjects enrolled"), k=5)
    assert len(hits) == 2 and hits[0]["doc_id"] == "marek" and hits[0]["score"] >= hits[1]["score"]
    with pytest.raises(ValueError, match="dvc repro rag_index"):
        Index(tmp_path, "intfloat/multilingual-e5-small")


@pytest.mark.parametrize("reply, status, n_sources", [
    ("PPMI a inclus 423 patients non traités [1].", "generee", 1), (NOT_FOUND, "introuvable", 0),
    ("Réponse sans aucune source.", "citations_invalides", 2), ("Source inventée [7].", "citations_invalides", 2)])
def test_answer_keeps_only_valid_citations(index, reply, status, n_sources):
    llm = FakeListChatModel(responses=[reply])
    answer = ask("untreated Parkinson subjects enrolled", index, HashEmbedder(), 5, llm)
    assert answer.status == status and len(answer.sources) == n_sources
    if status == "generee":
        assert answer.text == reply and answer.sources[0]["n"] == 1 and answer.sources[0]["citation"] == "Marek 2018"


def test_without_llm_or_for_medical_questions(index):
    extractive = ask("voice mPower", index, HashEmbedder(), 5)
    assert extractive.status == "extractif" and "mPower" in extractive.text
    assert extractive.sources[0]["citation"] == "Johnson 2026" and extractive.sources[0]["n"] == 1
    refused = ask("Quelle dose de lévodopa dois-je prendre ?", index, HashEmbedder(), 5)
    assert refused.status == "refus" and refused.sources == [] and refused.hits == []
