"""GET /summary (résultats agrégés) et POST /rag/ask (assistant documentaire, réponse sourcée ou refus)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .predictor import DISCLAIMER
from .rag_service import answer, assistant
from .security import require_key
from .state import state
from .summary import load_summary

router = APIRouter()


class Question(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class Source(BaseModel):
    n: int
    citation: str
    where: str


class RagResponse(BaseModel):
    status: str = Field(..., description="generee, introuvable, refus, extractif ou citations_invalides")
    text: str
    sources: list[Source]
    generative: bool = Field(..., description="vrai si un LLM a rédigé la réponse")
    disclaimer: str


@router.get("/summary")
def summary(_: str = Depends(require_key)) -> dict:
    return {**load_summary(), "disclaimer": DISCLAIMER}


@router.post("/rag/ask", response_model=RagResponse)
def rag_ask(request: Question, _: str = Depends(require_key)) -> RagResponse:
    try:
        rag = assistant(state)
    except (ImportError, FileNotFoundError, OSError) as exc:
        raise HTTPException(503, "Assistant indisponible : `pip install -e \".[rag]\"` puis "
                                 "`dvc repro rag_index`.") from exc
    result, generative = answer(request.question, rag)
    sources = [Source(n=s["n"], citation=s["citation"], where=s["where"]) for s in result.sources]
    return RagResponse(status=result.status, text=result.text, sources=sources, generative=generative,
                       disclaimer=DISCLAIMER)
