"""Découpage en passages (LangChain : coupe d'abord aux paragraphes, puis aux phrases, puis aux mots).
Chaque passage garde sa source (document, page ou section) : c'est ce que l'assistant cite."""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .corpus import Page


def make_chunks(pages: list[Page], size: int, overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = []
    for page in pages:
        for text in splitter.split_text(page.text):
            chunks.append({"id": len(chunks), "doc_id": page.doc_id, "citation": page.citation,
                           "where": page.where, "text": text})
    return chunks
