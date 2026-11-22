"""Assistant documentaire : corpus (sources permises), extraction PDF/Markdown, nettoyage, découpage."""
import pytest
import yaml

from parkinson_mm.rag.chunks import make_chunks
from parkinson_mm.rag.corpus import load_manifest, load_pages
from parkinson_mm.rag.extract import clean, cut_references, read_parts


def tiny_pdf(path, pages):
    """PDF minimal écrit à la main (une ligne de texte par page), lisible par pypdf."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {len(objs)} 0 R "
                    "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> >> >> >>")
        kids.append(f"{len(objs)} 0 R")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    body, offsets = "%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(body))
        body += f"{i} 0 obj\n{obj}\nendobj\n"
    xref = "".join(f"{o:010d} 00000 n \n" for o in offsets)
    body += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n{xref}trailer << /Size {len(objs) + 1} /Root 1 0 R >>\n"
    path.write_bytes((body + f"startxref\n{len(body)}\n%%EOF\n").encode("latin-1"))


def test_pdf_pages_markdown_sections_and_references_cut(tmp_path):
    tiny_pdf(tmp_path / "a.pdf", ["PPMI enrolled 423 untreated subjects", "Methods text", "References", "Smith 2020"])
    assert [w for w, _ in read_parts(tmp_path / "a.pdf")] == ["p. 1", "p. 2", "p. 3"]
    assert "423" in read_parts(tmp_path / "a.pdf")[0][1]
    (tmp_path / "R.md").write_text("# Titre\nintro\n## Limites\npas de validation externe\n", encoding="utf-8")
    assert [w for w, _ in read_parts(tmp_path / "R.md")] == ["introduction", "section « Limites »"]
    assert cut_references(["a", "b", "x\nReferences\ny", "z"]) == ["a", "b", "x\n"]
    assert cut_references(["References\nfaux sommaire", "b"]) == ["References\nfaux sommaire", "b"]   # 1re moitié
    assert clean("multi-\nmodal  ﬂuid\n\n\nfin") == "multimodal fluid\nfin"


@pytest.mark.parametrize("bad", ["data/processed/pd_patients.csv", "reports/tables/models/x.csv", "mlflow.db",
                                 "../ailleurs.md"])
def test_forbidden_sources_are_refused(tmp_path, bad):
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({"documents": [{"id": "x", "fichier": bad, "citation": "x"}]}))
    with pytest.raises(ValueError, match="interdite"):
        load_manifest(tmp_path, "m.yaml")


def test_patient_table_is_refused_and_chunks_keep_their_source(tmp_path):
    docs = [{"id": "p", "fichier": "p.md", "citation": "Projet"}]
    (tmp_path / "m.yaml").write_text(yaml.safe_dump({"documents": docs}))
    (tmp_path / "p.md").write_text("## Résultats\n" + "texte agrégé. " * 200, encoding="utf-8")
    chunks = make_chunks(load_pages(tmp_path, "m.yaml"), 300, 50)
    assert len(chunks) > 3 and {c["where"] for c in chunks} == {"section « Résultats »"}
    (tmp_path / "p.md").write_text("## Fuite\n| PATNO | NP3 |\n|---|---|\n| 3001 | 20 |\n", encoding="utf-8")
    with pytest.raises(ValueError, match="patient"):
        load_pages(tmp_path, "m.yaml")
