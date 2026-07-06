"""Tests for Chunker (cfg-driven) and loaders (config-injected, no global settings)."""
from __future__ import annotations

import pytest

from rag.chunking import Chunker
from rag.config import RagConfig
from rag.loaders import (
    CSVLoader,
    TextLoader,
    TxtLoader,
    WebLoader,
    get_loader,
)
from rag.types import RawDocument


# ── Chunker ──
def test_chunker_respects_size_and_sets_parent():
    cfg = RagConfig(chunk_size=20, chunk_overlap=4, min_chunk_size=1)
    doc = RawDocument(
        text="",
        metadata={"source": "s", "source_type": "text", "title": "T"},
        sections=[{"heading": "Intro", "text": " ".join(f"word{i} sentence." for i in range(40))}],
    )
    chunks = Chunker(cfg).chunk(doc)
    assert len(chunks) > 1                       # split into multiple chunks
    assert all(c.metadata["section_title"] == "Intro" for c in chunks)
    assert all(c.parent_text and "Intro" in c.parent_text for c in chunks)


def test_chunker_never_splits_mid_sentence():
    cfg = RagConfig(chunk_size=10, chunk_overlap=2, min_chunk_size=1)
    doc = RawDocument(text="", sections=[{"heading": "", "text": "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."}])
    chunks = Chunker(cfg).chunk(doc)
    for c in chunks:
        # each chunk ends at a sentence boundary (last char is terminal punctuation)
        assert c.text.rstrip()[-1] in ".!?"


def test_chunker_two_configs_independent():
    doc = RawDocument(text="", sections=[{"heading": "", "text": " ".join(f"s{i} x." for i in range(30))}])
    big = Chunker(RagConfig(chunk_size=500, min_chunk_size=1)).chunk(doc)
    small = Chunker(RagConfig(chunk_size=10, chunk_overlap=2, min_chunk_size=1)).chunk(doc)
    assert len(small) > len(big)                 # smaller chunk_size → more chunks


# ── Loaders ──
def test_text_loader_splits_paragraphs():
    doc = TextLoader().load("Para one.\n\nPara two.", title="Notes")
    assert doc.metadata["title"] == "Notes"
    assert doc.metadata["source_type"] == "text"
    assert len(doc.sections) == 2


def test_txt_loader_reads_file(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("First para.\n\nSecond para.", encoding="utf-8")
    doc = TxtLoader().load(str(p))
    assert doc.metadata["source_type"] == "text"
    assert "First para." in doc.text


def test_csv_loader_groups_rows(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,role\nAlice,dev\nBob,pm\n", encoding="utf-8")
    doc = CSVLoader().load(str(p))
    assert doc.metadata["source_type"] == "csv"
    assert "Alice" in doc.text and "role: pm" in doc.text


def test_get_loader_dispatch_by_type():
    assert get_loader("file.pdf")[1] == "file"
    assert get_loader("file.csv")[0].__class__.__name__ == "CSVLoader"
    assert get_loader("http://example.com")[1] == "url"


def test_get_loader_web_receives_config():
    cfg = RagConfig(web_ssl_verify=False)
    loader, kind = get_loader("https://example.com", cfg)
    assert kind == "url"
    assert isinstance(loader, WebLoader)
    assert loader.config.web_ssl_verify is False


def test_get_loader_unsupported_raises():
    with pytest.raises(ValueError):
        get_loader("file.xyz")


def test_txt_loader_captures_file_mtime(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("First para.\n\nSecond para.", encoding="utf-8")
    doc = TxtLoader().load(str(p))
    assert doc.metadata.get("source_fetched_at")
    assert doc.metadata.get("source_last_modified")   # from file mtime
