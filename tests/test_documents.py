import asyncio
import io
import json
from pathlib import Path

import httpx
import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfWriter

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.main import create_app
from app.models.document_chunk import ParsedDocument, SourceRecord
from app.services.chunking import chunk_document, get_tokenizer
from app.services.document_parser import parse_json, parse_pdf

PDF = Path(__file__).parent / "fixtures" / "document.pdf"


def config(**values):
    return Settings(_env_file=None, **values)


def test_pdf_pages_and_text():
    doc = parse_pdf(PDF.read_bytes(), config())
    assert doc.page_count == 2
    assert [r.page for r in doc.records] == [1, 2]
    assert "AWS" in doc.records[0].text
    assert "24 hours" in doc.records[1].text
    chunks = chunk_document(doc, config())
    assert [c.page for c in chunks] == [1, 2]
    assert all(c.json_path is None for c in chunks)


@pytest.mark.parametrize(
    "kind,code",
    [
        ("corrupt", "INVALID_DOCUMENT"),
        ("encrypted", "ENCRYPTED_PDF"),
        ("blank", "NO_EXTRACTABLE_TEXT"),
        ("pages", "TOO_MANY_PAGES"),
        ("text", "DOCUMENT_TOO_LARGE"),
    ],
)
def test_pdf_errors(kind, code):
    contents = PDF.read_bytes()
    settings = config()
    if kind in {"encrypted", "blank"}:
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        if kind == "encrypted":
            writer.encrypt("password")
        buffer = io.BytesIO()
        writer.write(buffer)
        contents = buffer.getvalue()
    elif kind == "corrupt":
        contents = b"%PDF-1.7\ncorrupt"
    elif kind == "pages":
        settings = config(max_pdf_pages=1)
    else:
        settings = config(max_extracted_chars=1)
    with pytest.raises(ServiceError) as exc:
        parse_pdf(contents, settings)
    assert exc.value.code == code


def test_json_paths_types_and_order():
    raw = {"a.b": [{'quote"': "AWS", "active": False, "count": 0, "missing": None}]}
    doc = parse_json(json.dumps(raw).encode(), config())
    assert [r.text for r in doc.records] == ["AWS", "false", "0", "null"]
    assert doc.records[0].json_path == '$["a.b"][0]["quote\\""]'
    assert all(r.page is None for r in doc.records)


@pytest.mark.parametrize(
    "raw,code",
    [
        (b"{}", "EMPTY_DOCUMENT"),
        (b"[]", "EMPTY_DOCUMENT"),
        (b'{"x":" "}', "EMPTY_DOCUMENT"),
        (b"null", "INVALID_DOCUMENT"),
        (b"{", "INVALID_DOCUMENT"),
        (b"\xff", "INVALID_DOCUMENT"),
        (b'{"x":NaN}', "INVALID_DOCUMENT"),
        (b'{"x":1e999}', "INVALID_DOCUMENT"),
        (b'{"x":1,"x":2}', "INVALID_DOCUMENT"),
        (b'{"x":"\\ud800"}', "INVALID_DOCUMENT"),
    ],
)
def test_invalid_json(raw, code):
    with pytest.raises(ServiceError) as exc:
        parse_json(raw, config())
    assert exc.value.code == code


@pytest.mark.parametrize(
    "settings,raw,code",
    [
        (config(max_json_depth=1), b'{"x":{"y":1}}', "JSON_TOO_DEEP"),
        (config(max_document_records=1), b"[1,2]", "TOO_MANY_RECORDS"),
        (config(max_extracted_chars=3), b'{"x":"AWS"}', "DOCUMENT_TOO_LARGE"),
    ],
)
def test_json_limits(settings, raw, code):
    with pytest.raises(ServiceError) as exc:
        parse_json(raw, settings)
    assert exc.value.code == code


def test_unicode_chunk_budget_offsets_overlap_and_determinism():
    text = "Hello 世界 नमस्ते 🙂 <|endoftext|> " * 30
    doc = ParsedDocument((SourceRecord(text, "json", json_path='$["x"]'),))
    settings = config(chunk_size_tokens=40, chunk_overlap_tokens=8)
    chunks = chunk_document(doc, settings)
    assert len(chunks) > 1
    assert chunks == chunk_document(doc, settings)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    tokenizer = get_tokenizer()
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(text)
    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert len(tokenizer.encode_ordinary(chunk.retrieval_text)) <= 40
        assert "\ufffd" not in chunk.text
    for left, right in zip(chunks, chunks[1:]):
        assert left.start_char < right.start_char <= left.end_char
        overlap = text[right.start_char : left.end_char]
        assert len(tokenizer.encode_ordinary(overlap)) <= 8


def test_langchain_splitter_receives_one_provenance_document_per_record(monkeypatch):
    seen = []
    original = RecursiveCharacterTextSplitter.split_documents

    def observe(self, documents):
        seen.extend(documents)
        return original(self, documents)

    monkeypatch.setattr(RecursiveCharacterTextSplitter, "split_documents", observe)
    document = ParsedDocument(
        (
            SourceRecord("First paragraph.\n\nSecond paragraph.", "pdf", page=1),
            SourceRecord("AWS is the provider.", "json", json_path='$["cloud"]'),
        ),
        page_count=1,
    )

    chunks = chunk_document(document, config(chunk_size_tokens=8, chunk_overlap_tokens=2))

    assert chunks
    assert all(isinstance(item, Document) for item in seen)
    assert [item.metadata["record_index"] for item in seen] == [0, 1]
    assert seen[0].metadata == {"source_type": "pdf", "record_index": 0, "page": 1}
    assert seen[1].metadata == {
        "source_type": "json",
        "record_index": 1,
        "json_path": '$["cloud"]',
    }


def test_chunk_limit_and_long_path():
    doc = ParsedDocument((SourceRecord("word " * 100, "pdf", page=1),))
    with pytest.raises(ServiceError, match="chunk limit"):
        chunk_document(
            doc, config(chunk_size_tokens=10, chunk_overlap_tokens=2, max_document_chunks=1)
        )
    doc = ParsedDocument((SourceRecord("AWS", "json", json_path="key " * 100),))
    with pytest.raises(ServiceError) as exc:
        chunk_document(doc, config(chunk_size_tokens=10, chunk_overlap_tokens=2))
    assert exc.value.code == "SOURCE_PATH_TOO_LONG"


def test_tokenizer_is_offline(monkeypatch):
    import requests

    def fail(*args, **kwargs):
        raise AssertionError("Network call attempted")

    monkeypatch.setattr(requests, "get", fail)
    get_tokenizer.cache_clear()
    assert get_tokenizer().encode_ordinary("AWS")


@pytest.mark.parametrize(
    "name,contents,status,code",
    [
        ("doc.pdf", PDF.read_bytes(), 503, "PROVIDER_NOT_CONFIGURED"),
        ("doc.json", b'{"cloud":"AWS"}', 503, "PROVIDER_NOT_CONFIGURED"),
        ("doc.pdf", b"not a PDF", 422, "INVALID_DOCUMENT"),
        ("doc.json", b"not JSON", 422, "INVALID_DOCUMENT"),
    ],
)
def test_document_ingestion_api(name, contents, status, code):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(config())), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/v1/qa",
                files={
                    "questions_file": ("q.json", b'["Cloud?"]', "application/json"),
                    "document_file": (name, contents, "application/octet-stream"),
                },
            )

    response = asyncio.run(run())
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
