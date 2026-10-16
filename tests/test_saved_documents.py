import asyncio
import json
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.logging import JSONFormatter, logger
from app.dependencies import get_document_library, get_document_service_factory
from app.main import create_app
from app.services.answer_generator import EvidenceQuote, GroundedAnswer
from app.services.document_service import DocumentService
from app.services.document_store import DocumentLibrary
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService


class CountingEmbeddings:
    def __init__(self):
        self.calls = []

    async def embed(self, texts):
        self.calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]


class Generator:
    async def generate(self, question, context):
        chunk = next(item for item in context if "AWS" in item.text)
        return GroundedAnswer(
            answerable=True,
            answer="AWS",
            evidence=[EvidenceQuote(chunk_id=chunk.chunk_id, quote="AWS")],
            explanation=None,
        )


TOKENS = {"acme": "acme-token", "northwind": "northwind-token"}


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def client_for(tmp_path: Path, tokens: dict[str, str] | None = None):
    settings = Settings(
        _env_file=None,
        chroma_persist_dir=tmp_path,
        tenant_tokens=TOKENS if tokens is None else tokens,
    )
    application = create_app(settings)
    library = DocumentLibrary(tmp_path)
    embeddings = CountingEmbeddings()

    def service():
        retrieval = RetrievalService(embeddings, library.store)
        return DocumentService(embeddings, library, QAService(retrieval, Generator(), settings))

    @asynccontextmanager
    async def factory():
        yield service()

    application.dependency_overrides[get_document_library] = lambda: library
    application.dependency_overrides[get_document_service_factory] = lambda: factory
    return application, embeddings, library


def test_follow_up_questions_do_not_embed_the_document_again(tmp_path):
    application, embeddings, library = client_for(tmp_path)
    before = library.store.client.count_collections()

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            uploaded = await client.post(
                "/api/v1/documents",
                headers=auth("acme-token"),
                files={"document_file": ("source.json", b'{"cloud":"AWS"}', "application/json")},
            )
            assert uploaded.status_code == 200, uploaded.text
            document_id = uploaded.json()["document_id"]
            tenant_id = uploaded.json()["tenant_id"]
            first_embeddings = [list(call) for call in embeddings.calls]

            async def ask(question):
                return await client.post(
                    f"/api/v1/documents/{document_id}/questions",
                    headers=auth("acme-token"),
                    files={
                        "questions_file": (
                            "questions.json",
                            json.dumps([question]).encode(),
                            "application/json",
                        )
                    },
                )

            first = await ask("Which cloud?")
            second = await ask("Which provider?")
            missing = await client.post(
                "/api/v1/documents/00000000-0000-0000-0000-000000000000/questions",
                headers=auth("acme-token"),
                files={
                    "questions_file": ("questions.json", b'["Which cloud?"]', "application/json")
                },
            )
            return document_id, tenant_id, first, second, missing, first_embeddings

    document_id, tenant_id, first, second, missing, upload_calls = asyncio.run(run())
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert first.json()["results"][0]["citations"][0]["snippet"] == "AWS"
    assert missing.status_code == 404
    assert upload_calls == [['$["cloud"]: AWS']]
    assert embeddings.calls[1:] == [["Which cloud?"], ["Which provider?"]]
    assert tenant_id == "acme"
    assert library.get(document_id, "acme") is not None
    assert library.get(document_id, "northwind") is None
    assert library.store.client.count_collections() == before + 1


def test_another_upload_keeps_the_earlier_collection(tmp_path):
    application, _, library = client_for(tmp_path)
    pdf = (Path(__file__).parent / "fixtures/document.pdf").read_bytes()

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            first = await client.post(
                "/api/v1/documents",
                headers=auth("acme-token"),
                files={"document_file": ("one.json", b'{"cloud":"AWS"}', "application/json")},
            )
            second = await client.post(
                "/api/v1/documents",
                headers=auth("acme-token"),
                files={"document_file": ("two.pdf", pdf, "application/pdf")},
            )
            answer = await client.post(
                f"/api/v1/documents/{first.json()['document_id']}/questions",
                headers=auth("acme-token"),
                files={"questions_file": ("q.json", b'["Which cloud?"]', "application/json")},
            )
            return first, second, answer

    first, second, answer = asyncio.run(run())
    assert first.status_code == second.status_code == answer.status_code == 200
    assert first.json()["document_id"] != second.json()["document_id"]
    assert library.get(first.json()["document_id"], "acme") is not None
    assert library.get(second.json()["document_id"], "acme") is not None
    assert answer.json()["results"][0]["answer"] == "AWS"


def test_saved_index_survives_reopening_the_library(tmp_path):
    settings = Settings(_env_file=None, chroma_persist_dir=tmp_path)
    embeddings = CountingEmbeddings()
    library = DocumentLibrary(tmp_path)
    retrieval = RetrievalService(embeddings, library.store)
    service = DocumentService(embeddings, library, QAService(retrieval, Generator(), settings))

    async def save(active):
        from app.services.ingestion import ingest_document

        ingestion = await ingest_document(b'{"cloud":"AWS"}', "json", settings)
        return await active.save(ingestion, len(b'{"cloud":"AWS"}'), "acme")

    saved = asyncio.run(save(service))
    del library, service
    reopened = DocumentLibrary(tmp_path)
    found = reopened.get(saved.document_id, "acme")
    assert reopened.get(saved.document_id, "northwind") is None
    assert found is not None
    assert found.collection_name == saved.collection_name
    contexts = reopened.store.search_saved(found.collection_name, [[1.0, 0.0]], top_k=1)
    assert contexts[0][0].text == "AWS"


def test_missing_and_unknown_tokens_are_rejected(tmp_path):
    application, _, _ = client_for(tmp_path)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            missing = await client.post("/api/v1/documents")
            unknown = await client.post(
                "/api/v1/documents/00000000-0000-0000-0000-000000000000/questions",
                headers=auth("wrong-token"),
            )
            return missing, unknown

    missing, unknown = asyncio.run(run())
    for response in (missing, unknown):
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "AUTHENTICATION_REQUIRED"
        assert response.json()["error"]["message"] == "Authentication is required."
        assert response.headers["www-authenticate"] == "Bearer"


def test_other_tenant_gets_the_same_not_found_as_an_unknown_document(tmp_path):
    application, embeddings, _ = client_for(tmp_path)
    buffer: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record):
            buffer.append(JSONFormatter().format(record))

    handler = Collect()
    logger.addHandler(handler)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            uploaded = await client.post(
                "/api/v1/documents",
                headers=auth("acme-token"),
                files={"document_file": ("source.json", b'{"cloud":"AWS"}', "application/json")},
            )
            foreign = await client.post(
                f"/api/v1/documents/{uploaded.json()['document_id']}/questions",
                headers=auth("northwind-token"),
                files={"questions_file": ("q.json", b'["Which cloud?"]', "application/json")},
            )
            missing = await client.post(
                "/api/v1/documents/00000000-0000-0000-0000-000000000000/questions",
                headers=auth("northwind-token"),
                files={"questions_file": ("q.json", b'["Which cloud?"]', "application/json")},
            )
            return uploaded, foreign, missing

    try:
        uploaded, foreign, missing = asyncio.run(run())
    finally:
        logger.removeHandler(handler)
    assert uploaded.status_code == 200
    assert foreign.status_code == missing.status_code == 404
    foreign_error = foreign.json()["error"]
    missing_error = missing.json()["error"]
    assert foreign_error["code"] == missing_error["code"] == "DOCUMENT_NOT_FOUND"
    assert foreign_error["message"] == missing_error["message"] == "Document was not found."
    assert "acme-token" not in "\n".join(buffer)
    assert "northwind-token" not in "\n".join(buffer)
    assert embeddings.calls == [['$["cloud"]: AWS']]


def test_one_shot_qa_does_not_require_a_tenant_token(tmp_path):
    application, _, _ = client_for(tmp_path)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            return await client.post("/api/v1/qa")

    response = asyncio.run(run())
    assert response.status_code == 422
    assert response.json()["error"]["code"] != "AUTHENTICATION_REQUIRED"


def test_unconfigured_tenants_refuse_saved_documents(tmp_path):
    application, _, _ = client_for(tmp_path, tokens={})

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application), base_url="http://test"
        ) as client:
            return await client.post("/api/v1/documents", headers=auth("acme-token"))

    response = asyncio.run(run())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "TENANTS_NOT_CONFIGURED"


def test_documents_without_an_owner_stay_unreachable(tmp_path):
    database = tmp_path / "documents.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                byte_count INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
            ("11111111-1111-1111-1111-111111111111", "doclegacy", "json", 1, 2),
        )
    library = DocumentLibrary(tmp_path)
    assert library.get("11111111-1111-1111-1111-111111111111", "acme") is None


def test_duplicate_or_invalid_tenant_tokens_are_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tenant_tokens={"acme": "same-token", "northwind": "same-token"})
    with pytest.raises(ValidationError):
        Settings(_env_file=None, tenant_tokens="[\"not-an-object\"]")
