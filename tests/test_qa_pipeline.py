import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.dependencies import get_qa_service_factory
from app.main import create_app
from app.models.document_chunk import DocumentChunk, ParsedDocument
from app.services.answer_generator import EvidenceQuote, GroundedAnswer
from app.services.ingestion import IngestedDocument
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import ChromaVectorStore


class FakeEmbeddings:
    async def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeGenerator:
    def __init__(self):
        self.calls = []
        self.active = 0
        self.peak = 0

    async def generate(self, question, context):
        self.calls.append(question)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.005)
            if question == "Unknown?":
                return GroundedAnswer(
                    answerable=False,
                    answer=None,
                    evidence=[],
                    explanation="The cloud provider is named, but nothing else is covered.",
                )
            if question == "Failure?":
                raise ServiceError("LLM_TIMEOUT", "Provider timed out.", 504)
            chunk = next((c for c in context if "AWS" in c.text), context[0])
            return GroundedAnswer(
                answerable=True,
                answer="AWS",
                evidence=[EvidenceQuote(chunk_id=chunk.chunk_id, quote="AWS")],
                explanation=None,
            )
        finally:
            self.active -= 1


@pytest.mark.parametrize(
    "filename,content",
    [
        ("source.json", b'{"cloud":"AWS","region":"US"}'),
        ("source.pdf", (Path(__file__).parent / "fixtures/document.pdf").read_bytes()),
    ],
)
def test_full_endpoint_with_real_ingestion_retrieval_and_partial_failure(filename, content):
    generator = FakeGenerator()
    settings = Settings(_env_file=None, max_concurrent_generations=2)
    app = create_app(settings)
    store = ChromaVectorStore()
    before = store.client.count_collections()

    @asynccontextmanager
    async def factory():
        yield QAService(
            RetrievalService(FakeEmbeddings(), store),
            generator,
            settings,
            app.state.generation_semaphore,
        )

    app.dependency_overrides[get_qa_service_factory] = lambda: factory

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/v1/qa",
                files={
                    "questions_file": (
                        "q.json",
                        json.dumps(["Cloud?", "Unknown?", "Failure?", "Cloud?"]),
                        "application/json",
                    ),
                    "document_file": (filename, content, "application/octet-stream"),
                },
            )

    response = asyncio.run(run())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert [r["status"] for r in body["results"]] == ["answered", "not_found", "error", "answered"]
    assert body["results"][0] == body["results"][3]
    assert body["results"][1]["answer"] == "not found"
    assert body["results"][1]["explanation"] == (
        "The cloud provider is named, but nothing else is covered."
    )
    assert body["results"][1]["citations"] == []
    assert "explanation" not in body["results"][0]
    assert body["results"][2]["answer"] is None
    assert body["metadata"]["questions_received"] == 4
    assert body["metadata"]["questions_answered"] == 2
    assert body["metadata"]["questions_not_found"] == 1
    assert body["metadata"]["questions_failed"] == 1
    citation = body["results"][0]["citations"][0]
    assert citation["snippet"] == "AWS"
    assert (
        citation.get("page") == 1
        if filename.endswith(".pdf")
        else citation.get("json_path") == '$["cloud"]'
    )
    assert sorted(generator.calls) == ["Cloud?", "Failure?", "Unknown?"]
    assert generator.peak == 2
    assert store.client.count_collections() == before


def sample():
    return IngestedDocument(ParsedDocument(()), (DocumentChunk("a", "AWS", "pdf", page=1),))


def test_concurrency_shared_across_requests():
    async def run():
        generator = FakeGenerator()
        settings = Settings(_env_file=None, max_concurrent_generations=2)
        semaphore = asyncio.Semaphore(2)

        def service():
            return QAService(
                RetrievalService(FakeEmbeddings(), ChromaVectorStore()),
                generator,
                settings,
                semaphore,
            )

        responses = await asyncio.gather(
            service().answer(sample(), ["A", "B", "C"], uuid4()),
            service().answer(sample(), ["D", "E", "F"], uuid4()),
        )
        assert generator.peak == 2
        assert [r.question for r in responses[0].results] == ["A", "B", "C"]

    asyncio.run(run())


def test_hanging_question_times_out_without_losing_success():
    class Generator(FakeGenerator):
        async def generate(self, question, context):
            if question == "Hang":
                await asyncio.sleep(10)
            return await super().generate(question, context)

    async def run():
        service = QAService(
            RetrievalService(FakeEmbeddings(), ChromaVectorStore()),
            Generator(),
            Settings(_env_file=None, llm_timeout_seconds=0.1),
        )
        return await service.answer(sample(), ["Hang", "Okay"], uuid4())

    response = asyncio.run(run())
    assert [r.status for r in response.results] == ["error", "answered"]
    assert response.results[0].error.code == "LLM_TIMEOUT"


def test_embedding_failure_is_request_level_error():
    class BrokenEmbeddings:
        async def embed(self, texts):
            raise ServiceError("EMBEDDING_TIMEOUT", "Embedding provider timed out.", 504)

    app = create_app(Settings(_env_file=None))

    @asynccontextmanager
    async def factory():
        yield QAService(
            RetrievalService(BrokenEmbeddings(), ChromaVectorStore()),
            FakeGenerator(),
            app.state.settings,
        )

    app.dependency_overrides[get_qa_service_factory] = lambda: factory

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/v1/qa",
                files={
                    "questions_file": ("q.json", b'["Cloud?"]', "application/json"),
                    "document_file": ("d.json", b'{"cloud":"AWS"}', "application/json"),
                },
            )

    response = asyncio.run(run())
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "EMBEDDING_TIMEOUT"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


def test_all_question_failures_still_return_result_list():
    async def run():
        service = QAService(
            RetrievalService(FakeEmbeddings(), ChromaVectorStore()),
            FakeGenerator(),
            Settings(_env_file=None),
        )
        return await service.answer(sample(), ["Failure?", "Failure?"], uuid4())

    response = asyncio.run(run())
    assert response.metadata.questions_failed == 2
    assert all(r.status == "error" for r in response.results)


def test_cancellation_stops_generation_and_releases_capacity():
    async def run():
        started = asyncio.Event()
        stopped = asyncio.Event()

        class Generator:
            async def generate(self, question, context):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

        semaphore = asyncio.Semaphore(1)
        service = QAService(
            RetrievalService(FakeEmbeddings(), ChromaVectorStore()),
            Generator(),
            Settings(_env_file=None),
            semaphore,
        )
        task = asyncio.create_task(service.answer(sample(), ["A", "B"], uuid4()))
        await asyncio.wait_for(started.wait(), timeout=3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()
        await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
        semaphore.release()

    asyncio.run(run())


def test_unexpected_failure_has_safe_error_envelope():
    app = create_app(Settings(_env_file=None))

    @app.get("/test-failure")
    async def fail():
        raise RuntimeError("secret document text")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.get("/test-failure")

    response = asyncio.run(run())
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "secret" not in response.text
    assert response.headers["X-Request-ID"] == response.json()["error"]["request_id"]
