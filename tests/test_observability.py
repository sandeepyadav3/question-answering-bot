import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

import httpx
import pytest
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.logging import (
    JSONFormatter,
    RequestMetrics,
    begin_provider_call,
    emit,
    logger,
    record_usage,
    request_metrics,
)
from app.dependencies import get_qa_service_factory
from app.main import create_app
from app.services.answer_generator import OpenAIAnswerGenerator
from app.services.embeddings import OpenAIEmbeddingService
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import ChromaVectorStore


@pytest.fixture
def captured():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)


def events(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


@pytest.mark.parametrize("provided", [str(uuid4()), "malformed-secret-request-id", "x" * 100])
def test_request_ids_and_validation_logs(provided, captured):
    app = create_app(Settings(_env_file=None))

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post("/api/v1/qa", headers={"X-Request-ID": provided})

    response = asyncio.run(run())
    actual = response.headers["X-Request-ID"]
    UUID(actual)
    if len(provided) == 36:
        assert actual == provided
    else:
        assert actual != provided
        assert provided not in captured.getvalue()
    assert response.json()["error"]["request_id"] == actual
    completed = [e for e in events(captured) if e["event"] == "request_completed"]
    assert len(completed) == 1
    assert completed[0]["http_status"] == 422
    assert completed[0]["request_id"] == actual
    assert request_metrics.get() is None


def test_concurrent_full_sdk_pipeline_logs_usage_without_content(captured):
    settings = Settings(_env_file=None)
    app = create_app(settings)

    def provider(request):
        body = json.loads(request.content)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"index": index, "embedding": [1.0, 0.0]}
                        for index in range(len(body["input"]))
                    ],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
            )
        payload = json.loads(body["messages"][1]["content"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "answerable": True,
                                    "answer": "SECRET_ANSWER",
                                    "evidence": [
                                        {
                                            "chunk_id": payload["chunks"][0]["chunk_id"],
                                            "quote": "SECRET_DOCUMENT",
                                        }
                                    ],
                                    "explanation": None,
                                }
                            ),
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    @asynccontextmanager
    async def factory():
        async with AsyncOpenAI(
            api_key="SECRET_API_KEY",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider)),
        ) as client:
            yield QAService(
                RetrievalService(OpenAIEmbeddingService(client, settings), ChromaVectorStore()),
                OpenAIAnswerGenerator(client, settings),
                settings,
                app.state.generation_semaphore,
            )

    app.dependency_overrides[get_qa_service_factory] = lambda: factory
    ids = [str(uuid4()), str(uuid4())]

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:

            async def one(index):
                return await client.post(
                    "/api/v1/qa",
                    headers={"X-Request-ID": ids[index]},
                    files={
                        "questions_file": (
                            "SECRET_FILENAME.json",
                            json.dumps(["SECRET_Q1", "SECRET_Q2"][: index + 1]),
                            "application/json",
                        ),
                        "document_file": (
                            "SECRET_SOURCE.json",
                            b'{"cloud":"SECRET_DOCUMENT"}',
                            "application/json",
                        ),
                    },
                )

            return await asyncio.gather(one(0), one(1))

    responses = asyncio.run(run())
    assert all(response.status_code == 200 for response in responses)
    assert "SECRET_" not in captured.getvalue()
    completed = {e["request_id"]: e for e in events(captured) if e["event"] == "request_completed"}
    assert set(completed) == set(ids)
    for index, request_id in enumerate(ids):
        event = completed[request_id]
        assert event["question_count"] == index + 1
        assert event["answered_count"] == index + 1
        assert event["generation_calls"] == index + 1
        assert event["embedding_calls"] == 2
        assert event["embedding_input_tokens"] == 6
        assert event["generation_input_tokens"] == 10 * (index + 1)
        assert event["generation_output_tokens"] == 2 * (index + 1)
        assert event["usage_missing_calls"] == 0
    assert request_metrics.get() is None


def test_formatter_drops_arbitrary_fields_and_exception_text(captured):
    create_app(Settings(_env_file=None))
    emit("safe_event", document="SECRET_DOCUMENT", prompt="SECRET_PROMPT", request_id=str(uuid4()))
    record = logging.LogRecord("document_qa", logging.ERROR, "", 1, "SECRET_MESSAGE", (), None)
    assert "SECRET" not in JSONFormatter().format(record)
    assert "SECRET" not in captured.getvalue()


def test_unknown_usage_is_not_reported_as_measured_zero():
    token = request_metrics.set(RequestMetrics(str(uuid4())))
    try:
        begin_provider_call("generation")
        record_usage("generation", None)
        assert request_metrics.get().generation_calls == 1
        assert request_metrics.get().usage_missing_calls == 1
    finally:
        request_metrics.reset(token)


def test_processing_deadline_cancels_hung_work(captured):
    settings = Settings(_env_file=None, request_timeout_seconds=0.2)
    app = create_app(settings)
    closed = []

    class Service:
        async def answer(self, *args):
            await asyncio.Event().wait()

    @asynccontextmanager
    async def factory():
        try:
            yield Service()
        finally:
            closed.append(True)

    app.dependency_overrides[get_qa_service_factory] = lambda: factory

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/v1/qa",
                files={
                    "questions_file": ("q.json", b'["Q"]', "application/json"),
                    "document_file": ("d.json", b'{"x":"AWS"}', "application/json"),
                },
            )

    response = asyncio.run(run())
    assert response.status_code == 504
    assert response.json()["error"]["code"] == "REQUEST_TIMEOUT"
    assert closed == [True]
    assert [e for e in events(captured) if e["event"] == "request_completed"][0][
        "http_status"
    ] == 504


def test_deadline_while_waiting_for_generation_does_not_leak_tasks_or_permits():
    from app.services.answer_generator import GroundedAnswer

    settings = Settings(_env_file=None, request_timeout_seconds=0.2, max_concurrent_generations=1)
    app = create_app(settings)
    calls = []

    class Embeddings:
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class Generator:
        async def generate(self, question, context):
            calls.append(question)
            return GroundedAnswer(answerable=False, answer=None, evidence=[], explanation=None)

    @asynccontextmanager
    async def factory():
        yield QAService(
            RetrievalService(Embeddings(), ChromaVectorStore()),
            Generator(),
            settings,
            app.state.generation_semaphore,
        )

    app.dependency_overrides[get_qa_service_factory] = lambda: factory

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:

            async def post():
                return await client.post(
                    "/api/v1/qa",
                    files={
                        "questions_file": ("q.json", b'["Q"]', "application/json"),
                        "document_file": ("d.json", b'{"x":"AWS"}', "application/json"),
                    },
                )

            await app.state.generation_semaphore.acquire()
            try:
                first = await post()
                assert first.status_code == 504
                assert app.state.generation_semaphore.locked()
                assert calls == []
            finally:
                app.state.generation_semaphore.release()
            second = await post()
            assert second.status_code == 200
            assert calls == ["Q"]

    asyncio.run(run())
