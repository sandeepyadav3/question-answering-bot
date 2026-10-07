import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.services.embeddings import OpenAIEmbeddingService, validate_vectors


def run_embed(handler, texts, **config):
    async def run():
        async with AsyncOpenAI(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            service = OpenAIEmbeddingService(client, Settings(_env_file=None, **config))
            return await service.embed(texts)

    return asyncio.run(run())


def test_batches_and_reordered_provider_response():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(int(text)), 1.0], "object": "embedding"}
                    for i, text in reversed(list(enumerate(body["input"])))
                ],
                "model": body["model"],
                "object": "list",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    result = run_embed(handler, ["1", "2", "3", "4", "5"], embedding_batch_size=2)
    assert [call["input"] for call in calls] == [["1", "2"], ["3", "4"], ["5"]]
    assert result == [[float(i), 1.0] for i in range(1, 6)]
    assert all(call["encoding_format"] == "float" for call in calls)


def test_token_budget_batches():
    calls = []

    def handler(request):
        inputs = json.loads(request.content)["input"]
        calls.append(inputs)
        return httpx.Response(
            200, json={"data": [{"index": i, "embedding": [1.0, 2.0]} for i in range(len(inputs))]}
        )

    run_embed(handler, ["hello world"] * 3, embedding_batch_tokens=3)
    assert len(calls) == 3


@pytest.mark.parametrize(
    "status,code",
    [
        (429, "EMBEDDING_RATE_LIMIT"),
        (500, "EMBEDDING_PROVIDER_ERROR"),
        (401, "EMBEDDING_PROVIDER_ERROR"),
    ],
)
def test_provider_errors_are_sanitized_without_retries(status, code):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "sensitive provider details"}})

    with pytest.raises(ServiceError) as exc:
        run_embed(handler, ["hello"])
    assert exc.value.code == code
    assert "sensitive" not in exc.value.message
    assert len(calls) == 1


def test_timeout():
    async def handler(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={})

    with pytest.raises(ServiceError) as exc:
        run_embed(handler, ["hello"], embedding_timeout_seconds=0.01)
    assert exc.value.status_code == 504


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 1, "embedding": [1.0, 2.0]}],
        [{"index": 0, "embedding": [0.0, 0.0]}],
    ],
)
def test_invalid_response(data):
    with pytest.raises(ServiceError) as exc:
        run_embed(lambda request: httpx.Response(200, json={"data": data}), ["hello"])
    assert exc.value.code == "INVALID_EMBEDDINGS"


@pytest.mark.parametrize(
    "vectors,count",
    [
        ([[1.0, 2.0]], 2),
        ([[1.0], [1.0, 2.0]], 2),
        ([[float("nan")]], 1),
        ([[float("inf")]], 1),
        ([[0.0, 0.0]], 1),
        ([[True]], 1),
        ([[]], 1),
    ],
)
def test_invalid_vectors(vectors, count):
    with pytest.raises(ServiceError):
        validate_vectors(vectors, count)


def test_empty_and_oversized_input_make_no_calls():
    def handler(request):
        raise AssertionError("Unexpected call")

    assert run_embed(handler, []) == []
    with pytest.raises(ServiceError) as exc:
        run_embed(handler, ["hello world"], embedding_batch_tokens=1)
    assert exc.value.code == "EMBEDDING_INPUT_TOO_LONG"
