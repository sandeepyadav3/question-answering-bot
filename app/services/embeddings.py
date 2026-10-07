import asyncio
import math
from collections.abc import Sequence
from typing import Protocol

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.core.logging import begin_provider_call, record_usage
from app.services.chunking import get_tokenizer


class EmbeddingService(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def validate_vectors(vectors: Sequence[Sequence[float]], count: int) -> list[list[float]]:
    try:
        if len(vectors) != count:
            raise ValueError("Wrong vector count")
        dimension = len(vectors[0]) if vectors else 0
        result = []
        for vector in vectors:
            if not dimension or len(vector) != dimension:
                raise ValueError("Inconsistent dimensions")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector
            ):
                raise ValueError("Nonnumeric vector")
            values = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in values) or not any(values):
                raise ValueError("Non-finite or zero vector")
            result.append(values)
        return result
    except (TypeError, ValueError, OverflowError) as exc:
        raise ServiceError(
            "INVALID_EMBEDDINGS", "Embedding provider returned invalid vectors.", 502
        ) from exc


class OpenAIEmbeddingService:
    """Injected async client; caller owns its lifecycle. No SDK retries."""

    def __init__(self, client: AsyncOpenAI, settings: Settings):
        self.client = client
        self.settings = settings

    def _batches(self, texts: Sequence[str]) -> list[list[str]]:
        tokenizer = get_tokenizer()
        batches: list[list[str]] = []
        current: list[str] = []
        tokens = 0
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ServiceError(
                    "INVALID_EMBEDDING_INPUT", "Embedding text must be nonblank.", 422
                )
            size = len(tokenizer.encode_ordinary(text))
            if size > min(self.settings.embedding_batch_tokens, 8191):
                raise ServiceError(
                    "EMBEDDING_INPUT_TOO_LONG", "Text exceeds the embedding token limit.", 422
                )
            if current and (
                len(current) >= self.settings.embedding_batch_size
                or tokens + size > self.settings.embedding_batch_tokens
            ):
                batches.append(current)
                current, tokens = [], 0
            current.append(text)
            tokens += size
        if current:
            batches.append(current)
        return batches

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        batches = await run_in_threadpool(self._batches, texts)
        vectors = []
        try:
            for batch in batches:
                begin_provider_call("embedding")
                async with asyncio.timeout(self.settings.embedding_timeout_seconds):
                    response = await self.client.embeddings.create(
                        model=self.settings.embedding_model,
                        input=batch,
                        encoding_format="float",
                    )
                record_usage("embedding", response.usage)
                rows = sorted(response.data, key=lambda row: row.index)
                if [row.index for row in rows] != list(range(len(batch))):
                    raise ServiceError(
                        "INVALID_EMBEDDINGS", "Embedding response indices are invalid.", 502
                    )
                vectors.extend(validate_vectors([row.embedding for row in rows], len(batch)))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ServiceError(
                "INVALID_EMBEDDINGS", "Embedding response is malformed.", 502
            ) from exc
        except (TimeoutError, APITimeoutError) as exc:
            raise ServiceError("EMBEDDING_TIMEOUT", "Embedding provider timed out.", 504) from exc
        except RateLimitError as exc:
            raise ServiceError(
                "EMBEDDING_RATE_LIMIT", "Embedding provider is rate limited.", 502
            ) from exc
        except APIError as exc:
            raise ServiceError(
                "EMBEDDING_PROVIDER_ERROR", "Embedding provider request failed.", 502
            ) from exc
        return validate_vectors(vectors, len(texts))
