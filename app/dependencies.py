from contextlib import asynccontextmanager
from functools import partial

from fastapi import Request
from openai import AsyncOpenAI
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.services.answer_generator import OpenAIAnswerGenerator
from app.services.embeddings import OpenAIEmbeddingService
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import ChromaVectorStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


@asynccontextmanager
async def open_retrieval_service(settings: Settings):
    """Called by QA orchestration after input validation in the next phase."""
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise ServiceError("PROVIDER_NOT_CONFIGURED", "AI provider is not configured.", 503)
    async with AsyncOpenAI(
        api_key=key.get_secret_value(),
        max_retries=0,
        timeout=settings.embedding_timeout_seconds,
    ) as client:
        yield RetrievalService(
            OpenAIEmbeddingService(client, settings),
            await run_in_threadpool(ChromaVectorStore),
            top_k=settings.retrieval_top_k,
            fetch_k=settings.retrieval_fetch_k,
            min_similarity=settings.retrieval_min_similarity,
            mmr_lambda=settings.retrieval_mmr_lambda,
        )


@asynccontextmanager
async def open_qa_service(settings: Settings, semaphore):
    key = settings.openai_api_key
    if key is None or not key.get_secret_value().strip():
        raise ServiceError("PROVIDER_NOT_CONFIGURED", "AI provider is not configured.", 503)
    async with AsyncOpenAI(
        api_key=key.get_secret_value(),
        max_retries=0,
        timeout=max(settings.embedding_timeout_seconds, settings.llm_timeout_seconds),
    ) as client:
        retrieval = RetrievalService(
            OpenAIEmbeddingService(client, settings),
            await run_in_threadpool(ChromaVectorStore),
            top_k=settings.retrieval_top_k,
            fetch_k=settings.retrieval_fetch_k,
            min_similarity=settings.retrieval_min_similarity,
            mmr_lambda=settings.retrieval_mmr_lambda,
        )
        yield QAService(retrieval, OpenAIAnswerGenerator(client, settings), settings, semaphore)


def get_qa_service_factory(request: Request):
    # Resolving this dependency performs no provider I/O before upload validation.
    return partial(
        open_qa_service, request.app.state.settings, request.app.state.generation_semaphore
    )
