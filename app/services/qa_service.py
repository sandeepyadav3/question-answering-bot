import asyncio
from time import perf_counter
from uuid import UUID

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.core.logging import emit, question_index, request_metrics
from app.schemas.answers import ErrorDetail, ErrorResult
from app.schemas.responses import QAResponse, ResponseMetadata
from app.services.answer_generator import AnswerGenerator, validate_grounding
from app.services.ingestion import IngestedDocument
from app.services.retrieval import RetrievalService


class QAService:
    def __init__(
        self,
        retrieval: RetrievalService,
        generator: AnswerGenerator,
        settings: Settings,
        semaphore: asyncio.Semaphore | None = None,
    ):
        self.retrieval = retrieval
        self.generator = generator
        self.settings = settings
        self.semaphore = (
            semaphore
            if semaphore is not None
            else asyncio.Semaphore(settings.max_concurrent_generations)
        )

    async def answer(
        self,
        ingestion: IngestedDocument,
        questions: list[str],
        request_id: UUID,
        started_at: float | None = None,
    ) -> QAResponse:
        started_at = perf_counter() if started_at is None else started_at
        retrieval_started = perf_counter()
        retrieved = await self.retrieval.retrieve(ingestion.chunks, questions)
        return await self._complete(
            retrieved, len(ingestion.chunks), request_id, started_at, retrieval_started
        )

    async def answer_saved(
        self,
        collection_name: str,
        chunk_count: int,
        questions: list[str],
        request_id: UUID,
        started_at: float | None = None,
    ) -> QAResponse:
        started_at = perf_counter() if started_at is None else started_at
        retrieval_started = perf_counter()
        retrieved = await self.retrieval.retrieve_saved(collection_name, questions)
        return await self._complete(
            retrieved, chunk_count, request_id, started_at, retrieval_started
        )

    async def _complete(
        self, retrieved, chunk_count, request_id, started_at, retrieval_started
    ) -> QAResponse:
        emit(
            "stage_completed",
            stage="retrieval",
            latency_ms=(perf_counter() - retrieval_started) * 1000,
        )

        async def answer_one(question, context):
            try:
                async with self.semaphore:
                    # Timeout begins after acquiring capacity, not while waiting in the queue.
                    async with asyncio.timeout(self.settings.llm_timeout_seconds):
                        output = await self.generator.generate(question, context)
                    return validate_grounding(question, output, context)
            except TimeoutError:
                return ErrorResult(
                    question=question,
                    error=ErrorDetail(
                        code="LLM_TIMEOUT",
                        message="Unable to process this question: provider timed out.",
                    ),
                )
            except ServiceError as exc:
                return ErrorResult(
                    question=question, error=ErrorDetail(code=exc.code, message=exc.message)
                )
            # Cancellation deliberately propagates; do not turn it into a question error.

        async def observed_answer(index, question, context):
            token = question_index.set(index)
            started = perf_counter()
            status = "cancelled"
            try:
                result = await answer_one(question, context)
                status = result.status
                return result
            except Exception:
                status = "error"
                raise
            finally:
                emit(
                    "question_completed",
                    status=status,
                    latency_ms=(perf_counter() - started) * 1000,
                )
                question_index.reset(token)

        tasks = [
            asyncio.create_task(observed_answer(index, question, context))
            for index, (question, context) in enumerate(
                zip(retrieved.unique_questions, retrieved.contexts, strict=True)
            )
        ]
        try:
            unique_results = await asyncio.gather(*tasks)
        finally:
            # Also cancel sibling calls if an unexpected programming failure escapes.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        results = [unique_results[index] for index in retrieved.question_indices]
        metrics = request_metrics.get()
        if metrics:
            metrics.answered_count = sum(r.status == "answered" for r in results)
            metrics.not_found_count = sum(r.status == "not_found" for r in results)
            metrics.failed_count = sum(r.status == "error" for r in results)
        return QAResponse(
            request_id=request_id,
            results=results,
            metadata=ResponseMetadata(
                questions_received=len(results),
                questions_answered=sum(r.status == "answered" for r in results),
                questions_not_found=sum(r.status == "not_found" for r in results),
                questions_failed=sum(r.status == "error" for r in results),
                chunk_count=chunk_count,
                latency_ms=(perf_counter() - started_at) * 1000,
            ),
        )
