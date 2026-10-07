from collections.abc import Sequence
from dataclasses import dataclass

from starlette.concurrency import run_in_threadpool

from app.models.document_chunk import DocumentChunk
from app.services.embeddings import EmbeddingService, validate_vectors
from app.services.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    unique_questions: tuple[str, ...]
    contexts: tuple[tuple[DocumentChunk, ...], ...]
    # Map each original question to its unique question/context index.
    question_indices: tuple[int, ...]


class RetrievalService:
    def __init__(
        self,
        embeddings: EmbeddingService,
        store: ChromaVectorStore,
        top_k: int = 5,
        fetch_k: int = 20,
        min_similarity: float = 0.15,
        mmr_lambda: float = 0.7,
    ):
        self.embeddings = embeddings
        self.store = store
        self.top_k = top_k
        self.fetch_k = fetch_k
        self.min_similarity = min_similarity
        self.mmr_lambda = mmr_lambda

    async def retrieve(
        self, chunks: Sequence[DocumentChunk], questions: Sequence[str]
    ) -> RetrievalResult:
        unique = tuple(dict.fromkeys(questions))
        if not chunks or not unique:
            raise ValueError("Chunks and questions must be nonempty")
        indices = {question: index for index, question in enumerate(unique)}
        document_vectors = await self.embeddings.embed([chunk.retrieval_text for chunk in chunks])
        document_vectors = validate_vectors(document_vectors, len(chunks))
        question_vectors = validate_vectors(await self.embeddings.embed(unique), len(unique))
        # Keep create/add/query/delete together in one worker, including its finally.
        # Cancellation of the awaiting task cannot strand a partially built index.
        contexts = await run_in_threadpool(
            self.store.search,
            chunks,
            document_vectors,
            question_vectors,
            self.top_k,
            self.fetch_k,
            self.min_similarity,
            self.mmr_lambda,
        )
        return RetrievalResult(
            unique,
            tuple(tuple(context) for context in contexts),
            tuple(indices[question] for question in questions),
        )
