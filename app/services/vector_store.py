import math
from collections.abc import Sequence
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.exceptions import ServiceError
from app.models.document_chunk import DocumentChunk
from app.services.embeddings import validate_vectors


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm)


def _mmr_rerank(
    candidate_ids: Sequence[str],
    by_vector: dict[str, list[float]],
    question_vector: Sequence[float],
    top_k: int,
    mmr_lambda: float,
) -> list[str]:
    remaining = set(candidate_ids)
    relevance = {
        chunk_id: _cosine_similarity(by_vector[chunk_id], question_vector)
        for chunk_id in candidate_ids
    }
    selected: list[str] = []
    while remaining and len(selected) < top_k:
        if not selected:
            winner = min(remaining, key=lambda chunk_id: (-relevance[chunk_id], chunk_id))
        else:
            def rank(chunk_id: str) -> tuple[float, str]:
                redundancy = max(
                    _cosine_similarity(by_vector[chunk_id], by_vector[chosen])
                    for chosen in selected
                )
                score = mmr_lambda * relevance[chunk_id] - (1 - mmr_lambda) * redundancy
                return (-score, chunk_id)

            winner = min(remaining, key=rank)
        selected.append(winner)
        remaining.remove(winner)
    return selected


def chroma_settings() -> ChromaSettings:
    return ChromaSettings(anonymized_telemetry=False)


def _chunk_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    item: dict[str, str | int] = {
        "source_type": chunk.source_type,
        "chunk_id": chunk.chunk_id,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }
    if chunk.page is not None:
        item["page"] = chunk.page
    if chunk.json_path is not None:
        item["json_path"] = chunk.json_path
    return item


def _chunk_from_stored(chunk_id: str, text: str, metadata: dict) -> DocumentChunk:
    page = metadata.get("page")
    json_path = metadata.get("json_path")
    return DocumentChunk(
        chunk_id=chunk_id,
        text=text,
        source_type=metadata["source_type"],
        page=int(page) if page is not None else None,
        json_path=str(json_path) if json_path else None,
        start_char=int(metadata.get("start_char") or 0),
        end_char=int(metadata.get("end_char") or 0),
    )


class ChromaVectorStore:
    def __init__(self, client=None):
        self.client = (
            client
            if client is not None
            else chromadb.EphemeralClient(settings=chroma_settings())
        )

    def _add(
        self,
        collection,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
    ):
        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                embeddings=list(embeddings[start : start + batch_size]),
                documents=[chunk.text for chunk in batch],
                metadatas=[_chunk_metadata(chunk) for chunk in batch],
            )

    def _query(
        self,
        collection,
        chunks: Sequence[DocumentChunk],
        document_vectors: Sequence[Sequence[float]],
        question_vectors: Sequence[Sequence[float]],
        top_k: int,
        fetch_k: int,
        min_similarity: float,
        mmr_lambda: float,
    ) -> list[list[DocumentChunk]]:
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        if len(by_id) != len(chunks):
            raise ValueError("Chunk IDs must be unique within a document")
        documents = validate_vectors(document_vectors, len(chunks))
        questions = validate_vectors(question_vectors, len(question_vectors))
        if not questions:
            return []
        if len(questions[0]) != len(documents[0]):
            raise ServiceError(
                "INVALID_EMBEDDINGS", "Document and question dimensions differ.", 502
            )
        by_vector = {
            chunk.chunk_id: vector for chunk, vector in zip(chunks, documents, strict=True)
        }
        result = collection.query(
            query_embeddings=questions,
            n_results=min(fetch_k, len(chunks)),
            include=["distances"],
        )
        contexts = []
        for ids, distances, question in zip(
            result["ids"], result["distances"], questions, strict=True
        ):
            candidates = [
                chunk_id
                for chunk_id, distance in zip(ids, distances, strict=True)
                if 1 - distance >= min_similarity
            ]
            ranked = _mmr_rerank(
                candidates, by_vector, question, min(top_k, len(candidates)), mmr_lambda
            )
            contexts.append([by_id[chunk_id] for chunk_id in ranked])
        return contexts

    def save(
        self,
        collection_name: str,
        chunks: Sequence[DocumentChunk],
        document_vectors: Sequence[Sequence[float]],
    ) -> None:
        if not chunks:
            raise ValueError("Nonempty chunks are required")
        embeddings = validate_vectors(document_vectors, len(chunks))
        collection = None
        try:
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,
                configuration={"hnsw": {"space": "cosine"}},
            )
            self._add(collection, chunks, embeddings)
        except ServiceError:
            raise
        except Exception as exc:
            if collection is not None:
                self.client.delete_collection(name=collection_name)
            raise ServiceError("VECTOR_STORE_ERROR", "Document indexing failed.", 500) from exc

    def search_saved(
        self,
        collection_name: str,
        question_vectors: Sequence[Sequence[float]],
        top_k: int,
        fetch_k: int | None = None,
        min_similarity: float = -1,
        mmr_lambda: float = 1,
    ) -> list[list[DocumentChunk]]:
        fetch_k = top_k if fetch_k is None else fetch_k
        try:
            collection = self.client.get_collection(collection_name)
            stored = collection.get(include=["embeddings", "documents", "metadatas"])
            chunks = [
                _chunk_from_stored(chunk_id, text, metadata)
                for chunk_id, text, metadata in zip(
                    stored["ids"], stored["documents"], stored["metadatas"], strict=True
                )
            ]
            vectors = [[float(value) for value in vector] for vector in stored["embeddings"]]
            return self._query(
                collection,
                chunks,
                vectors,
                question_vectors,
                top_k,
                fetch_k,
                min_similarity,
                mmr_lambda,
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                "VECTOR_STORE_ERROR", "Saved document retrieval failed.", 500
            ) from exc

    def search(
        self,
        chunks: Sequence[DocumentChunk],
        document_vectors: Sequence[Sequence[float]],
        question_vectors: Sequence[Sequence[float]],
        top_k: int,
        fetch_k: int | None = None,
        min_similarity: float = -1,
        mmr_lambda: float = 1,
    ) -> list[list[DocumentChunk]]:
        if not chunks or top_k < 1:
            raise ValueError("Nonempty chunks and positive top_k are required")
        fetch_k = top_k if fetch_k is None else fetch_k
        if fetch_k < top_k:
            raise ValueError("fetch_k must be at least top_k")
        if not -1 <= min_similarity <= 1:
            raise ValueError("min_similarity must be between -1 and 1")
        if not 0 <= mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")
        documents = validate_vectors(document_vectors, len(chunks))
        questions = validate_vectors(question_vectors, len(question_vectors))
        if not questions:
            return []
        if len(questions[0]) != len(documents[0]):
            raise ServiceError(
                "INVALID_EMBEDDINGS", "Document and question dimensions differ.", 502
            )
        name = "qa_" + uuid4().hex
        collection = None
        try:
            collection = self.client.create_collection(
                name=name, embedding_function=None, configuration={"hnsw": {"space": "cosine"}}
            )
            self._add(collection, chunks, documents)
            # Resolve against authoritative source objects, not reconstructed DB metadata.
            return self._query(
                collection,
                chunks,
                documents,
                questions,
                top_k,
                fetch_k,
                min_similarity,
                mmr_lambda,
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("VECTOR_STORE_ERROR", "Document retrieval failed.", 500) from exc
        finally:
            if collection is not None:
                try:
                    self.client.delete_collection(name=name)
                except Exception as exc:
                    raise ServiceError(
                        "VECTOR_STORE_CLEANUP_ERROR", "Temporary index cleanup failed.", 500
                    ) from exc
