from starlette.concurrency import run_in_threadpool

from app.services.document_store import DocumentLibrary, StoredDocument
from app.services.embeddings import EmbeddingService, validate_vectors
from app.services.ingestion import IngestedDocument
from app.services.qa_service import QAService


class DocumentService:
    def __init__(self, embeddings: EmbeddingService, library: DocumentLibrary, qa: QAService):
        self.embeddings = embeddings
        self.library = library
        self.qa = qa

    async def save(
        self, ingestion: IngestedDocument, byte_count: int, tenant_id: str
    ) -> StoredDocument:
        vectors = validate_vectors(
            await self.embeddings.embed([chunk.retrieval_text for chunk in ingestion.chunks]),
            len(ingestion.chunks),
        )
        return await run_in_threadpool(
            self.library.save,
            tenant_id,
            ingestion.chunks[0].source_type,
            ingestion.chunks,
            vectors,
            byte_count,
        )

    async def answer(self, document, questions, request_id, started_at):
        return await self.qa.answer_saved(
            document.collection_name,
            document.chunk_count,
            questions,
            request_id,
            started_at,
        )
