from dataclasses import dataclass

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.models.document_chunk import DocumentChunk, ParsedDocument
from app.services.chunking import chunk_document
from app.services.document_parser import parse_document


@dataclass(frozen=True)
class IngestedDocument:
    document: ParsedDocument
    chunks: tuple[DocumentChunk, ...]


async def ingest_document(
    contents: bytes, source_type: str, settings: Settings
) -> IngestedDocument:
    def process():
        document = parse_document(contents, source_type, settings)
        return IngestedDocument(document, tuple(chunk_document(document, settings)))

    # AnyIO's shared worker pool bounds offloaded synchronous work.
    return await run_in_threadpool(process)
