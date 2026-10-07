from functools import lru_cache
from pathlib import Path

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tiktoken.load import load_tiktoken_bpe

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.models.document_chunk import DocumentChunk, ParsedDocument, SourceRecord


@lru_cache(maxsize=1)
def get_tokenizer() -> tiktoken.Encoding:
    # Public cl100k_base vocabulary and pattern from tiktoken (MIT licensed).
    ranks = load_tiktoken_bpe(
        str(Path(__file__).resolve().parents[1] / "data" / "cl100k_base.tiktoken"),
        expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    )
    return tiktoken.Encoding(
        name="document_qa_cl100k",
        pat_str=(
            r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+"
            r"| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
        ),
        mergeable_ranks=ranks,
        special_tokens={},
    )


def _source_document(record: SourceRecord, record_index: int) -> Document:
    metadata = {"source_type": record.source_type, "record_index": record_index}
    if record.page is not None:
        metadata["page"] = record.page
    if record.json_path is not None:
        metadata["json_path"] = record.json_path
    return Document(page_content=record.text, metadata=metadata)


def _split_record(
    document: Document, prefix: str, settings: Settings, tokenizer: tiktoken.Encoding
) -> list[Document]:
    prefix_tokens = len(tokenizer.encode_ordinary(prefix))
    available_tokens = settings.chunk_size_tokens - prefix_tokens
    if available_tokens <= 0:
        raise ServiceError(
            "SOURCE_PATH_TOO_LONG", "JSON path exceeds the chunk token budget.", 422
        )

    def token_length(text: str) -> int:
        return len(tokenizer.encode_ordinary(text))

    # Normally token counts for ``prefix + text`` are no larger than the two
    # independent counts. Recheck the authoritative retrieval text anyway and
    # reduce the content budget if a tokenizer boundary behaves differently.
    while available_tokens > 0:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_tokens,
            chunk_overlap=min(settings.chunk_overlap_tokens, available_tokens - 1),
            length_function=token_length,
            keep_separator=True,
            add_start_index=True,
            strip_whitespace=False,
        )
        chunks = splitter.split_documents([document])
        if all(
            len(tokenizer.encode_ordinary(prefix + chunk.page_content))
            <= settings.chunk_size_tokens
            for chunk in chunks
        ):
            return chunks
        available_tokens -= 1
    raise ServiceError("CHUNK_BUDGET_TOO_SMALL", "Token budget cannot fit source text.", 422)


def _find_chunk_start(
    source: str,
    text: str,
    previous_start: int,
    previous_end: int,
    overlap_tokens: int,
    tokenizer: tiktoken.Encoding,
) -> int:
    if previous_start < 0:
        return source.find(text)
    candidate = source.find(text, previous_start + 1)
    while 0 <= candidate <= previous_end:
        overlap = source[candidate:previous_end]
        if len(tokenizer.encode_ordinary(overlap)) <= overlap_tokens:
            return candidate
        candidate = source.find(text, candidate + 1)
    return -1


def chunk_document(document: ParsedDocument, settings: Settings) -> list[DocumentChunk]:
    tokenizer = get_tokenizer()
    chunks = []
    for record_index, record in enumerate(document.records):
        prefix = f"{record.json_path}: " if record.json_path else ""
        langchain_document = _source_document(record, record_index)
        split_documents = _split_record(langchain_document, prefix, settings, tokenizer)
        previous_start = -1
        previous_end = 0
        for chunk_index, split_document in enumerate(split_documents):
            if len(chunks) >= settings.max_document_chunks:
                raise ServiceError("TOO_MANY_CHUNKS", "Document exceeds the chunk limit.", 422)
            text = split_document.page_content
            start = _find_chunk_start(
                record.text,
                text,
                previous_start,
                previous_end,
                settings.chunk_overlap_tokens,
                tokenizer,
            )
            if start < 0:
                raise ServiceError(
                    "CHUNK_PROVENANCE_ERROR",
                    "Unable to preserve source chunk offsets.",
                    500,
                )
            previous_start = start
            end = start + len(text)
            previous_end = end
            stem = f"pdf-p{record.page}" if record.source_type == "pdf" else f"json-r{record_index}"
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{stem}-c{chunk_index}",
                    text=text,
                    source_type=record.source_type,
                    page=record.page,
                    json_path=record.json_path,
                    start_char=start,
                    end_char=end,
                )
            )
    return chunks
