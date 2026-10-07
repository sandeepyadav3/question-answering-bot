from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SourceRecord:
    text: str
    source_type: Literal["pdf", "json"]
    page: int | None = None
    json_path: str | None = None


@dataclass(frozen=True)
class ParsedDocument:
    records: tuple[SourceRecord, ...]
    page_count: int = 0


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_type: Literal["pdf", "json"]
    page: int | None = None
    json_path: str | None = None
    # Exact offsets into the source record, useful for verification and citation assembly.
    start_char: int = 0
    end_char: int = 0

    @property
    def retrieval_text(self) -> str:
        return f"{self.json_path}: {self.text}" if self.json_path else self.text
