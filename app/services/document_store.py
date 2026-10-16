import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import chromadb

from app.core.exceptions import ServiceError
from app.models.document_chunk import DocumentChunk
from app.services.vector_store import ChromaVectorStore, chroma_settings


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    collection_name: str
    source_type: str
    chunk_count: int
    byte_count: int
    tenant_id: str


class DocumentLibrary:
    """Durable document index. Each upload owns one Chroma collection."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.store = ChromaVectorStore(
            chromadb.PersistentClient(
                path=str(directory / "chroma"), settings=chroma_settings()
            )
        )
        self._database = directory / "documents.sqlite"
        self._lock = threading.Lock()
        with self._connect() as connection:
            self._ensure_schema(connection)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                collection_name TEXT NOT NULL UNIQUE,
                source_type TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                byte_count INTEGER NOT NULL,
                tenant_id TEXT NOT NULL DEFAULT ''
            )
            """
        )
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(documents)")}
        if "tenant_id" not in columns:
            # Existing rows have no owner and stay unreachable to every tenant.
            connection.execute(
                "ALTER TABLE documents ADD COLUMN tenant_id TEXT NOT NULL DEFAULT ''"
            )

    def save(
        self,
        tenant_id: str,
        source_type: str,
        chunks: tuple[DocumentChunk, ...] | list[DocumentChunk],
        vectors: list[list[float]],
        byte_count: int,
    ) -> StoredDocument:
        if not tenant_id:
            raise ValueError("A tenant is required")
        document_id = str(uuid4())
        collection_name = "doc" + document_id.replace("-", "")
        with self._lock:
            self.store.save(collection_name, chunks, vectors)
            try:
                with self._connect() as connection:
                    connection.execute(
                        """
                        INSERT INTO documents (
                            document_id, collection_name, source_type, chunk_count,
                            byte_count, tenant_id
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document_id,
                            collection_name,
                            source_type,
                            len(chunks),
                            byte_count,
                            tenant_id,
                        ),
                    )
            except Exception as exc:
                self.store.client.delete_collection(name=collection_name)
                raise ServiceError("VECTOR_STORE_ERROR", "Document indexing failed.", 500) from exc
        return StoredDocument(
            document_id, collection_name, source_type, len(chunks), byte_count, tenant_id
        )

    def get(self, document_id: str, tenant_id: str) -> StoredDocument | None:
        if not tenant_id:
            return None
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ? AND tenant_id = ?",
                (document_id, tenant_id),
            ).fetchone()
        if row is None:
            return None
        return StoredDocument(
            row["document_id"],
            row["collection_name"],
            row["source_type"],
            row["chunk_count"],
            row["byte_count"],
            row["tenant_id"],
        )
