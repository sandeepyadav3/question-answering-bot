from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    service_name: str = Field(default="Document QA", min_length=1)
    max_document_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_questions_file_bytes: int = Field(default=1024 * 1024, gt=0)
    multipart_overhead_bytes: int = Field(default=65536, gt=0)
    max_questions: int = Field(default=25, gt=0)
    max_question_length: int = Field(default=2000, gt=0)
    max_pdf_pages: int = Field(default=200, gt=0)
    max_extracted_chars: int = Field(default=1_000_000, gt=0)
    max_json_depth: int = Field(default=64, gt=0)
    max_document_records: int = Field(default=10000, gt=0)
    max_document_chunks: int = Field(default=10000, gt=0)
    chunk_size_tokens: int = Field(default=800, gt=0)
    chunk_overlap_tokens: int = Field(default=120, ge=0)
    retrieval_top_k: int = Field(default=5, gt=0)
    retrieval_fetch_k: int = Field(default=20, gt=0)
    retrieval_min_similarity: float = Field(default=0.15, ge=-1, le=1, allow_inf_nan=False)
    retrieval_mmr_lambda: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    openai_api_key: SecretStr | None = None
    embedding_model: str = "text-embedding-3-small"
    generation_model: str = "gpt-4o-mini"
    max_answer_tokens: int = Field(default=800, gt=0, le=4096)
    embedding_batch_size: int = Field(default=64, gt=0, le=2048)
    embedding_batch_tokens: int = Field(default=16000, gt=0, le=300000)
    max_concurrent_generations: int = Field(default=4, gt=0)
    request_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)
    llm_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    embedding_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_related_limits(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        if self.retrieval_fetch_k < self.retrieval_top_k:
            raise ValueError("retrieval_fetch_k must be at least retrieval_top_k")
        return self
