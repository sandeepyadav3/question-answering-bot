from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.answers import ContractModel, ErrorDetail, QuestionResult


class HealthResponse(ContractModel):
    status: Literal["ok"] = "ok"


class RequestError(ErrorDetail):
    request_id: UUID


class ErrorResponse(ContractModel):
    error: RequestError


class ResponseMetadata(ContractModel):
    questions_received: int = Field(ge=1)
    questions_answered: int = Field(ge=0)
    questions_not_found: int = Field(ge=0)
    questions_failed: int = Field(ge=0)
    chunk_count: int = Field(ge=1)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)


class QAResponse(ContractModel):
    request_id: UUID
    results: list[QuestionResult] = Field(min_length=1)
    metadata: ResponseMetadata

    @model_validator(mode="after")
    def validate_counts(self) -> "QAResponse":
        actual = (
            len(self.results),
            sum(r.status == "answered" for r in self.results),
            sum(r.status == "not_found" for r in self.results),
            sum(r.status == "error" for r in self.results),
        )
        reported = (
            self.metadata.questions_received,
            self.metadata.questions_answered,
            self.metadata.questions_not_found,
            self.metadata.questions_failed,
        )
        if actual != reported:
            raise ValueError("metadata question counts must match results")
        return self
