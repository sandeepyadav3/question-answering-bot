from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

NonBlankText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PDFCitation(ContractModel):
    source_type: Literal["pdf"] = "pdf"
    page: int = Field(ge=1)
    snippet: NonBlankText


class JSONCitation(ContractModel):
    source_type: Literal["json"] = "json"
    json_path: NonBlankText
    snippet: NonBlankText


Citation = Annotated[PDFCitation | JSONCitation, Field(discriminator="source_type")]


class ErrorDetail(ContractModel):
    code: NonBlankText
    message: NonBlankText


class AnsweredResult(ContractModel):
    question: NonBlankText
    status: Literal["answered"] = "answered"
    answer: NonBlankText
    citations: list[Citation] = Field(min_length=1)


class NotFoundResult(ContractModel):
    question: NonBlankText
    status: Literal["not_found"] = "not_found"
    # Preserve the assignment's required literal rather than alternative wording.
    answer: Literal["not found"] = "not found"
    # Optional one-sentence note on what the document covers and what is missing.
    explanation: NonBlankText | None = None
    citations: list[Citation] = Field(default_factory=list, max_length=0)


class ErrorResult(ContractModel):
    question: NonBlankText
    status: Literal["error"] = "error"
    answer: None = None
    citations: list[Citation] = Field(default_factory=list, max_length=0)
    error: ErrorDetail


QuestionResult = Annotated[
    AnsweredResult | NotFoundResult | ErrorResult, Field(discriminator="status")
]
