from time import perf_counter
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.upload_route import DocumentUploadRoute, QuestionsUploadRoute
from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.core.logging import request_metrics
from app.dependencies import (
    get_document_library,
    get_document_service_factory,
    get_settings,
    require_tenant,
)
from app.schemas.responses import DocumentCreated, ErrorResponse, QAResponse
from app.services.ingestion import ingest_document
from app.services.question_parser import parse_questions
from app.services.uploads import read_bounded, validate_file_type

upload_router = APIRouter(route_class=DocumentUploadRoute)
questions_router = APIRouter(route_class=QuestionsUploadRoute)


@upload_router.post(
    "/api/v1/documents",
    response_model=DocumentCreated,
    tags=["documents"],
    responses={code: {"model": ErrorResponse} for code in (401, 413, 415, 422, 500, 502, 503, 504)},
    description="Store one PDF or JSON document and its embeddings for the authenticated tenant.",
)
async def upload_document(
    document_file: Annotated[UploadFile, File(description="Source PDF or JSON document")],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(require_tenant)],
    service_factory=Depends(get_document_service_factory),
) -> DocumentCreated:
    source_type = validate_file_type(document_file)
    contents = await read_bounded(document_file, settings.max_document_bytes)
    ingestion = await ingest_document(contents, source_type, settings)
    metrics = request_metrics.get()
    if metrics:
        metrics.document_type = source_type
        metrics.document_size_bytes = len(contents)
        metrics.page_count = ingestion.document.page_count
        metrics.chunk_count = len(ingestion.chunks)
    async with service_factory() as service:
        saved = await service.save(ingestion, len(contents), tenant_id)
    return DocumentCreated(
        document_id=UUID(saved.document_id),
        source_type=saved.source_type,
        chunk_count=saved.chunk_count,
        tenant_id=saved.tenant_id,
    )


@questions_router.post(
    "/api/v1/documents/{document_id}/questions",
    response_model=QAResponse,
    tags=["documents"],
    responses={
        code: {"model": ErrorResponse} for code in (401, 404, 413, 415, 422, 500, 502, 503, 504)
    },
    description="Answer questions from a document saved by the authenticated tenant.",
)
async def ask_saved_document(
    document_id: str,
    request: Request,
    questions_file: Annotated[UploadFile, File(description="Questions JSON file")],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(require_tenant)],
    library=Depends(get_document_library),
    service_factory=Depends(get_document_service_factory),
) -> QAResponse:
    started_at = perf_counter()
    try:
        canonical_id = str(UUID(document_id))
    except ValueError:
        raise ServiceError("DOCUMENT_NOT_FOUND", "Document was not found.", 404) from None
    saved = library.get(canonical_id, tenant_id)
    if saved is None:
        raise ServiceError("DOCUMENT_NOT_FOUND", "Document was not found.", 404)
    validate_file_type(questions_file, questions=True)
    questions = parse_questions(
        await read_bounded(questions_file, settings.max_questions_file_bytes), settings
    )
    metrics = request_metrics.get()
    if metrics:
        metrics.document_type = saved.source_type
        metrics.document_size_bytes = saved.byte_count
        metrics.chunk_count = saved.chunk_count
        metrics.question_count = len(questions)
        metrics.unique_question_count = len(set(questions))
    async with service_factory() as service:
        return await service.answer(saved, questions, request.state.request_id, started_at)
