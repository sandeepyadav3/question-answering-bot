from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.upload_route import UploadRoute
from app.core.config import Settings
from app.core.logging import emit, request_metrics
from app.dependencies import get_qa_service_factory, get_settings
from app.schemas.responses import ErrorResponse, QAResponse
from app.services.ingestion import ingest_document
from app.services.question_parser import parse_questions
from app.services.uploads import read_bounded, validate_file_type

router = APIRouter(route_class=UploadRoute)


@router.post(
    "/api/v1/qa",
    response_model=QAResponse,
    tags=["qa"],
    responses={code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 502, 503, 504)},
    description="Answer questions from one uploaded PDF or JSON document with source citations.",
)
async def answer_questions(
    request: Request,
    questions_file: Annotated[UploadFile, File(description="Questions JSON file")],
    document_file: Annotated[UploadFile, File(description="Source PDF or JSON document")],
    settings: Annotated[Settings, Depends(get_settings)],
    service_factory=Depends(get_qa_service_factory),
) -> QAResponse:
    started_at = perf_counter()
    validate_file_type(questions_file, questions=True)
    source_type = validate_file_type(document_file)
    contents = await read_bounded(questions_file, settings.max_questions_file_bytes)
    questions = parse_questions(contents, settings)
    document = await read_bounded(document_file, settings.max_document_bytes)
    metrics = request_metrics.get()
    if metrics:
        metrics.document_type = source_type
        metrics.document_size_bytes = len(document)
        metrics.question_count = len(questions)
        metrics.unique_question_count = len(set(questions))
    ingestion_started = perf_counter()
    ingestion = await ingest_document(document, source_type, settings)
    if metrics:
        metrics.page_count = ingestion.document.page_count
        metrics.chunk_count = len(ingestion.chunks)
    emit(
        "stage_completed", stage="ingestion", latency_ms=(perf_counter() - ingestion_started) * 1000
    )
    async with service_factory() as service:
        return await service.answer(ingestion, questions, request.state.request_id, started_at)
