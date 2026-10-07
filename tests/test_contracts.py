from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from app.core.config import Settings
from app.schemas.answers import QuestionResult
from app.schemas.responses import QAResponse

result_adapter = TypeAdapter(QuestionResult)
PDF_CITATION = {"source_type": "pdf", "page": 7, "snippet": "Hosted on AWS."}


@pytest.mark.parametrize(
    "result",
    [
        {"question": "Where?", "status": "answered", "answer": "AWS", "citations": [PDF_CITATION]},
        {
            "question": "Where?",
            "status": "answered",
            "answer": "AWS",
            "citations": [{"source_type": "json", "json_path": "$.cloud", "snippet": "AWS"}],
        },
        {"question": "Where?", "status": "not_found"},
        {
            "question": "Where?",
            "status": "not_found",
            "explanation": "Hosting is described, but no certification is mentioned.",
        },
        {
            "question": "Where?",
            "status": "error",
            "error": {"code": "LLM_TIMEOUT", "message": "Unable to process this question."},
        },
    ],
)
def test_valid_result_variants(result):
    parsed = result_adapter.validate_python(result)
    assert parsed.status == result["status"]


@pytest.mark.parametrize(
    "result",
    [
        {"status": "answered", "answer": "AWS", "citations": []},
        {"status": "answered", "answer": " ", "citations": [PDF_CITATION]},
        {"status": "answered", "answer": "AWS", "citations": [{**PDF_CITATION, "page": 0}]},
        {
            "status": "answered",
            "answer": "AWS",
            "citations": [{**PDF_CITATION, "json_path": "$.cloud"}],
        },
        {"status": "not_found", "answer": "AWS"},
        {"status": "not_found", "citations": [PDF_CITATION]},
        {"status": "not_found", "explanation": "   "},
        {"status": "answered", "answer": "AWS", "citations": [PDF_CITATION], "explanation": "x"},
        {"status": "error", "error": {"code": "FAIL", "message": "Failed"}, "explanation": "x"},
        {"status": "error", "answer": "AWS", "error": {"code": "FAIL", "message": "Failed"}},
        {"status": "error"},
    ],
)
def test_inconsistent_results_rejected(result):
    with pytest.raises(ValidationError):
        result_adapter.validate_python({"question": "Where?", **result})


def test_response_counts_include_errors_and_duplicates():
    response = QAResponse.model_validate(
        {
            "request_id": str(uuid4()),
            "results": [{"question": "Where?", "status": "not_found"}] * 2,
            "metadata": {
                "questions_received": 2,
                "questions_answered": 0,
                "questions_not_found": 2,
                "questions_failed": 0,
                "chunk_count": 1,
                "latency_ms": 12.5,
            },
        }
    )
    assert response.results[0].answer == "not found"
    assert response.results[0].explanation is None
    bad = response.model_dump()
    bad["metadata"]["questions_received"] = 1
    with pytest.raises(ValidationError, match="counts must match"):
        QAResponse.model_validate(bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_questions": 0},
        {"llm_timeout_seconds": -1},
        {"chunk_size_tokens": 100, "chunk_overlap_tokens": 100},
    ],
)
def test_invalid_resource_configuration_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("MAX_QUESTIONS", "12")
    assert Settings(_env_file=None).max_questions == 12
