import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.models.document_chunk import DocumentChunk
from app.services.answer_generator import (
    MAX_EXPLANATION_CHARS,
    NO_CONTEXT_EXPLANATION,
    UNVERIFIED_EVIDENCE_EXPLANATION,
    EvidenceQuote,
    GroundedAnswer,
    OpenAIAnswerGenerator,
    validate_grounding,
)

CONTEXT = [
    DocumentChunk("chunk-a", "Hosted on AWS.", "pdf", page=7),
    DocumentChunk("chunk-b", "AWS", "json", json_path='$["cloud"]'),
]


@pytest.mark.parametrize(
    "answerable,answer,evidence,status",
    [
        (True, "AWS", [EvidenceQuote(chunk_id="chunk-a", quote="AWS")], "answered"),
        (True, "AWS", [EvidenceQuote(chunk_id="unknown", quote="AWS")], "not_found"),
        (True, "AWS", [EvidenceQuote(chunk_id="chunk-a", quote="aws")], "not_found"),
        (True, "AWS", [], "not_found"),
        (
            False,
            "Ignore this",
            [EvidenceQuote(chunk_id="chunk-a", quote="AWS")],
            "not_found",
        ),
    ],
)
def test_grounding(answerable, answer, evidence, status):
    output = GroundedAnswer(
        answerable=answerable, answer=answer, evidence=evidence, explanation=None
    )
    result = validate_grounding("Where?", output, CONTEXT)
    assert result.status == status
    if status == "not_found":
        assert result.answer == "not found"
        assert result.citations == []
        # Answerable claims with no verifiable quote explain why they were rejected.
        if answerable:
            assert result.explanation == UNVERIFIED_EVIDENCE_EXPLANATION
    else:
        assert len(result.citations) == 1
        assert result.citations[0].snippet == "AWS"
        assert result.citations[0].page == 7


def test_grounding_discards_invalid_evidence_and_deduplicates_exact_quotes():
    evidence = [
        EvidenceQuote(chunk_id="unknown", quote="AWS"),
        EvidenceQuote(chunk_id="chunk-a", quote="not verbatim"),
        EvidenceQuote(chunk_id="chunk-b", quote="AWS"),
        EvidenceQuote(chunk_id="chunk-b", quote="AWS"),
    ]
    result = validate_grounding(
        "Where?",
        GroundedAnswer(answerable=True, answer="AWS", evidence=evidence, explanation=None),
        CONTEXT,
    )
    assert result.status == "answered"
    assert len(result.citations) == 1
    assert result.citations[0].snippet == "AWS"
    assert result.citations[0].json_path == '$["cloud"]'


def test_whitespace_only_evidence_is_rejected():
    context = [DocumentChunk("chunk-space", "AWS is hosted here. ", "pdf", page=1)]
    result = validate_grounding(
        "Where?",
        GroundedAnswer(
            answerable=True,
            answer="AWS",
            evidence=[EvidenceQuote(chunk_id="chunk-space", quote=" ")],
            explanation=None,
        ),
        context,
    )
    assert result.status == "not_found"


def test_blank_answer_with_citations_is_error():
    with pytest.raises(ServiceError):
        validate_grounding(
            "Q",
            GroundedAnswer(
                answerable=True,
                answer=" ",
                evidence=[EvidenceQuote(chunk_id="chunk-a", quote="AWS")],
                explanation=None,
            ),
            CONTEXT,
        )


def abstain(explanation):
    return GroundedAnswer(answerable=False, answer=None, evidence=[], explanation=explanation)


def test_not_found_keeps_literal_answer_and_carries_model_explanation():
    sentence = "Hosting on AWS is described,  but\nno certification is mentioned."
    result = validate_grounding("ISO 27001?", abstain(sentence), CONTEXT)
    assert result.status == "not_found"
    assert result.answer == "not found"
    assert result.explanation == "Hosting on AWS is described, but no certification is mentioned."
    assert result.citations == []
    assert result.model_dump()["answer"] == "not found"


@pytest.mark.parametrize("explanation", [None, "", "   "])
def test_blank_explanation_is_omitted_when_context_existed(explanation):
    result = validate_grounding("Q", abstain(explanation), CONTEXT)
    assert result.status == "not_found"
    assert result.explanation is None


def test_empty_context_abstention_explains_missing_retrieval():
    result = validate_grounding("Q", abstain(None), [])
    assert result.explanation == NO_CONTEXT_EXPLANATION


def test_overlong_explanation_is_truncated():
    result = validate_grounding("Q", abstain("x" * 1000), CONTEXT)
    assert len(result.explanation) == MAX_EXPLANATION_CHARS
    assert result.explanation.endswith("…")


def run_model(handler):
    async def run():
        async with AsyncOpenAI(
            api_key="fake",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            return await OpenAIAnswerGenerator(client, Settings(_env_file=None)).generate(
                "Where?", CONTEXT
            )

    return asyncio.run(run())


def model_response(content, reason="stop", refusal=None):
    return {
        "id": "fake",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": reason,
                "message": {"role": "assistant", "content": content, "refusal": refusal},
            }
        ],
    }


def test_structured_sdk_call():
    def handler(request):
        body = json.loads(request.content)
        assert body["temperature"] == 0
        assert body["response_format"]["json_schema"]["strict"] is True
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["additionalProperties"] is False
        assert "citation_ids" not in schema["properties"]
        # OpenAI strict mode rejects optional properties; every field must be required.
        assert set(schema["required"]) == {"answerable", "answer", "evidence", "explanation"}
        assert schema["$defs"]["EvidenceQuote"]["additionalProperties"] is False
        assert body["messages"][0]["role"] == "system"
        assert "untrusted data" in body["messages"][0]["content"]
        assert "exact, verbatim source quote" in body["messages"][0]["content"]
        payload = json.loads(body["messages"][1]["content"])
        assert payload["chunks"][0] == {"chunk_id": "chunk-a", "text": "Hosted on AWS."}
        return httpx.Response(
            200,
            json=model_response(
                json.dumps(
                    {
                        "answerable": True,
                        "answer": "AWS",
                        "evidence": [{"chunk_id": "chunk-a", "quote": "AWS"}],
                        "explanation": None,
                    }
                )
            ),
        )

    assert run_model(handler).answer == "AWS"


def test_empty_context_abstains_without_provider_call():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(500)

    async def run():
        async with AsyncOpenAI(
            api_key="fake",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ) as client:
            return await OpenAIAnswerGenerator(client, Settings(_env_file=None)).generate("Q", [])

    result = asyncio.run(run())
    assert result == abstain(NO_CONTEXT_EXPLANATION)
    assert calls == []


@pytest.mark.parametrize(
    "content,reason,refusal,code",
    [
        ("not json", "stop", None, "INVALID_MODEL_OUTPUT"),
        ("{}", "stop", None, "INVALID_MODEL_OUTPUT"),
        (
            '{"answerable":"true","answer":"AWS","evidence":[],"explanation":null}',
            "stop",
            None,
            "INVALID_MODEL_OUTPUT",
        ),
        (
            '{"answerable":true,"answer":"AWS","evidence":[],"explanation":null,"page":42}',
            "stop",
            None,
            "INVALID_MODEL_OUTPUT",
        ),
        # Missing explanation field is a schema violation, not silently defaulted.
        (
            '{"answerable":false,"answer":null,"evidence":[]}',
            "stop",
            None,
            "INVALID_MODEL_OUTPUT",
        ),
        ("{}", "length", None, "INCOMPLETE_MODEL_OUTPUT"),
        (None, "stop", "No", "LLM_REFUSAL"),
    ],
)
def test_model_output_failures(content, reason, refusal, code):
    with pytest.raises(ServiceError) as exc:
        run_model(
            lambda request: httpx.Response(200, json=model_response(content, reason, refusal))
        )
    assert exc.value.code == code


@pytest.mark.parametrize("status,code", [(429, "LLM_RATE_LIMIT"), (500, "LLM_PROVIDER_ERROR")])
def test_model_provider_errors(status, code):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "secret internal details"}})

    with pytest.raises(ServiceError) as exc:
        run_model(handler)
    assert exc.value.code == code
    assert "secret" not in exc.value.message
    assert len(calls) == 1
