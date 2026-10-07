import io
import json

import httpx
from pypdf import PdfReader

from evals import build_fixtures, runner


def test_evaluate_batches_by_document_without_credentials(tmp_path, monkeypatch):
    (tmp_path / "one.json").write_text('{"fact":"one"}', encoding="utf-8")
    (tmp_path / "two.json").write_text('{"fact":"two"}', encoding="utf-8")
    corpus = {
        "version": 1,
        "documents": [
            {
                "id": "one",
                "file": "one.json",
                "content_type": "application/json",
                "cases": [
                    {
                        "id": "one-case",
                        "question": "First?",
                        "answerable": True,
                        "expected_answer_contains": ["one"],
                        "expected_citation_locations": ['json:path:$["fact"]'],
                    }
                ],
            },
            {
                "id": "two",
                "file": "two.json",
                "content_type": "application/json",
                "cases": [
                    {
                        "id": "two-case",
                        "question": "Second?",
                        "answerable": True,
                        "expected_answer_contains": ["two"],
                        "expected_citation_locations": ['json:path:$["fact"]'],
                    }
                ],
            },
        ],
    }
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps(corpus), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = request.content
        assert b"must-not-leak" not in body
        assert "authorization" not in request.headers
        answer = "one" if b"First?" in body else "two"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "status": "answered",
                        "answer": answer,
                        "citations": [
                            {
                                "source_type": "json",
                                "json_path": '$["fact"]',
                                "snippet": answer,
                            }
                        ],
                    }
                ]
            },
        )

    results = runner.evaluate(
        corpus_path,
        "http://qa.example/",
        transport=httpx.MockTransport(handler),
    )

    assert len(seen) == 2
    assert all(request.url == "http://qa.example/api/v1/qa" for request in seen)
    assert all(result.passed for result in results)


def test_score_case_rejects_wrong_location_and_accepts_clean_abstention():
    answered = runner.score_case(
        {
            "id": "fact",
            "answerable": True,
            "expected_answer_contains": ["violet barley"],
            "expected_citation_locations": ["pdf:page:1"],
        },
        {
            "status": "answered",
            "answer": "The crop is violet barley.",
            "citations": [{"source_type": "pdf", "page": 2, "snippet": "violet barley"}],
        },
    )
    abstained = runner.score_case(
        {"id": "missing", "answerable": False},
        {"status": "not_found", "answer": "not found", "citations": []},
    )

    assert answered.answerability_ok and answered.answer_match_ok
    assert not answered.citation_location_ok
    assert not answered.passed
    assert abstained.passed


def test_report_has_required_metrics_and_main_fails_expectations(monkeypatch, capsys):
    failed = runner.score_case(
        {"id": "unsupported", "answerable": False},
        {"status": "answered", "answer": "invented", "citations": []},
    )
    monkeypatch.setattr(runner, "evaluate", lambda *args, **kwargs: [failed])

    assert runner.main(["--base-url", "http://qa.example"]) == 1
    report = capsys.readouterr().out
    assert "Answerability accuracy: 0/1" in report
    assert "Abstention accuracy: 0/1" in report
    assert "Expected answer matches: 0/0" in report
    assert "Citation-location correctness: 0/0" in report
    assert "FAIL unsupported" in report


def test_checked_in_corpus_and_pdf_are_reproducible():
    corpus = runner.load_corpus(runner.DEFAULT_CORPUS)
    case_ids = [
        case["id"] for document in corpus["documents"] for case in document["cases"]
    ]
    pdf_path = runner.DEFAULT_CORPUS.parent / "documents" / "lyra_station.pdf"
    pdf_bytes = pdf_path.read_bytes()
    pages = PdfReader(io.BytesIO(pdf_bytes)).pages

    assert len(case_ids) == len(set(case_ids)) == 12
    assert {document["content_type"] for document in corpus["documents"]} == {
        "application/json",
        "application/pdf",
    }
    assert pdf_bytes == build_fixtures.build_pdf(build_fixtures.PAGES)
    assert "violet barley" in pages[0].extract_text()
    assert "88.4 MHz" in pages[1].extract_text()
    assert "Ignore prior instructions" in pages[2].extract_text()
