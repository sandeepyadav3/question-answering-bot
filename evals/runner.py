"""Run the reproducible grounding corpus against a document-QA server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

DEFAULT_CORPUS = Path(__file__).with_name("corpus.json")
QA_PATH = "/api/v1/qa"


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    expected_answerable: bool
    answerability_ok: bool
    answer_match_ok: bool
    citation_location_ok: bool
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def load_corpus(path: Path) -> dict[str, Any]:
    corpus = json.loads(path.read_text(encoding="utf-8"))
    if corpus.get("version") != 1 or not corpus.get("documents"):
        raise ValueError("Corpus must have version 1 and at least one document")
    return corpus


def _location(citation: dict[str, Any]) -> str | None:
    source_type = citation.get("source_type")
    if source_type == "pdf" and isinstance(citation.get("page"), int):
        return f"pdf:page:{citation['page']}"
    if source_type == "json" and isinstance(citation.get("json_path"), str):
        return f"json:path:{citation['json_path']}"
    return None


def score_case(case: dict[str, Any], result: dict[str, Any]) -> CaseResult:
    expected_status = "answered" if case["answerable"] else "not_found"
    status = result.get("status")
    failures: list[str] = []
    answerability_ok = status == expected_status
    if not answerability_ok:
        failures.append(f"expected status {expected_status!r}, got {status!r}")

    answer_match_ok = True
    citation_location_ok = True
    if case["answerable"]:
        answer = result.get("answer")
        expected_parts = case.get("expected_answer_contains", [])
        answer_match_ok = isinstance(answer, str) and all(
            part.casefold() in answer.casefold() for part in expected_parts
        )
        if not answer_match_ok:
            failures.append(f"answer did not contain all expected text: {expected_parts!r}")

        expected_locations = set(case.get("expected_citation_locations", []))
        actual_locations = {
            location
            for citation in result.get("citations", [])
            if isinstance(citation, dict) and (location := _location(citation))
        }
        citation_location_ok = bool(actual_locations) and actual_locations == expected_locations
        if not citation_location_ok:
            failures.append(
                f"expected citation locations {sorted(expected_locations)!r}, "
                f"got {sorted(actual_locations)!r}"
            )
    else:
        answer_match_ok = result.get("answer") == "not found"
        if not answer_match_ok:
            failures.append("abstention answer was not exactly 'not found'")
        citation_location_ok = result.get("citations") == []
        if not citation_location_ok:
            failures.append("abstention included citations")

    return CaseResult(
        case_id=case["id"],
        expected_answerable=case["answerable"],
        answerability_ok=answerability_ok,
        answer_match_ok=answer_match_ok,
        citation_location_ok=citation_location_ok,
        failures=tuple(failures),
    )


def _evaluate_document(
    client: httpx.Client,
    base_url: str,
    corpus_dir: Path,
    document: dict[str, Any],
) -> list[CaseResult]:
    cases = document["cases"]
    questions = json.dumps({"questions": [case["question"] for case in cases]}).encode()
    document_path = corpus_dir / document["file"]
    with document_path.open("rb") as document_file:
        response = client.post(
            f"{base_url.rstrip('/')}{QA_PATH}",
            files={
                "questions_file": ("questions.json", questions, "application/json"),
                "document_file": (
                    document_path.name,
                    document_file,
                    document["content_type"],
                ),
            },
        )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(cases):
        raise ValueError(
            f"{document['id']}: expected {len(cases)} results, "
            f"got {len(results) if isinstance(results, list) else 'an invalid response'}"
        )
    return [score_case(case, result) for case, result in zip(cases, results, strict=True)]


def evaluate(
    corpus_path: Path,
    base_url: str,
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = 120.0,
) -> list[CaseResult]:
    corpus = load_corpus(corpus_path)
    with httpx.Client(transport=transport, timeout=timeout) as client:
        return [
            result
            for document in corpus["documents"]
            for result in _evaluate_document(client, base_url, corpus_path.parent, document)
        ]


def format_report(results: list[CaseResult]) -> str:
    total = len(results)
    supported = [result for result in results if result.expected_answerable]
    unsupported = [result for result in results if not result.expected_answerable]

    def metric(label: str, passed: int, denominator: int) -> str:
        percentage = 100.0 * passed / denominator if denominator else 100.0
        return f"{label}: {passed}/{denominator} ({percentage:.1f}%)"

    lines = [
        metric("Answerability accuracy", sum(r.answerability_ok for r in results), total),
        metric(
            "Abstention accuracy",
            sum(r.answerability_ok and r.answer_match_ok for r in unsupported),
            len(unsupported),
        ),
        metric(
            "Expected answer matches",
            sum(r.answer_match_ok for r in supported),
            len(supported),
        ),
        metric(
            "Citation-location correctness",
            sum(r.citation_location_ok for r in supported),
            len(supported),
        ),
        metric("Cases passed", sum(r.passed for r in results), total),
    ]
    for result in results:
        if not result.passed:
            lines.append(f"FAIL {result.case_id}: {'; '.join(result.failures)}")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:8000"),
        help="Document-QA server base URL (default: EVAL_BASE_URL or localhost)",
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        results = evaluate(args.corpus.resolve(), args.base_url, timeout=args.timeout)
    except (OSError, ValueError, json.JSONDecodeError, httpx.HTTPError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2
    print(format_report(results))
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
