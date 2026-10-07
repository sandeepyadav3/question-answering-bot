import json

from app.core.config import Settings
from app.core.exceptions import ServiceError


def parse_questions(contents: bytes, settings: Settings) -> list[str]:
    def reject_constant(value: str):
        raise ValueError("Nonstandard JSON constant")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        data = json.loads(
            contents.decode("utf-8-sig"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_keys,
        )
    except (UnicodeError, ValueError, RecursionError):
        raise ServiceError("INVALID_QUESTIONS_JSON", "Questions must be valid UTF-8 JSON.", 422)
    if isinstance(data, dict) and set(data) == {"questions"}:
        data = data["questions"]
    if not isinstance(data, list):
        raise ServiceError(
            "INVALID_QUESTIONS", "Use a JSON array or an object containing only questions.", 422
        )
    if not data:
        raise ServiceError("EMPTY_QUESTIONS", "At least one question is required.", 422)
    if len(data) > settings.max_questions:
        raise ServiceError(
            "TOO_MANY_QUESTIONS",
            f"A maximum of {settings.max_questions} questions is allowed.",
            422,
        )
    questions = []
    for index, question in enumerate(data):
        if not isinstance(question, str) or not question.strip():
            raise ServiceError(
                "INVALID_QUESTION", f"Question {index + 1} must be a nonblank string.", 422
            )
        if len(question) > settings.max_question_length:
            raise ServiceError(
                "QUESTION_TOO_LONG",
                f"Question {index + 1} exceeds {settings.max_question_length} characters.",
                422,
            )
        try:
            question.encode("utf-8")
        except UnicodeError as exc:
            raise ServiceError(
                "INVALID_QUESTION", f"Question {index + 1} contains invalid Unicode.", 422
            ) from exc
        questions.append(question.strip())
    return questions
