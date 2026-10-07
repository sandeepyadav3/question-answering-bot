import json
import logging
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass
class RequestMetrics:
    request_id: str
    embedding_calls: int = 0
    generation_calls: int = 0
    embedding_input_tokens: int = 0
    generation_input_tokens: int = 0
    generation_output_tokens: int = 0
    usage_missing_calls: int = 0
    document_type: str | None = None
    document_size_bytes: int = 0
    page_count: int = 0
    chunk_count: int = 0
    question_count: int = 0
    unique_question_count: int = 0
    answered_count: int = 0
    not_found_count: int = 0
    failed_count: int = 0


request_metrics: ContextVar[RequestMetrics | None] = ContextVar("request_metrics", default=None)
question_index: ContextVar[int | None] = ContextVar("question_index", default=None)
ALLOWED_FIELDS = set(RequestMetrics.__dataclass_fields__) | {
    "route",
    "http_status",
    "latency_ms",
    "stage",
    "status",
    "question_index",
    "error_code",
}
logger = logging.getLogger("document_qa")


class JSONFormatter(logging.Formatter):
    def format(self, record):
        # Never serialize arbitrary messages, exception strings, prompts or tracebacks.
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event_name", "application_log"),
        }
        payload.update(
            {
                key: value
                for key, value in getattr(record, "safe_fields", {}).items()
                if key in ALLOWED_FIELDS
            }
        )
        return json.dumps(payload, allow_nan=False)


def configure_logging():
    if not any(getattr(handler, "_document_qa", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler._document_qa = True
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # SDK/parser diagnostics can include raw input or response bodies. Emit our safe events instead.
    for name in ("openai", "httpx", "httpcore", "pypdf", "python_multipart", "chromadb"):
        dependency_logger = logging.getLogger(name)
        dependency_logger.handlers = [logging.NullHandler()]
        dependency_logger.propagate = False


def emit(event: str, **fields):
    metrics = request_metrics.get()
    if metrics is not None:
        fields.setdefault("request_id", metrics.request_id)
    index = question_index.get()
    if index is not None:
        fields.setdefault("question_index", index)
    logger.info("", extra={"event_name": event, "safe_fields": fields})


def begin_provider_call(kind: str):
    metrics = request_metrics.get()
    if metrics is not None:
        field = "embedding_calls" if kind == "embedding" else "generation_calls"
        setattr(metrics, field, getattr(metrics, field) + 1)
        # Mark unknown until a usable provider usage object arrives.
        metrics.usage_missing_calls += 1


def record_usage(kind: str, usage):
    metrics = request_metrics.get()
    if metrics is None or usage is None:
        return
    prompt = getattr(usage, "prompt_tokens", None)
    completion = getattr(usage, "completion_tokens", None) if kind == "generation" else 0
    if not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in (prompt, completion)
    ):
        return
    metrics.usage_missing_calls -= 1
    if kind == "embedding":
        metrics.embedding_input_tokens += prompt
    else:
        metrics.generation_input_tokens += prompt
        metrics.generation_output_tokens += completion


def summary():
    metrics = request_metrics.get()
    return asdict(metrics) if metrics else {}
