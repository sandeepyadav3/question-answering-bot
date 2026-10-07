import io
import json
import math

from pypdf import PdfReader

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.models.document_chunk import ParsedDocument, SourceRecord


def parse_document(contents: bytes, source_type: str, settings: Settings) -> ParsedDocument:
    if source_type == "pdf":
        return parse_pdf(contents, settings)
    if source_type == "json":
        return parse_json(contents, settings)
    raise ServiceError("UNSUPPORTED_FILE_TYPE", "Only PDF and JSON are supported.", 415)


def parse_pdf(contents: bytes, settings: Settings) -> ParsedDocument:
    if not contents.startswith(b"%PDF-"):
        raise ServiceError("INVALID_DOCUMENT", "The uploaded PDF could not be parsed.", 422)
    try:
        reader = PdfReader(io.BytesIO(contents), strict=True)
        if reader.is_encrypted:
            raise ServiceError("ENCRYPTED_PDF", "Encrypted PDFs are not supported.", 422)
        page_count = len(reader.pages)
        if page_count > settings.max_pdf_pages:
            raise ServiceError("TOO_MANY_PAGES", "PDF exceeds the configured page limit.", 422)
        records = []
        total = 0
        for page_number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            total += len(text)
            if total > settings.max_extracted_chars:
                raise ServiceError("DOCUMENT_TOO_LARGE", "Extracted text exceeds the limit.", 422)
            if text.strip():
                records.append(SourceRecord(text=text, source_type="pdf", page=page_number))
        if not records:
            raise ServiceError(
                "NO_EXTRACTABLE_TEXT",
                "No extractable text found. Scanned/image-only PDFs are not supported.",
                422,
            )
        return ParsedDocument(tuple(records), page_count=page_count)
    except ServiceError:
        raise
    except Exception as exc:
        # Third-party parsers raise several exception types for malformed objects.
        # Do not expose PDF contents or parser internals in client errors.
        raise ServiceError(
            "INVALID_DOCUMENT", "The uploaded PDF could not be parsed.", 422
        ) from exc


def parse_json(contents: bytes, settings: Settings) -> ParsedDocument:
    def reject_constant(value):
        raise ValueError("Nonstandard constant")

    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    try:
        data = json.loads(
            contents.decode("utf-8-sig"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_keys,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ServiceError("INVALID_DOCUMENT", "Document must be valid UTF-8 JSON.", 422) from exc
    if not isinstance(data, (dict, list)):
        raise ServiceError("INVALID_DOCUMENT", "JSON document must be an object or array.", 422)
    records = []
    total = 0
    # Iterative traversal avoids recursion errors; reverse push preserves source order.
    stack = [(data, "$", 0)]
    while stack:
        value, path, depth = stack.pop()
        if depth > settings.max_json_depth:
            raise ServiceError("JSON_TOO_DEEP", "JSON exceeds the nesting limit.", 422)
        if isinstance(value, dict):
            for key, child in reversed(value.items()):
                # Bracket notation handles dots, quotes and brackets in arbitrary keys.
                child_path = path + "[" + json.dumps(key, ensure_ascii=True) + "]"
                stack.append((child, child_path, depth + 1))
        elif isinstance(value, list):
            for index in range(len(value) - 1, -1, -1):
                stack.append((value[index], f"{path}[{index}]", depth + 1))
        else:
            if isinstance(value, str):
                # Reject lone surrogates that cannot be embedded as UTF-8.
                try:
                    value.encode("utf-8")
                except UnicodeError as exc:
                    raise ServiceError("INVALID_DOCUMENT", "Invalid Unicode in JSON.", 422) from exc
                text = value
            else:
                if isinstance(value, float) and not math.isfinite(value):
                    raise ServiceError("INVALID_DOCUMENT", "JSON numbers must be finite.", 422)
                text = json.dumps(value, ensure_ascii=False, allow_nan=False)
            if not text.strip():
                continue
            total += len(path) + len(text)
            if total > settings.max_extracted_chars:
                raise ServiceError("DOCUMENT_TOO_LARGE", "Extracted text exceeds the limit.", 422)
            if len(records) >= settings.max_document_records:
                raise ServiceError("TOO_MANY_RECORDS", "JSON exceeds the record limit.", 422)
            records.append(SourceRecord(text=text, source_type="json", json_path=path))
    if not records:
        raise ServiceError("EMPTY_DOCUMENT", "JSON contains no usable leaf values.", 422)
    return ParsedDocument(tuple(records))
