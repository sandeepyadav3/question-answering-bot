import asyncio
import io
import json
from uuid import uuid4

import httpx
import pytest
from starlette.datastructures import UploadFile

from app.core.body_limit import BodyLimitMiddleware
from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.main import create_app
from app.services.question_parser import parse_questions
from app.services.uploads import read_bounded


def settings(**kw):
    return Settings(_env_file=None, **kw)


@pytest.mark.parametrize("value", [[" A? ", "B?", " A? "], {"questions": [" A? ", "B?", " A? "]}])
def test_question_formats_preserve_order_and_duplicates(value):
    assert parse_questions(json.dumps(value).encode(), settings()) == ["A?", "B?", "A?"]


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        (b"{", "INVALID_QUESTIONS_JSON"),
        (b"\xff", "INVALID_QUESTIONS_JSON"),
        (b"[NaN]", "INVALID_QUESTIONS_JSON"),
        (b'{"questions":[],"questions":["Q"]}', "INVALID_QUESTIONS_JSON"),
        (b"[]", "EMPTY_QUESTIONS"),
        (b"{}", "INVALID_QUESTIONS"),
        (b"null", "INVALID_QUESTIONS"),
        (b'{"questions":"Q"}', "INVALID_QUESTIONS"),
        (b'{"questions":["Q"],"extra":1}', "INVALID_QUESTIONS"),
        (b'["Q", " "]', "INVALID_QUESTION"),
        (b"[false]", "INVALID_QUESTION"),
        (b"[1]", "INVALID_QUESTION"),
    ],
)
def test_invalid_questions(raw, code):
    with pytest.raises(ServiceError) as exc:
        parse_questions(raw, settings())
    assert exc.value.code == code
    assert exc.value.status_code == 422


def test_question_limits_at_boundary():
    config = settings(max_questions=2, max_question_length=3)
    assert parse_questions(b'["abc","def"]', config) == ["abc", "def"]
    for raw, code in [(b'["a","b","c"]', "TOO_MANY_QUESTIONS"), (b'["abcd"]', "QUESTION_TOO_LONG")]:
        with pytest.raises(ServiceError) as exc:
            parse_questions(raw, config)
        assert exc.value.code == code


def test_bom_and_deep_json():
    assert parse_questions(b'\xef\xbb\xbf["Q"]', settings()) == ["Q"]
    with pytest.raises(ServiceError):
        parse_questions(b"[" * 2000 + b"]" * 2000, settings())


def test_bounded_reader_does_not_trust_upload_size():
    async def run():
        upload = UploadFile(io.BytesIO(b"12345"), size=1)
        with pytest.raises(ServiceError) as exc:
            await read_bounded(upload, 4)
        assert exc.value.status_code == 413
        await upload.close()
        upload = UploadFile(io.BytesIO(b"1234"))
        assert await read_bounded(upload, 4) == b"1234"
        await upload.close()

    asyncio.run(run())


def post(files, **config):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings(**config))),
            base_url="http://test",
        ) as client:
            return await client.post("/api/v1/qa", files=files)

    return asyncio.run(run())


def files(
    questions=b'["Q"]', document=b'{"x":1}', filename="document.json", mime="application/json"
):
    return [
        ("questions_file", ("questions.json", questions, "application/json")),
        ("document_file", (filename, document, mime)),
    ]


@pytest.mark.parametrize(
    ("uploads", "config", "status", "code"),
    [
        (files(), {}, 503, "PROVIDER_NOT_CONFIGURED"),
        (
            files(filename="DOC.PDF", document=b"%PDF-1.7\n", mime="application/pdf"),
            {},
            422,
            "INVALID_DOCUMENT",
        ),
        (files(mime="application/octet-stream"), {}, 503, "PROVIDER_NOT_CONFIGURED"),
        (files(filename="doc.txt"), {}, 415, "UNSUPPORTED_FILE_TYPE"),
        (files(mime="application/pdf"), {}, 415, "UNSUPPORTED_FILE_TYPE"),
        (files(questions=b"[]"), {}, 422, "EMPTY_QUESTIONS"),
        (files(document=b""), {}, 422, "EMPTY_FILE"),
        (files(questions=b""), {}, 422, "EMPTY_FILE"),
        (files(document=b"12345"), {"max_document_bytes": 4}, 413, "FILE_TOO_LARGE"),
        (files(document=b"1234"), {"max_document_bytes": 4}, 422, "INVALID_DOCUMENT"),
        (files(), {"max_questions_file_bytes": 4}, 413, "FILE_TOO_LARGE"),
        (files(), {"max_questions_file_bytes": 5}, 503, "PROVIDER_NOT_CONFIGURED"),
        ([files()[0], files()[0]], {}, 422, "INVALID_REQUEST"),
        (files() + [("extra", ("extra.json", b"{}"))], {}, 422, "INVALID_REQUEST"),
        (files() + [("extra", (None, "value"))], {}, 422, "INVALID_REQUEST"),
    ],
)
def test_upload_api(uploads, config, status, code):
    response = post(uploads, **config)
    assert response.status_code == status, response.text
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "headers", [[], [(b"content-length", b"1")], [(b"content-length", b"100")]]
)
def test_body_limit_with_missing_false_or_large_content_length(headers):
    async def run():
        called = False

        async def downstream(scope, receive, send):
            nonlocal called
            called = True

        incoming = iter(
            [
                {"type": "http.request", "body": b"1234", "more_body": True},
                {"type": "http.request", "body": b"56789", "more_body": False},
            ]
        )
        sent = []

        async def receive():
            return next(incoming)

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/qa",
            "headers": headers,
            "state": {"request_id": uuid4()},
        }
        await BodyLimitMiddleware(downstream, max_bytes=8)(scope, receive, send)
        assert not called
        assert sent[0]["status"] == 413
        assert json.loads(sent[1]["body"])["error"]["code"] == "REQUEST_TOO_LARGE"

    asyncio.run(run())


def test_body_at_limit_replays_unchanged():
    async def run():
        sent = []

        async def downstream(scope, receive, send):
            sent.append(await receive())

        async def receive():
            return {"type": "http.request", "body": b"1234", "more_body": False}

        async def send(message):
            pass

        scope = {"type": "http", "method": "POST", "path": "/api/v1/qa", "headers": []}
        await BodyLimitMiddleware(downstream, 4)(scope, receive, send)
        assert sent == [{"type": "http.request", "body": b"1234", "more_body": False}]

    asyncio.run(run())


def test_whole_request_limit_returns_correlated_error():
    response = post(
        files(), max_document_bytes=2, max_questions_file_bytes=5, multipart_overhead_bytes=1
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "REQUEST_TOO_LARGE"
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


@pytest.mark.parametrize(
    "content_type, body",
    [
        ("multipart/form-data", b"no boundary"),
        ("multipart/form-data; boundary=broken", b"not multipart"),
        ("application/json", b"{}"),
    ],
)
def test_malformed_request(content_type, body):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(settings())), base_url="http://test"
        ) as client:
            return await client.post(
                "/api/v1/qa", content=body, headers={"content-type": content_type}
            )

    response = asyncio.run(run())
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_uploads_closed_after_validation_error(monkeypatch):
    closed = []
    original = UploadFile.close

    async def close(upload):
        await original(upload)
        closed.append(upload.file.closed)

    monkeypatch.setattr(UploadFile, "close", close)
    assert post(files(questions=b"[]")).status_code == 422
    assert len(closed) >= 2
    assert all(closed)


def test_questions_reject_unpaired_unicode_surrogates():
    with pytest.raises(ServiceError) as exc:
        parse_questions(b'["\\ud800"]', settings())
    assert exc.value.code == "INVALID_QUESTION"
