from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def client():
    with TestClient(create_app(Settings(_env_file=None))) as test_client:
        yield test_client


def test_health_without_provider_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    UUID(response.headers["X-Request-ID"])


def test_qa_requires_provider_after_validation(client):
    response = client.post(
        "/api/v1/qa",
        files={
            "questions_file": ("questions.json", b'["Where?"]', "application/json"),
            "document_file": ("document.json", b'{"region":"US"}', "application/json"),
        },
    )
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "PROVIDER_NOT_CONFIGURED"
    assert error["request_id"] == response.headers["X-Request-ID"]


@pytest.mark.parametrize("field", [None, "questions_file", "document_file"])
def test_both_upload_fields_required(client, field):
    files = {field: ("input.json", b"{}", "application/json")} if field else None
    response = client.post("/api/v1/qa", files=files)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_openapi_exposes_multipart_and_error_contract(client):
    schema = client.get("/openapi.json").json()
    endpoint = schema["paths"]["/api/v1/qa"]["post"]
    body = endpoint["requestBody"]["content"]["multipart/form-data"]["schema"]
    body_schema = schema["components"]["schemas"][body["$ref"].split("/")[-1]]
    assert set(body_schema["required"]) == {"questions_file", "document_file"}
    error = endpoint["responses"]["422"]["content"]["application/json"]["schema"]
    assert error["$ref"].endswith("/ErrorResponse")


def test_missing_route_uses_error_envelope(client):
    response = client.get("/missing")
    assert response.status_code == 404
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]
