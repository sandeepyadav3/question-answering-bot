import asyncio
from pathlib import Path

import httpx
import pytest

from app.core.config import Settings
from app.main import create_app


@pytest.mark.parametrize(
    "path,content_type",
    [
        ("/", "text/html"),
        ("/static/app.js", "javascript"),
        ("/static/styles.css", "text/css"),
        ("/static/examples/questions.json", "application/json"),
        ("/static/examples/document.json", "application/json"),
        ("/static/examples/document.pdf", "application/pdf"),
    ],
)
def test_frontend_assets_and_content_policy(path, content_type):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(Settings(_env_file=None))),
            base_url="http://test",
        ) as client:
            return await client.get(path)

    response = asyncio.run(run())
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "script-src 'self'" in response.headers["content-security-policy"]
    if path == "/":
        assert 'name="questions_file"' in response.text
        assert 'name="document_file"' in response.text
        assert 'color-scheme" content="dark"' in response.text


def test_samples_pass_real_document_and_question_parsers():
    from app.services.chunking import chunk_document
    from app.services.document_parser import parse_document
    from app.services.question_parser import parse_questions

    samples = Path(__file__).parents[1] / "app/static/examples"
    settings = Settings(_env_file=None)
    assert len(parse_questions((samples / "questions.json").read_bytes(), settings)) == 3
    for kind in ("pdf", "json"):
        document = parse_document((samples / f"document.{kind}").read_bytes(), kind, settings)
        assert chunk_document(document, settings)
