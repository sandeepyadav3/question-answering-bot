import pytest


@pytest.fixture(autouse=True)
def no_real_provider_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
