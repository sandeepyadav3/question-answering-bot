import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.exceptions import ServiceError
from app.models.document_chunk import DocumentChunk
from app.services.retrieval import RetrievalService
from app.services.vector_store import ChromaVectorStore


def chunks(prefix=""):
    return [
        DocumentChunk("a", prefix + "AWS", "json", json_path='$["cloud"]'),
        DocumentChunk("b", prefix + "Azure", "pdf", page=2),
    ]


def test_real_chroma_search_metadata_and_cleanup():
    store = ChromaVectorStore()
    before = store.client.count_collections()
    source = chunks()
    result = store.search(source, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], 1)
    assert result == [[source[0]], [source[1]]]
    assert result[0][0] is source[0]
    assert store.client.count_collections() == before
    assert len(store.search(source, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]], 50)[0]) == 2


def test_wider_candidates_are_filtered_at_similarity_boundary():
    class Collection:
        def add(self, **kwargs):
            pass

        def query(self, **kwargs):
            assert kwargs["n_results"] == 3
            return {
                "ids": [["a", "b", "c"]],
                "distances": [[0.1, 0.5, 0.5000001]],
            }

    class Client:
        def create_collection(self, **kwargs):
            return Collection()

        def get_max_batch_size(self):
            return 100

        def delete_collection(self, name):
            pass

    source = [
        DocumentChunk("a", "A", "pdf", page=1),
        DocumentChunk("b", "B", "pdf", page=2),
        DocumentChunk("c", "C", "pdf", page=3),
    ]
    result = ChromaVectorStore(Client()).search(
        source,
        [[1.0, 0.0], [0.8, 0.6], [0.6, 0.8]],
        [[1.0, 0.0]],
        top_k=3,
        fetch_k=3,
        min_similarity=0.5,
        mmr_lambda=1,
    )
    assert [[chunk.chunk_id for chunk in row] for row in result] == [["a", "b"]]


def test_mmr_prefers_diversity_and_has_deterministic_order():
    source = [
        DocumentChunk("a", "A", "pdf", page=1),
        DocumentChunk("b", "B", "pdf", page=2),
        DocumentChunk("c", "C", "pdf", page=3),
    ]
    vectors = [[1.0, 0.0], [0.99, 0.1], [0.6, 0.8]]
    store = ChromaVectorStore()
    first = store.search(
        source, vectors, [[1.0, 0.0]], top_k=2, fetch_k=3, min_similarity=-1, mmr_lambda=0.3
    )
    second = store.search(
        source, vectors, [[1.0, 0.0]], top_k=2, fetch_k=3, min_similarity=-1, mmr_lambda=0.3
    )
    assert [chunk.chunk_id for chunk in first[0]] == ["a", "c"]
    assert [chunk.chunk_id for chunk in second[0]] == ["a", "c"]


def test_all_weak_candidates_produce_empty_context():
    source = chunks()
    result = ChromaVectorStore().search(
        source,
        [[1.0, 0.0], [0.0, 1.0]],
        [[-1.0, 0.0]],
        top_k=2,
        fetch_k=2,
        min_similarity=0.1,
        mmr_lambda=0.7,
    )
    assert result == [[]]


def test_concurrent_collections_with_same_chunk_ids_are_isolated():
    store = ChromaVectorStore()
    before = store.client.count_collections()

    def run(prefix):
        source = chunks(prefix)
        return store.search(source, [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]], 1)[0][0].text

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, ["one ", "two ", "three ", "four "]))
    assert results == ["one AWS", "two AWS", "three AWS", "four AWS"]
    assert store.client.count_collections() == before


def test_unique_question_embeddings_and_original_positions():
    class FakeEmbeddings:
        def __init__(self):
            self.calls = []

        async def embed(self, texts):
            self.calls.append(list(texts))
            return [[1.0, 0.0] if "AWS" in text else [0.0, 1.0] for text in texts]

    provider = FakeEmbeddings()
    result = asyncio.run(
        RetrievalService(provider, ChromaVectorStore(), top_k=1).retrieve(
            chunks(), ["AWS?", "Azure?", "AWS?"]
        )
    )
    assert provider.calls == [['$["cloud"]: AWS', "Azure"], ["AWS?", "Azure?"]]
    assert result.unique_questions == ("AWS?", "Azure?")
    assert result.question_indices == (0, 1, 0)
    assert [row[0].chunk_id for row in result.contexts] == ["a", "b"]


def test_retrieval_service_passes_quality_controls():
    class Provider:
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class Store:
        def search(self, *args):
            self.args = args
            return [[]]

    store = Store()
    result = asyncio.run(
        RetrievalService(
            Provider(),
            store,
            top_k=2,
            fetch_k=7,
            min_similarity=0.4,
            mmr_lambda=0.6,
        ).retrieve(chunks(), ["Q"])
    )
    assert store.args[3:] == (2, 7, 0.4, 0.6)
    assert result.contexts == ((),)


@pytest.mark.parametrize(
    "overrides",
    [
        {"retrieval_top_k": 5, "retrieval_fetch_k": 4},
        {"retrieval_min_similarity": -1.1},
        {"retrieval_min_similarity": 1.1},
        {"retrieval_mmr_lambda": -0.1},
        {"retrieval_mmr_lambda": 1.1},
    ],
)
def test_invalid_retrieval_configuration_rejected(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_retrieval_configuration_defaults():
    settings = Settings(_env_file=None)
    assert settings.retrieval_fetch_k == 20
    assert settings.retrieval_min_similarity == 0.15
    assert settings.retrieval_mmr_lambda == 0.7


@pytest.mark.parametrize("failure", ["add", "query"])
def test_cleanup_on_index_or_search_failure(failure):
    class Collection:
        def add(self, **kwargs):
            if failure == "add":
                raise RuntimeError("private document details")

        def query(self, **kwargs):
            raise RuntimeError("private query details")

    class Client:
        deleted = []

        def create_collection(self, **kwargs):
            self.created = kwargs["name"]
            assert kwargs["embedding_function"] is None
            return Collection()

        def get_max_batch_size(self):
            return 1

        def delete_collection(self, name):
            self.deleted.append(name)

    client = Client()
    with pytest.raises(ServiceError) as exc:
        ChromaVectorStore(client).search(chunks(), [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]], 1)
    assert exc.value.code == "VECTOR_STORE_ERROR"
    assert "private" not in exc.value.message
    assert client.deleted == [client.created]


def test_bad_dimensions_do_not_create_collection():
    class Client:
        def create_collection(self, **kwargs):
            raise AssertionError("Must validate before creating a collection")

    with pytest.raises(ServiceError):
        ChromaVectorStore(Client()).search(chunks(), [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0, 1.0]], 1)


def test_cleanup_failure_is_explicit():
    class Collection:
        def add(self, **kwargs):
            pass

        def query(self, **kwargs):
            return {"ids": [["a"]], "distances": [[0.0]]}

    class Client:
        def create_collection(self, **kwargs):
            return Collection()

        def get_max_batch_size(self):
            return 100

        def delete_collection(self, name):
            raise RuntimeError("private cleanup details")

    with pytest.raises(ServiceError) as exc:
        ChromaVectorStore(Client()).search(
            chunks(), [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0]], 1
        )
    assert exc.value.code == "VECTOR_STORE_CLEANUP_ERROR"
    assert "private" not in exc.value.message


def test_embedding_failure_prevents_index_work():
    class Provider:
        async def embed(self, texts):
            raise ServiceError("EMBEDDING_TIMEOUT", "Timed out", 504)

    class Store:
        def search(self, *args):
            raise AssertionError("Should not index after provider failure")

    with pytest.raises(ServiceError) as exc:
        asyncio.run(RetrievalService(Provider(), Store()).retrieve(chunks(), ["Q"]))
    assert exc.value.status_code == 504


def test_collection_cleanup_survives_caller_cancellation():
    import threading

    started = threading.Event()
    release = threading.Event()
    cleaned = threading.Event()

    class Collection:
        def add(self, **kwargs):
            pass

        def query(self, **kwargs):
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("Test synchronization timed out")
            return {"ids": [["a"]], "distances": [[0.0]]}

    class Client:
        def create_collection(self, **kwargs):
            return Collection()

        def get_max_batch_size(self):
            return 100

        def delete_collection(self, name):
            cleaned.set()

    class Provider:
        async def embed(self, texts):
            return [[1.0, 0.0] for _ in texts]

    async def run():
        service = RetrievalService(Provider(), ChromaVectorStore(Client()))
        task = asyncio.create_task(service.retrieve(chunks(), ["Q"]))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
        assert await asyncio.to_thread(cleaned.wait, 3)

    asyncio.run(run())


def test_missing_key_fails_before_opening_provider(monkeypatch):
    from app.dependencies import open_retrieval_service

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    async def run():
        async with open_retrieval_service(Settings(_env_file=None)):
            raise AssertionError("Should fail without key")

    with pytest.raises(ServiceError) as exc:
        asyncio.run(run())
    assert exc.value.code == "PROVIDER_NOT_CONFIGURED"
