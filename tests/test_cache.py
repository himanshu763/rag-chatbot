from rag.cache import InMemoryCacheStore
from rag.interfaces import CacheStore


class _Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def test_store_conforms_to_protocol():
    assert isinstance(InMemoryCacheStore(), CacheStore)


def test_set_and_get():
    s = InMemoryCacheStore()
    s.set("k", {"v": 1})
    assert s.get("k") == {"v": 1}
    assert s.get("missing") is None


def test_ttl_expiry():
    clk = _Clock()
    s = InMemoryCacheStore(clock=clk)
    s.set("k", 1, ttl_seconds=10)
    clk.t = 9
    assert s.get("k") == 1
    clk.t = 11               # past ttl
    assert s.get("k") is None


def test_none_ttl_never_expires():
    clk = _Clock()
    s = InMemoryCacheStore(clock=clk)
    s.set("k", 1, ttl_seconds=None)
    clk.t = 1e9
    assert s.get("k") == 1


def test_max_entries_evicts_oldest():
    s = InMemoryCacheStore(max_entries=2)
    s.set("a", 1); s.set("b", 2); s.set("c", 3)   # "a" evicted
    assert s.get("a") is None
    assert s.get("b") == 2 and s.get("c") == 3


def test_items_omits_expired():
    clk = _Clock()
    s = InMemoryCacheStore(clock=clk)
    s.set("a", 1, ttl_seconds=10)
    s.set("b", 2, ttl_seconds=None)
    clk.t = 20
    keys = {k for k, _ in s.items()}
    assert keys == {"b"}


def test_clear():
    s = InMemoryCacheStore()
    s.set("a", 1); s.clear()
    assert s.get("a") is None


# ── SemanticCache ──
from rag.cache import SemanticCache
from rag.config import RagConfig
from rag.types import OrchestratorResult
from tests.fakes import FakeEmbedder


def _cache(**cfg_kw):
    cfg = RagConfig(**cfg_kw)
    return SemanticCache(InMemoryCacheStore(), FakeEmbedder(), cfg)


def _result(text):
    return OrchestratorResult(response=text)


def test_exact_hit_after_store():
    c = _cache()
    c.store_result("what is x?", "v1", [], _result("answer"))
    hit = c.lookup("what is x?", "v1", [])
    assert hit is not None and hit.response == "answer"


def test_miss_when_not_stored():
    assert _cache().lookup("anything", "v1", []) is None


def test_corpus_version_mismatch_misses():
    c = _cache()
    c.store_result("q", "v1", [], _result("a"))
    assert c.lookup("q", "v2", []) is None       # KB changed → miss


def test_history_fingerprint_mismatch_misses():
    c = _cache(cache_semantic_threshold=0.99)
    c.store_result("q", "v1", [{"role": "user", "content": "prior"}], _result("a"))
    assert c.lookup("q", "v1", []) is None        # different history → miss


def test_semantic_hit_on_different_query_when_threshold_low():
    # exact key differs (different query string) → falls to semantic scan; low threshold
    # makes the single stored entry (same version+history) qualify.
    c = _cache(cache_semantic_threshold=0.0)
    c.store_result("How much does it cost?", "v1", [], _result("cached"))
    hit = c.lookup("what is the price?", "v1", [])
    assert hit is not None and hit.response == "cached"


def test_semantic_miss_below_threshold():
    c = _cache(cache_semantic_threshold=0.999999)
    c.store_result("alpha beta gamma", "v1", [], _result("a"))
    assert c.lookup("completely different words here", "v1", []) is None


def test_embedder_failure_falls_back_to_exact(monkeypatch):
    c = _cache()
    c.store_result("q", "v1", [], _result("a"))
    def boom(texts): raise RuntimeError("embed down")
    monkeypatch.setattr(c.embedder, "embed", boom)
    assert c.lookup("q", "v1", []).response == "a"     # exact key works w/o embedding
    assert c.lookup("qq", "v1", []) is None            # semantic path swallows error
