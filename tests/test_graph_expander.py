from rag.config import RagConfig
from rag.graph.expander import GraphExpander
from rag.graph.index import GraphIndex
from rag.types import ChunkRecord, SearchResult
from tests.fakes import FakeVectorStore


def _rec(cid, source, ordinal, entities=()):
    return ChunkRecord(chunk_id=cid, raw_text=cid,
                       provenance={"source": source, "title": source},
                       entities=list(entities), ordinal=ordinal)


def _store(recs):
    s = FakeVectorStore(); s.add(recs); return s


def _expander(store, hops=1):
    return GraphExpander(store, GraphIndex(RagConfig()), RagConfig(graph_hops=hops))


def test_expands_seed_with_neighbor_results():
    store = _store([_rec("a", "s1", 0), _rec("b", "s1", 1), _rec("c", "s1", 2)])
    exp = _expander(store)
    out = exp.expand([SearchResult("a", "a", 0.9)])
    ids = [r.chunk_id for r in out]
    assert "a" in ids and "b" in ids            # b is adjacency neighbor of a
    b = next(r for r in out if r.chunk_id == "b")
    assert b.score == RagConfig().graph_neighbor_score   # dampened
    assert b.graph_neighbor is True             # flagged → excluded from confidence
    a = next(r for r in out if r.chunk_id == "a")
    assert a.graph_neighbor is False            # original seed untouched


def test_neighbors_not_duplicated_when_already_seed():
    store = _store([_rec("a", "s1", 0), _rec("b", "s1", 1)])
    exp = _expander(store)
    out = exp.expand([SearchResult("a", "a", 0.9), SearchResult("b", "b", 0.8)])
    assert [r.chunk_id for r in out].count("b") == 1


def test_rebuilds_on_store_count_change():
    store = _store([_rec("a", "s1", 0), _rec("b", "s1", 1)])
    exp = _expander(store)
    exp.expand([SearchResult("a", "a", 0.9)])          # builds at count=2
    store.add([_rec("c", "s1", 2)])                     # now b-c adjacency exists
    out = exp.expand([SearchResult("b", "b", 0.9)])
    assert "c" in [r.chunk_id for r in out]


def test_two_expanders_independent():
    s1 = _store([_rec("a", "s1", 0), _rec("b", "s1", 1)])
    s2 = _store([_rec("x", "s2", 0)])
    e1, e2 = _expander(s1), _expander(s2)
    assert "b" in [r.chunk_id for r in e1.expand([SearchResult("a", "a", 0.9)])]
    assert [r.chunk_id for r in e2.expand([SearchResult("x", "x", 0.9)])] == ["x"]
