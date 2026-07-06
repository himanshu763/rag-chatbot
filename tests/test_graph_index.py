from rag.config import RagConfig
from rag.graph.index import GraphIndex
from rag.types import ChunkRecord


def _rec(cid, source, ordinal, entities=(), section="", parent=""):
    return ChunkRecord(chunk_id=cid, raw_text=cid,
                       provenance={"source": source, "title": source},
                       section_path=section, parent_text=parent or None,
                       entities=list(entities), ordinal=ordinal)


def _index(records):
    idx = GraphIndex(RagConfig())
    idx.build(records)
    return idx


def test_adjacency_links_consecutive_ordinals_same_source():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s1", 1), _rec("c", "s1", 2)])
    assert idx.neighbors("b", hops=1) == {"a", "c"}


def test_no_adjacency_across_sources():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s2", 0)])
    assert idx.neighbors("a", hops=1) == set()


def test_section_edges_link_same_section():
    idx = _index([_rec("a", "s1", 0, section="Intro"), _rec("b", "s1", 5, section="Intro")])
    assert "b" in idx.neighbors("a", hops=1)


def test_entity_edges_link_cross_document():
    idx = _index([_rec("a", "s1", 0, entities=["zephyr x1"]),
                  _rec("b", "s2", 0, entities=["zephyr x1"])])
    assert idx.neighbors("a", hops=1) == {"b"}


def test_max_entity_fanout_skips_overcommon_entity():
    cfg = RagConfig(max_entity_fanout=2)
    recs = [_rec(f"c{i}", "s1", i, entities=["common"]) for i in range(5)]
    idx = GraphIndex(cfg); idx.build(recs)
    # "common" in 5 chunks > fanout 2 → no entity edges; c0 and c4 are non-adjacent
    assert "c4" not in idx.neighbors("c0", hops=1)


def test_two_hops_reach_further():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s1", 1), _rec("c", "s1", 2)])
    assert idx.neighbors("a", hops=1) == {"b"}
    assert idx.neighbors("a", hops=2) == {"b", "c"}


def test_seed_excluded_from_its_neighbors():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s1", 1)])
    assert "a" not in idx.neighbors("a", hops=2)


def test_expand_unions_neighbors_minus_seeds():
    idx = _index([_rec("a", "s1", 0), _rec("b", "s1", 1), _rec("c", "s1", 2)])
    assert idx.expand({"a", "b"}, hops=1) == {"c"}


def test_get_record():
    idx = _index([_rec("a", "s1", 0)])
    assert idx.get_record("a").chunk_id == "a"
    assert idx.get_record("missing") is None
