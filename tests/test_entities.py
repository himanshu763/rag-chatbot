from rag.entities import KeywordEntityExtractor
from rag.interfaces import EntityExtractor


def test_extractor_conforms_to_protocol():
    assert isinstance(KeywordEntityExtractor(), EntityExtractor)


def test_extracts_proper_noun_phrases_normalized():
    ents = KeywordEntityExtractor().extract("The Zephyr X1 turbine uses Fusion ERP Analytics.")
    assert "zephyr x1" in ents
    assert "fusion erp analytics" in ents


def test_excludes_common_lowercase_words():
    ents = KeywordEntityExtractor().extract("the turbine produces power at rated speed")
    assert ents == []


def test_deduplicates_and_is_deterministic():
    e = KeywordEntityExtractor()
    a = e.extract("Zephyr X1 and Zephyr X1 again")
    assert a == e.extract("Zephyr X1 and Zephyr X1 again")
    assert a.count("zephyr x1") == 1


def test_empty_text():
    assert KeywordEntityExtractor().extract("") == []


def test_drops_sentence_initial_common_word():
    ents = KeywordEntityExtractor().extract("The system is ready. Fusion ERP works.")
    assert "the" not in ents
    assert "fusion erp" in ents
