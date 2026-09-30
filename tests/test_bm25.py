from app.services import bm25_service


def test_word_tokenize_lowercases_and_removes_punctuation():
    assert bm25_service.word_tokenize("Primary Key!") == ["primary", "key"]


def test_bm25_returns_the_exact_keyword_chunk_first(monkeypatch):
    chunks = [
        {
            "chunk_id": 1,
            "filename": "keys.pdf",
            "page_label": "1",
            "page_content": "A primary key uniquely identifies a row."
        },
        {
            "chunk_id": 2,
            "filename": "models.pdf",
            "page_label": "2",
            "page_content": "A relational model organizes data in tables."
        },
        {
            "chunk_id": 3,
            "filename": "security.txt",
            "page_label": None,
            "page_content": "Authentication protects access to sensitive data."
        },
    ]
    monkeypatch.setattr(bm25_service, "get_all_chunks", lambda: chunks)

    results = bm25_service.find_closest_chunks_bm25("primary key", top_k=2)

    assert results[0]["chunk_id"] == 1
    assert results[0]["bm25_score"] > 0


def test_bm25_returns_no_results_when_no_terms_match(monkeypatch):
    chunks = [{"chunk_id": 1, "filename": "keys.pdf", "page_label": "1", "page_content": "A primary key uniquely identifies a row."}]
    monkeypatch.setattr(bm25_service, "get_all_chunks", lambda: chunks)

    assert bm25_service.find_closest_chunks_bm25("photosynthesis", top_k=2) == []
