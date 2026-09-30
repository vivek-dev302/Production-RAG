from app.services.citation_service import normalize_text, verify_citations


TOP_CHUNKS = [
    {
        "filename": "keys.pdf",
        "page_label": "2",
        "page_content": "A primary key uniquely identifies each row in a table.",
    }
]


def test_normalize_text_ignores_case_punctuation_and_extra_spaces():
    assert normalize_text("  Primary-Key!  ") == "primarykey"


def test_verify_citations_accepts_a_matching_snippet_and_metadata():
    response = {
        "citations": [
            {
                "title": "keys.pdf",
                "page": "2",
                "snippet": "primary key uniquely identifies each row",
            }
        ]
    }

    is_valid, citations = verify_citations(response, TOP_CHUNKS)

    assert is_valid is True
    assert citations == response["citations"]


def test_verify_citations_rejects_matching_text_with_wrong_page():
    response = {
        "citations": [
            {
                "title": "keys.pdf",
                "page": "3",
                "snippet": "primary key uniquely identifies each row",
            }
        ]
    }

    is_valid, citations = verify_citations(response, TOP_CHUNKS)

    assert is_valid is False
    assert citations == response["citations"]


def test_verify_citations_rejects_an_empty_citation_list():
    is_valid, citations = verify_citations({"citations": []}, TOP_CHUNKS)

    assert is_valid is False
    assert citations == []
