from types import SimpleNamespace

from app.services import chunks_registry, document_registry


def test_document_and_chunks_are_saved_and_linked(monkeypatch, tmp_path):
    database_path = tmp_path / "test_documents.db"
    monkeypatch.setattr(document_registry, "DB_PATH", database_path)
    monkeypatch.setattr(chunks_registry, "DB_PATH", database_path)
    document_registry.create_documents_table()
    chunks_registry.create_chunks_table()

    document_id = document_registry.add_document(
        filename="notes.txt", file_path="data/uploads/notes.txt", file_type=".txt",
        size_bytes=42, chunk_count=2, content_hash="test-hash-123",
    )
    chunks = [
        SimpleNamespace(page_content="First chunk", metadata={}),
        SimpleNamespace(page_content="Second chunk", metadata={}),
    ]

    chunk_ids = chunks_registry.add_chunks(document_id, chunks)
    saved_chunks = chunks_registry.get_chunks_by_document_id(document_id)

    assert document_id == 1
    assert len(chunk_ids) == 2
    assert [chunk["chunk_index"] for chunk in saved_chunks] == [0, 1]
    assert [chunk["page_content"] for chunk in saved_chunks] == ["First chunk", "Second chunk"]


def test_delete_chunks_removes_only_the_selected_document(monkeypatch, tmp_path):
    database_path = tmp_path / "test_documents.db"
    monkeypatch.setattr(document_registry, "DB_PATH", database_path)
    monkeypatch.setattr(chunks_registry, "DB_PATH", database_path)
    document_registry.create_documents_table()
    chunks_registry.create_chunks_table()

    first_document_id = document_registry.add_document("first.txt", "first.txt", ".txt", 1, 1, "hash-1")
    second_document_id = document_registry.add_document("second.txt", "second.txt", ".txt", 1, 1, "hash-2")
    chunks_registry.add_chunks(first_document_id, [SimpleNamespace(page_content="first", metadata={})])
    chunks_registry.add_chunks(second_document_id, [SimpleNamespace(page_content="second", metadata={})])

    chunks_registry.delete_chunks_by_document_id(first_document_id)

    assert chunks_registry.get_chunks_by_document_id(first_document_id) == []
    assert len(chunks_registry.get_chunks_by_document_id(second_document_id)) == 1
