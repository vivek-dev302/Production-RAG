import sqlite3
from pathlib import Path

DB_PATH = Path("data/documents.db")


def create_documents_table():
    DB_PATH.parent.mkdir(exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,
            uploaded_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()

def add_document(filename, file_path, file_type, size_bytes, chunk_count,content_hash):
    connection = sqlite3.connect(DB_PATH)

    cursor = connection.execute("""
        INSERT INTO documents (
            filename, file_path, file_type, size_bytes, chunk_count,content_hash
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (filename, file_path, file_type, size_bytes, chunk_count,content_hash))

    connection.commit()
    document_id = cursor.lastrowid
    connection.close()

    return document_id

def get_all_documents():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    rows = connection.execute("""
        SELECT * FROM documents
        ORDER BY uploaded_at DESC
    """).fetchall()

    connection.close()

    return [dict(row) for row in rows]

def get_document_by_hash(content_hash):
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    row = connection.execute("""
        SELECT * FROM documents
        WHERE content_hash = ?
    """, (content_hash,)).fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)