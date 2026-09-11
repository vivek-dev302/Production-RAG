import sqlite3
from pathlib import Path

DB_PATH = Path("data/documents.db")


def create_chunks_table():
    DB_PATH.parent.mkdir(exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER NOT NULL,
            chunk_index INTEGER NOT NULL,
            page_content TEXT NOT NULL,
            page_label TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,

            UNIQUE(document_id, chunk_index)
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    connection.commit()
    connection.close()

def add_chunks(document_id,chunks):
    connection = sqlite3.connect(DB_PATH)

    for index, chunk in enumerate(chunks):
        page_label = chunk.metadata.get("page_label")

        connection.execute("""
            INSERT INTO chunks (
                document_id, chunk_index, page_content, page_label
            )
            VALUES (?, ?, ?, ?)
        """, (document_id, index, chunk.page_content, page_label))

    connection.commit()
    connection.close()

def get_all_chunks():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    rows = connection.execute("""
        SELECT * FROM chunks
        ORDER BY created_at DESC
    """).fetchall()

    connection.close()

    return [dict(row) for row in rows]

def get_chunks_by_document_id(document_id):
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    rows = connection.execute("""
        SELECT * FROM chunks
        WHERE document_id = ?
    """, (document_id,)).fetchall()

    connection.close()

    if rows is None:
        return None

    return [dict(row) for row in rows]