from typing import Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Question to ask about indexed documents"
    )


class Citation(BaseModel):
    citation_id: int
    title: str
    page: str | None = None
    snippet: str

class QueryResponse(BaseModel):
    status: Literal[
        "answered",
        "insufficient_evidence",
        "citation_validation_failed"
    ]
    response: str
    citations: list[Citation] = []
    is_cited: bool

class IngestionItem(BaseModel):
    status: Literal[
        "indexed",
        "already_indexed",
        "rejected",
        "failed"
    ]
    filename: str | None = None
    chunk_count: int | None = None
    document_id: int | None = None
    detail: str | None = None


class IngestionResponse(BaseModel):
    documents: list[IngestionItem]

class DocumentRecord(BaseModel):
    id: int
    filename: str
    file_path: str
    file_type: str
    size_bytes: int
    chunk_count: int
    content_hash: str
    uploaded_at: str

class DocumentListResponse(BaseModel):
    documents: list[DocumentRecord]