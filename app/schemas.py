from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

class Citation(BaseModel):
    document_id: str
    filename: str
    page: int | None
    chunk_id: str

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]