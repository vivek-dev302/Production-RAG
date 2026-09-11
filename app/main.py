from fastapi import FastAPI, UploadFile, File
from typing import List, Annotated
from app.services.ingestion import indexing_documents
from app.services.rag import ask_documents
from app.services.chunks_registry import create_chunks_table
from pathlib import Path
import hashlib
from app.services.document_registry import create_documents_table, get_all_documents,get_document_by_hash
from app.schemas import QueryRequest

app = FastAPI()

@app.on_event("startup")
def startup():
    create_documents_table()
    create_chunks_table()

@app.get("/")
def root():
    return {"message": "Hello, FastAPI!"}

@app.get("/health")
def health_check():
    return {"health": "ok"}

@app.get("/documents")
def list_documents():
    return {"documents": get_all_documents()}

@app.post("/documents/ingest", status_code=201)
async def ingest_documents(
    files: Annotated[List[UploadFile], File(description="Upload PDF, TXT, or Markdown files")]
    ):

    upload_folder = Path("data/uploads")
    upload_folder.mkdir(parents=True, exist_ok=True)

    skipped_files = []
    files_to_index = []
    hashes_in_this_request = set()

    for file in files:

        suffix = Path(file.filename).suffix.lower()
        if not file.filename or suffix not in {".pdf", ".txt", ".md"}:
            skipped_files.append({
                "status": "rejected",
                "filename": file.filename,
                "detail": "Only PDF, TXT, and Markdown files are supported."
            })
            continue

        content = await file.read()
        if not content:
            skipped_files.append({
                "status": "rejected",
                "filename": file.filename,
                "detail": "The upload file is empty."
            })
            continue
        
        content_hash = hashlib.sha256(content).hexdigest()

        if content_hash in hashes_in_this_request:
            skipped_files.append({
                "status": "already_indexed",
                "filename": file.filename,
                "detail": "This same file was included earlier in this upload."
            })
            continue

        existing_document = get_document_by_hash(content_hash)

        if existing_document:
            skipped_files.append({
                "status": "already_indexed",
                "filename": file.filename,
                "document_id": existing_document["id"]
            })
            continue

        hashes_in_this_request.add(content_hash)
        
        safe_filename = Path(file.filename).name
        file_path = upload_folder / safe_filename

        with open(file_path, "wb") as f:
            f.write(content)

        files_to_index.append({
            "filename": file.filename,
            "path":file_path, 
            "size": len(content),
            "content_hash":content_hash
        })

    indexing_results = indexing_documents(files_to_index)

    return {
        "documents": skipped_files + indexing_results
    }

@app.post("/query")
def ask_query(query: QueryRequest):
    return ask_documents(query.question)