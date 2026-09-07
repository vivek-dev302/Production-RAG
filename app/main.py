from fastapi import FastAPI, UploadFile, File
from typing import List, Annotated
from app.services.ingestion import indexing_documents
from app.services.rag import ask_documents
    
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, FastAPI!"}

@app.get("/health")
def health_check():
    return {"health": "ok"}

@app.get("/documents")
def list_documents():
    return {"documents": ["doc1","doc2","doc3"]}



@app.post("/documents/ingest", status_code=201)
async def ingest_documents(files: Annotated[List[UploadFile], File(description="Upload multiple files")]):
    results = []
    for file in files:
        content = await file.read()
        with open(file.filename, "wb") as f:
            f.write(content)
        results.append({"filename": file.filename, "size": len(content)})
    return indexing_documents(results)

@app.post("/query")
def ask_query(user_query):
    return ask_documents(user_query)

