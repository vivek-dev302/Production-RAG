import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from app.services.document_registry import add_document, delete_document
from app.services.chunks_registry import add_chunks, delete_chunks_by_document_id
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.services.embeddings import embedding_model

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size = 650,
    chunk_overlap=100 
)
def load_file(file_path):
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()

    return TextLoader(file_path, encoding="utf-8").load()

def indexing_documents(files):
    results = []

    for file in files:
        document_id = None

        try:
            print("loading document")
            docs = load_file(file["path"])

            print("chunking...")
            chunks = text_splitter.split_documents(docs)

            if not chunks:
                results.append({
                    "status": "failed",
                    "filename": file["filename"],
                    "detail": "No readable text was found."
                })
                continue

            print("adding document in sqlite")
            document_id = add_document(
                filename=file["filename"],
                file_path=str(file["path"]),
                file_type=Path(file["path"]).suffix,
                size_bytes=file["size"],
                chunk_count=len(chunks),
                content_hash=file["content_hash"]
            )

            chunk_ids = add_chunks(document_id,chunks)

            for chunk_index, (chunk, chunk_id) in enumerate(zip(chunks, chunk_ids)):
                chunk.metadata["chunk_id"] = chunk_id
                chunk.metadata["document_id"] = document_id
                chunk.metadata["chunk_index"] = chunk_index
                chunk.metadata["filename"] = file["filename"]

            print("attempting to store chunks")
            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embedding_model,
                url=os.getenv("QDRANT_URL"),
                collection_name=os.getenv("QDRANT_COLLECTION")
            )
            
            results.append({
                "status": "indexed",
                "filename": file["filename"],
                "chunk_count": len(chunks)
            })

        except Exception as error:
            if document_id is not None:
                delete_chunks_by_document_id(document_id)
                delete_document(document_id)
                
            results.append({
                "status": "failed",
                "filename": file["filename"],
                "detail": str(error)
            })

    return results