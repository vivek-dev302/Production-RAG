import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from app.services.document_registry import add_document
from app.services.chunks_registry import add_chunks
from langchain_community.document_loaders import PyPDFLoader, TextLoader

load_dotenv()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size = 650,
    chunk_overlap=100 
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

def load_file(file_path):
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(file_path).load()

    return TextLoader(file_path, encoding="utf-8").load()

def indexing_documents(files):
    results = []

    for file in files:
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

            print("attempting to store chunks")
            QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embedding_model,
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=os.getenv("QDRANT_COLLECTION")
            )

            print("adding document in sqlite")
            # This runs only when Qdrant indexing succeeded.
            document_id = add_document(
                filename=file["filename"],
                file_path=str(file["path"]),
                file_type=Path(file["path"]).suffix,
                size_bytes=file["size"],
                chunk_count=len(chunks),
                content_hash=file["content_hash"]
            )

            add_chunks(document_id,chunks)
            
            results.append({
                "status": "indexed",
                "filename": file["filename"],
                "chunk_count": len(chunks)
            })

        except Exception as error:
            results.append({
                "status": "failed",
                "filename": file["filename"],
                "detail": str(error)
            })

    return results