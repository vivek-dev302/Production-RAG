def ask_documents_job(question: str):
    """RQ task: run the existing RAG pipeline outside the FastAPI request."""
    from app.services.rag import ask_documents
    return ask_documents(question)
