# Production RAG

A domain-specific **Ask My Documents** API built with FastAPI. Upload PDF, TXT, or Markdown documents, then ask questions grounded in the indexed content.

## What it does

- Accepts multiple PDF, TXT, and Markdown uploads.
- Rejects empty or unsupported files and prevents duplicate indexing with a SHA-256 content hash.
- Splits text into token-aware chunks of 650 tokens with 100-token overlap.
- Stores document and chunk records in SQLite, and embeddings in Qdrant.
- Uses hybrid retrieval: BM25 keyword search + Qdrant semantic search, then a cross-encoder reranker selects the final top 3 chunks.
- Generates answers with Gemini, citations, and an insufficient-evidence response when appropriate.
- Validates citation snippets against retrieved chunks, including filename and page metadata.

## Stack

- FastAPI
- LangChain
- Gemini embeddings and LLM
- Qdrant
- SQLite
- BM25 (`rank-bm25`)
- Sentence Transformers cross-encoder reranker
- Pytest and GitHub Actions

## Setup

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=your_collection_name
```

Run the API:

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to use the Swagger interface.

## API usage

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Check that the API is running. |
| `GET /documents` | List indexed documents from SQLite. |
| `POST /documents/ingest` | Upload one or more PDF, TXT, or Markdown files using the `files` form field. |
| `POST /query` | Ask a question about indexed documents. |

Example query body:

```json
{
  "question": "What is a primary key?"
}
```

The response contains `status`, `response`, `citations`, and `is_cited`. Each citation includes its source title, page, and supporting snippet.

## Testing and evaluation

Run unit tests:

```powershell
python -m pytest -q
```

The golden dataset is at `data/evaluation/golden_dataset.json`. Run evaluations from the project root:

```powershell
python -m evaluation.run_retrieval_evaluation
python -m evaluation.run_refusal_evaluation
python -m evaluation.run_citation_evaluation
```

The scripts save resumable reports under `data/evaluation/results/`. They use the Gemini API, so run them manually rather than in CI.

## CI

GitHub Actions runs the unit-test suite for every push and pull request through `.github/workflows/tests.yml`.
