# Production RAG

A domain-specific **Ask My Documents** API built with FastAPI. Upload PDF, TXT, or Markdown documents, then ask questions grounded in the indexed content.

## What it does

- Accepts multiple PDF, TXT, and Markdown uploads.
- Rejects empty or unsupported files and prevents duplicate indexing with a SHA-256 content hash.
- Splits text into token-aware chunks of 650 tokens with 100-token overlap.
- Stores document and chunk records in SQLite, and embeddings in Qdrant.
- Uses hybrid retrieval: BM25 keyword search + Qdrant semantic search, then a cross-encoder reranker selects the final top 3 chunks.
- Generates answers with a local Ollama model, citations, and an insufficient-evidence response when appropriate.
- Validates citation snippets against retrieved chunks, including filename and page metadata.

## Stack

- FastAPI
- LangChain
- Local Hugging Face embeddings and an Ollama LLM
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

The golden dataset is at `data/evaluation/golden_dataset.json`. The evaluation-only
dependencies are deliberately separate from the API/worker image. Install them once,
then run the single resumable evaluator from the project root:

```powershell
pip install -r evaluation/requirements.txt
python -m evaluation.run_evaluation
```

It uses the same local Ollama model as the RAG application. It runs one retrieval and
one answer generation per dataset question, then writes separate resumable reports for
BM25 and semantic retrieval at 6 chunks each, reranked retrieval at 3 chunks, with
hit-rate and mean-recall metrics, plus citations, refusal, faithfulness, and answer
correctness under `data/evaluation/results/`.

Run the evaluator against the Dockerized corpus with:

```powershell
docker compose --profile evaluation run --build --rm evaluation
```

The service shares the application SQLite/upload volume and the internal Qdrant
network. The golden dataset and generated reports are bind-mounted from
`data/evaluation/`, so you can inspect the reports directly on your machine.

## CI

GitHub Actions runs the unit-test suite for every push and pull request through `.github/workflows/tests.yml`.
