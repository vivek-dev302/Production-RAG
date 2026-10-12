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
- Redis and RQ for background query jobs
- BM25 (`rank-bm25`)
- Sentence Transformers cross-encoder reranker
- Docker Compose

## Run locally with Docker

### 1. Prerequisites

Install Docker Desktop and [Ollama](https://ollama.com/). Pull the local model once:

```powershell
ollama pull llama3.2:3b
```

Make sure Ollama is running. If the Ollama app is not already serving models:

```powershell
ollama serve
```

### 2. Create `.env`

Create a `.env` file in the project root:

```env
QDRANT_COLLECTION=Production_RAG
```

### 3. Start the application

```powershell
docker compose up --build
```

Docker starts FastAPI, the RQ worker, Redis, and Qdrant. The API is available at
`http://127.0.0.1:8000/docs`.

Use `docker compose up -d --build` to run it in the background. View worker logs with:

```powershell
docker compose logs -f worker
```

## API usage

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Check that the API is running. |
| `GET /documents` | List indexed documents from SQLite. |
| `POST /documents/ingest` | Upload one or more PDF, TXT, or Markdown files using the `files` form field. |
| `POST /query` | Queue a question about indexed documents. |
| `GET /query/{job_id}` | Poll the queued question until it finishes. |

### 1. Upload documents

In Swagger, use `POST /documents/ingest`, click **Try it out**, then upload one or
more PDF, TXT, or Markdown files through the `files` field. Wait for each document
to return `status: "indexed"` before querying.

### 2. Ask a question

Example query body:

```json
{
  "question": "What is a primary key?"
}
```

`POST /query` returns quickly with a job ID:

```json
{
  "status": "queued",
  "job_id": "your-job-id"
}
```

Use `GET /query/your-job-id` until its status is `finished`. Its `result` contains
the grounded answer, `is_cited`, and citations with document title, page, and
supporting snippet.

## Stop or reset

Stop containers while preserving indexed documents, SQLite data, Qdrant vectors, and
Redis data:

```powershell
docker compose down
```

To remove all local Docker data and start again from an empty index:

```powershell
docker compose down -v
```

`down -v` permanently deletes the local Qdrant index, uploaded files, SQLite database,
and Redis data.

## Tests and evaluation

Run unit tests from a local Python environment:

```powershell
python -m pytest -q
```

The repository includes a manually curated golden dataset in
`data/evaluation/golden_dataset.json` and evaluation tooling for retrieval, citation,
refusal, faithfulness, and answer correctness. No benchmark results are claimed in
this README.
