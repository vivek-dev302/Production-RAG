from sentence_transformers import CrossEncoder


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)


def rerank_chunks(query, chunks, top_k=3):
    if not chunks:
        return []

    pairs = [
        (query, chunk["page_content"])
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    scored_chunks = []

    for chunk, score in zip(chunks, scores):
        scored_chunks.append({
            **chunk,
            "reranker_score": float(score)
        })

    scored_chunks.sort(
        key=lambda chunk: chunk["reranker_score"],
        reverse=True
    )

    return scored_chunks[:top_k]