from rank_bm25 import BM25Okapi
import numpy as np
import re
from app.services.chunks_registry import get_all_chunks

def word_tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def find_closest_chunks_bm25(query, top_k=6):
    all_chunks = get_all_chunks()

    if not all_chunks:
        return []

    tokenized_corpus = [
        word_tokenize(chunk["page_content"])
        for chunk in all_chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # Expand the query (optional, commented out for now)
    # query = expand_query(query)

    # Tokenize the query
    tokenized_query = word_tokenize(query)

    # Get BM25 scores for the tokenized query
    scores = bm25.get_scores(tokenized_query)

    # Get the top N chunks
    top_n_indices = np.argsort(scores) [ ::- 1] [: top_k]

    results = []

    for index in top_n_indices:
        if scores[index] <= 0:
            continue

        results.append({
            **all_chunks[index],
            "bm25_score": float(scores[index])
        })

    return results