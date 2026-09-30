import re

from rapidfuzz import fuzz


def normalize_text(text):
    # Convert to lowercase, remove punctuation, and normalize whitespace.
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(text.split())


def verify_citations(response_dict, top_chunks):
    citations = response_dict.get("citations", [])

    if not citations:
        print("Error: No citations found in the response.")
        return False, []

    matches = []
    normalized_chunks = [normalize_text(chunk["page_content"]) for chunk in top_chunks]

    for citation in citations:
        cited_text = citation.get("snippet", "")
        if not cited_text:
            print(f"Error: Missing snippet in citation: {citation}")
            matches.append(False)
            continue

        normalized_citation = normalize_text(cited_text)
        match_found = False

        for index, chunk in enumerate(normalized_chunks):
            print(f"Normalized chunk: {chunk}\nNormalized citation: {normalized_citation}\n")

            file_match = top_chunks[index].get("filename") == citation.get("title")
            page_match = str(top_chunks[index].get("page_label")) == str(citation.get("page"))

            if normalized_citation in chunk and file_match and page_match:
                match_found = True
                break

            score = fuzz.partial_ratio(normalized_citation, chunk)
            if score >= 95 and file_match and page_match:
                match_found = True
                break

        matches.append(match_found)

    return all(matches), citations
