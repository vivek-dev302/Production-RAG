import os
import json
import re
from rapidfuzz import fuzz
from dotenv import load_dotenv, find_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.services.bm25_service import find_closest_chunks_bm25

load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

vector_db = QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    collection_name = os.getenv("QDRANT_COLLECTION")
)

def normalize_text(text):
    # Convert to lowercase, remove punctuation, and normalize whitespace
    text = re.sub(r'[^\w\s]', '', text. lower())
    return ' '.join(text.split())

def verify_citations(response_dict, top_chunks):
    citations = response_dict.get("citations", [])

    if not citations:
        print("Error: No citations found in the response.")
        return False, []

    matches = []
    normalized_chunks = [normalize_text(chunk.page_content) for chunk in top_chunks]

    for citation in citations:
        cited_text = citation.get("snippet", "")
        if not cited_text:
            print(f"Error: Missing snippet in citation: {citation}")
            matches.append(False)
            continue

        # Normalize the entire citation
        normalized_citation = normalize_text(cited_text)

        match_found = False
        for chunk in normalized_chunks:
            print(f"Normalized chunk: {chunk}\nNormalized citation: {normalized_citation}\n")

            # Check if the normalized citation is a substring of the normalized chunk
            if normalized_citation in chunk:
                match_found = True
                break

            # If not an exact match, use fuzzy matching as a fallback
            score = fuzz.partial_ratio(normalized_citation, chunk)

            if score >= 95:
                match_found = True
                break

        matches.append(match_found)

    return all(matches),citations

def ask_documents(user_query):
    is_cited = False
    top_k = os.getenv("TOP_K")
    search_results = vector_db.similarity_search(query=user_query,k=top_k)
    top_bm25_chunks = find_closest_chunks_bm25(user_query,top_k)

    context = "\n\n\n".join([
        f"Document Title: {result.metadata['source']}\nPage Number: {result.metadata['page_label']}\nPage Content: {result.page_content}"
        for result in search_results
    ])

    SYSTEM_PROMPT = f"""You are a research assistant. Use the provided document snippets to
    answer the query. Format your response with citations in structured JSON format:

    response format:
    {{
        "status": "status of query response",
        "response": "Your response here.",
        "citations": [
            {{
                "citation_id": id of the citation mention in response,
                "title": "Document Title",
                "page": page number of the document
                "snippet": "Exact snippet from the document"
            }}
        ]
    }}

    IMPORTANT CITATION RULES:
    1. Each citation MUST be a complete sentence or phrase from the original text.
    2. Citations MUST be VERBATIM and EXACT quotes from the provided documents.
    3. DO NOT use ellipses ( ... ) or any other shortening techniques in citations.
    4. DO NOT paraphrase or modify the original text in any way for citations.
    5. If you need to use multiple sentences in a citation, include them in full.
    6. Use MULTIPLE citations when necessary to fully support your response.
    7. Ensure that each citation DIRECTLY supports a specific part of your response.
    8. If you.cannot find relevant information in the provided documents, state this clearly in your
    response.

    Example of a correct response with multiple, relevant citations:
    {{
        "status": "answered",
        "response": "The mockingbird has unique vocal .abilities . [1]. It uses these abilities for
        various purposes, including defending its territory [2].",
        "citations": [
            {{  
                "citation_id": 1,
                "title": "Mockingbird Study",
                "page": 4,
                "snippet": "The mockingbird is known for its ability to mimic the calls of other \n birds and even mechanical sounds."
            }},
            {{
                "citation_id": 2,
                "title": "Mockingbird Behavior",
                "page": 2,
                "snippet": "Mockingbirds use their diverse vocal repertoire to defend their \n territories from intruders and to attract mates."
            }}
        ]
    }}

    Example of a response with insufficient evidence
    {{
        "status": "insufficient_evidence",
        "response": "I do not have enough evidence in the uploaded documents to answer that question.",
        "citations": []
    }}

    Here are the relevant documents for your query:
    {context} 

    Remember:
    1. Citations must be EXACT, COMPLETE sentences or phrases from the provided text.
    2. Do not modify, shorten, or paraphrase the original text in your citations.
    3. Use multiple citations when necessary to fully support your response.
    4. Ensure each citation is directly relevant to the part of your response it supports.
    5. If you cannot find relevant information in the provided documents, clearly state this in your
    response.
    6. only respond in valid json format no text before or after strictly follow json schema 

    Now, please answer the given query using the provided information and following these guidelines

    """

    max_checks = 3
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]
    for i in range(max_checks):
        response = llm.invoke(messages)
        response_text = response.content

        try:
            response_dict = json.loads(response_text)
            print(f"Attempt {i+1}: {response_dict['response']}")
            for i, citation in enumerate(response_dict.get('citations', []), start=1):
                print(f"[{i}]: {citation['snippet']}")
            print("\n")

            if response_dict.get("status") == "insufficient_evidence":
                return {
                    "status": "insufficient_evidence",
                    "response": response_dict["response"],
                    "citations": [],
                    "is_cited": False,
                }
            
        except json.JSONDecodeError:
            messages.extend([
                AIMessage(content=response_text),
                HumanMessage(
                    "Your previous response was not valid JSON. "
                    "Return only valid JSON matching the required schema. "
                    "Do not include Markdown fences or any text outside the JSON."
                ),
            ])
            continue

        # Verify the citations
        citation_check, citations = verify_citations(response_dict, search_results)

        if citation_check:
            is_cited = True
            print("Citation check passed!\n")
            print(f"Query: {user_query}\n")
            print(f"Answer: {response_dict['response']}\n\ncitations:\n")

            for i, citation in enumerate(response_dict.get('citations', []), start=1):
                print(f"[{i}]: Document: {citation['title']}; Page: {citation['page']}; Snippet: {citation['snippet']}")
                break

            return {
                "status": response_dict.get("status", "answered"),
                "response": response_dict["response"],
                "citations": response_dict["citations"],
                "is_cited": True,
            }
        
        else:
            print("Citations check failed retrying...")
            messages.extend([
                AIMessage(content=response_text),
                HumanMessage(
                    "Your previous response had one or more citations that could not "
                    "be verified against the provided document snippets. "
                    "Generate the answer again. Use only exact verbatim snippets from "
                    "the provided context, or return insufficient_evidence with an empty citations list."
                ),
            ])

    return {
        "status": "citation_validation_failed",
        "response": "I could not produce a response with verified citations. Please try again.",
        "citations": [],
        "is_cited": False,
    }

