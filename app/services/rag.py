from dotenv import load_dotenv, find_dotenv
import os
from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage


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

def ask_documents(user_query):
    top_k = os.getenv("TOP_K")
    search_results = vector_db.similarity_search(query=user_query,k=top_k)

    context = "\n\n\n".join([
        f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
        for result in search_results
    ])

    SYSTEM_PROMPT = f"""
    Your are a helpfull AI Assistant who answers user query based on the available cocntext retrieved from a PDF file along with page_contents and page number.
    you should only ans the user based on the following context and navigate the user to open the right page number to know more.

    Context:{context}
    """

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ])

    return {"response": response.content}
