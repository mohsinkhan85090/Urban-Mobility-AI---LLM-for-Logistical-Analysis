from retriever import retrieve_docs
from llm_layer import generate_rag_answer


def handle_rag_query(query: str) -> str:
    retrieved_docs = retrieve_docs(query)
    return generate_rag_answer(query, retrieved_docs)
