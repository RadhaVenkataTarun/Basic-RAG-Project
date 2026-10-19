
import os
import time
import re
from typing import TypedDict, List

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END


# 1. Configuration
PDF_PATH = "documents/RAG_Practice_Company_Handbook.pdf"


# 2. Load embedding model
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 3. Connect to existing ChromaDB
vector_store = Chroma(
    collection_name="langchain",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

print(
    "Existing ChromaDB chunk count:",
    vector_store._collection.count()
)


# 4. Initialize Gemini for generation and verification
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set in this terminal."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    temperature=0,
    max_output_tokens=256
)

verifier_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    google_api_key=api_key,
    temperature=0,
    max_output_tokens=10
)

print("Generation LLM: Gemini 3.1 Flash Lite")
print("Verification LLM: Gemini 3.1 Flash Lite")
print("RAG system ready.")


# 5. Shared LangGraph state
class RAGState(TypedDict):
    question: str
    documents: List[Document]
    context: str
    answer: str
    verification: str
    attempts: int
    sources: List[str]


# 6. Extract clean text from Gemini responses
def extract_response_text(content) -> str:

    if isinstance(content, list):
        return " ".join(
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ).strip()

    return str(content or "").strip()


# 7. Retrieve relevant chunks
def retrieve_node(state: RAGState) -> dict:

    question = state["question"]

    results = vector_store.similarity_search(
        question,
        k=6
    )

    context_parts = []
    sources = []

    for index, doc in enumerate(results, start=1):

        source = doc.metadata.get("source", PDF_PATH)
        page = doc.metadata.get("page")

        page_label = (
            str(page + 1)
            if isinstance(page, int)
            else "Unknown"
        )

        context_parts.append(
            f"[Source {index}: {source}, Page {page_label}]\n"
            f"{doc.page_content}"
        )

        source_label = f"{source} - Page {page_label}"

        if source_label not in sources:
            sources.append(source_label)

    context = "\n\n".join(context_parts)

    print(f"\nRetrieved {len(results)} relevant chunks.")
    print("\n========== RETRIEVED CONTEXT ==========")
    print(context)

    return {
        "documents": results,
        "context": context,
        "sources": sources
    }


# 8. Generate answer using Gemini
def generate_node(state: RAGState) -> dict:

    prompt = f"""
You are a document question-answering assistant.

Answer using only the supplied context.

Rules:
1. Support every factual claim with the context.
2. Do not invent names, dates, numbers, or facts.
3. If the answer is absent, say so clearly.
4. Answer directly and concisely.
5. Cite relevant document pages as [Page N].
6. Do not claim information is missing if it is present.

Context:
{state["context"]}

Question:
{state["question"]}

Answer:
"""

    print(
        f"\nGenerating answer with Gemini "
        f"(attempt {state['attempts'] + 1})..."
    )

    start_time = time.time()

    response = llm.invoke(prompt)

    generation_time = time.time() - start_time
    answer = extract_response_text(response.content)

    print(f"Generation time: {generation_time:.2f} seconds")
    print(f"Answer length: {len(answer)} characters")

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty answer."
        )

    return {
        "answer": answer,
        "attempts": state["attempts"] + 1
    }


# 9. Verify answer using Gemini
def verify_node(state: RAGState) -> dict:

    verification_prompt = f"""
You are a strict evidence verification assistant.

Question:
{state["question"]}

Retrieved document context:
{state["context"]}

Proposed answer:
{state["answer"]}

Rules:
1. Return SUPPORTED if the context supports the answer's
   factual claims, even if the wording differs.
2. Return UNSUPPORTED if a factual claim is contradicted
   or lacks evidence in the supplied context.
3. Return INCONCLUSIVE if the context is insufficient
   to decide.
4. Do not use outside knowledge.
5. Do not reject an answer merely because it is concise.
6. Return exactly one word:
   SUPPORTED, UNSUPPORTED, or INCONCLUSIVE.

Verdict:
"""

    print("\nVerifying answer with Gemini...")

    start_time = time.time()

    result = verifier_llm.invoke(verification_prompt)

    verification_time = time.time() - start_time

    verifier_text = extract_response_text(result.content)

    print(f"Verifier response: {verifier_text}")
    print(f"Verification time: {verification_time:.2f} seconds")

    match = re.fullmatch(
        r"\s*(SUPPORTED|UNSUPPORTED|INCONCLUSIVE)\s*[.!]?\s*",
        verifier_text.upper()
    )

    if match:
        verification = match.group(1)
    else:
        verification = "INCONCLUSIVE"

    print("Grounding verification:", verification)

    return {
        "verification": verification
    }


# 10. Fallback answer
def fallback_node(state: RAGState) -> dict:

    return {
        "answer": (
            "I could not produce an answer that passed "
            "the grounding check using the retrieved "
            "document passages. Please rephrase your "
            "question or consult the original document."
        )
    }


# 11. Route after retrieval
def route_after_retrieval(state: RAGState) -> str:

    if not state["context"].strip():
        return "no_context"

    return "generate"


# 12. Route after verification
def route_after_verification(state: RAGState) -> str:

    if state["verification"] == "SUPPORTED":
        return "finish"

    if state["attempts"] < 2:
        return "retry"

    return "fallback"


# 13. Build and compile LangGraph workflow
workflow = StateGraph(RAGState)

workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("verify", verify_node)
workflow.add_node("fallback", fallback_node)

workflow.add_edge(START, "retrieve")

workflow.add_conditional_edges(
    "retrieve",
    route_after_retrieval,
    {
        "generate": "generate",
        "no_context": "fallback"
    }
)

workflow.add_edge("generate", "verify")

workflow.add_conditional_edges(
    "verify",
    route_after_verification,
    {
        "finish": END,
        "retry": "generate",
        "fallback": "fallback"
    }
)

workflow.add_edge("fallback", END)

# FastAPI imports this compiled graph
app = workflow.compile()

print("LangGraph workflow compiled successfully.")


# 14. Interactive execution when run directly
if __name__ == "__main__":

    question = input("\nEnter your question: ").strip()

    if not question:
        print("Please enter a question.")

    else:
        initial_state: RAGState = {
            "question": question,
            "documents": [],
            "context": "",
            "answer": "",
            "verification": "",
            "attempts": 0,
            "sources": []
        }

        total_start_time = time.time()

        final_state = app.invoke(initial_state)

        total_time = time.time() - total_start_time

        print("\n========== RAG ANSWER ==========")
        print(final_state["answer"])

        print("\n========== SOURCES ==========")

        if final_state["sources"]:
            for source in final_state["sources"]:
                print(f"- {source}")
        else:
            print("No supporting document passages retrieved.")

        print("\n========== VERIFICATION ==========")
        print(final_state.get("verification", "NOT CHECKED"))

        print("Generation attempts:", final_state["attempts"])
        print(f"Total execution time: {total_time:.2f} seconds")
