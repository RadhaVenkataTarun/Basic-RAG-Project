
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_langgraph import app as rag_graph, RAGState


# 1. Initialize FastAPI
app = FastAPI(
    title="PDF RAG API",
    description="Ask questions using LangGraph with grounding verification.",
    version="2.0.0"
)


# 2. Request model
class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to ask about the PDF"
    )


# 3. Homepage
@app.get("/")
def home():
    return {
        "message": "LangGraph PDF RAG API is running.",
        "docs": "/docs",
        "ask_endpoint": "/ask"
    }


# 4. Ask endpoint using the main LangGraph workflow
@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    initial_state: RAGState = {
        "question": question,
        "documents": [],
        "context": "",
        "answer": "",
        "verification": "",
        "attempts": 0,
        "sources": []
    }

    start_time = time.time()

    try:
        print(f"\n[API] Received question: {question}")

        # Execute the existing LangGraph workflow
        result = rag_graph.invoke(initial_state)

        total_time = time.time() - start_time

        print("[API] LangGraph execution completed.")
        print(
            f"[API] Total execution time: "
            f"{total_time:.2f} seconds"
        )

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "verification": result.get(
                "verification", "NOT CHECKED"
            ),
            "generation_attempts": result.get("attempts", 0),
            "execution_time_seconds": round(total_time, 2)
        }

    except Exception as exc:
        print("[API ERROR]", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=(
                "Question processing failed. "
                "Check the server terminal for details."
            )
        ) from exc
