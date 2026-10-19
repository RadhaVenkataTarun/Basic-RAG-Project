
# Basic RAG Project — LangGraph + FastAPI

A Retrieval-Augmented Generation (RAG) application that answers questions from a PDF company handbook using semantic search, Gemini, LangGraph, and FastAPI.

## Features

- **PDF-based question answering:** Answers questions using information retrieved from the company handbook.
- **Semantic retrieval:** Uses Hugging Face embeddings and ChromaDB to find relevant document chunks.
- **LLM answer generation:** Uses Google Gemini 3.1 Flash Lite.
- **Grounding verification:** Checks whether generated answers are supported by the retrieved context.
- **Retry and fallback:** Retries unsupported answers and provides a fallback response if verification fails.
- **FastAPI integration:** Exposes the RAG workflow through a REST API.
- **Source tracking:** Returns the source document and page numbers for retrieved passages.

## Tech Stack

- Python
- LangChain
- LangGraph
- FastAPI
- Google Gemini API
- Hugging Face Sentence Transformers
- ChromaDB
- Uvicorn

## Project Structure

```text
Basic-RAG-Project/
├── documents/
│   └── RAG_Practice_Company_Handbook.pdf
├── api.py
├── rag_langgraph.py
├── check_chroma.py
├── rebuild_chroma.py
├── .gitignore
└── README.md
```

## How It Works

1. The PDF handbook is split into text chunks and stored in ChromaDB as embeddings.
2. The user submits a question through the FastAPI endpoint.
3. LangGraph retrieves the six most relevant chunks from ChromaDB.
4. Gemini generates an answer using the retrieved context.
5. A second Gemini call verifies whether the answer is supported.
6. If verification fails, LangGraph retries once before returning a fallback response.
7. The API returns the answer, sources, verification status, generation attempts, and execution time.

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/RadhaVenkataTarun/Basic-RAG-Project.git
cd Basic-RAG-Project
```

### 2. Create a virtual environment

```bash
python -m venv ragenv
```

Activate it in Windows PowerShell:

```powershell
.\ragenv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn langchain langgraph langchain-huggingface langchain-chroma langchain-google-genai sentence-transformers
```

### 4. Configure the Gemini API key

Obtain an API key through Google AI Studio.

Set it in PowerShell:

```powershell
$env:GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
```

Replace the placeholder with your own key. Never commit API keys to GitHub.

### 5. Prepare ChromaDB

Ensure the PDF is available at:

```text
documents/RAG_Practice_Company_Handbook.pdf
```

Build or rebuild the ChromaDB collection using the repository's indexing script:

```powershell
python rebuild_chroma.py
```

Run this step when setting up the project for the first time or when rebuilding the document index.

## Running the Application

### Option 1: Interactive LangGraph execution

```powershell
python rag_langgraph.py
```

Enter a question when prompted.

### Option 2: Run the FastAPI server

```powershell
uvicorn api:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Usage

### Endpoint

```text
POST /ask
```

### Request

```json
{
  "question": "What is the annual learning budget?"
}
```

### Example response

```json
{
  "question": "What is the annual learning budget?",
  "answer": "Eligible employees can use up to INR 20,000 per year for approved learning. [Page 2]",
  "sources": [
    "RAG_Practice_Company_Handbook.pdf - Page 2"
  ],
  "verification": "SUPPORTED",
  "generation_attempts": 1,
  "execution_time_seconds": 2.5
}
```

The response values above are illustrative; actual answers and timings depend on the question and execution.

## Utility Scripts

- `check_chroma.py` — Inspects the ChromaDB collection and stored chunks.
- `rebuild_chroma.py` — Rebuilds the document index from the PDF.
- `rag_langgraph.py` — Contains the retrieval, generation, verification, retry, and fallback workflow.
- `api.py` — Exposes the compiled LangGraph workflow through FastAPI.

## Important Notes

- The Gemini API key must be configured before running the application.
- ChromaDB is stored locally and excluded from Git.
- The PDF must be available locally when building the index.
- The first run may take longer because the embedding model needs to load.
- Verification is an automated check and does not guarantee that every answer is factually correct.
- Do not commit secrets, virtual environments, or generated database files.

## Future Improvements

- Add a web-based chat interface.
- Improve retrieval using hybrid search and reranking.
- Add automated evaluation for retrieval quality and answer correctness.
- Add conversation history and deployment support.

## License

Add a license if you intend to distribute this project publicly.
