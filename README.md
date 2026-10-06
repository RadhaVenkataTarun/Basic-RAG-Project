# Basic RAG Project

A basic Retrieval-Augmented Generation (RAG) application that allows users to retrieve relevant information from PDF documents and generate answers using a local Large Language Model (LLM).

The project combines document processing, vector embeddings, similarity search, and a local LLM to provide context-aware answers.

---

## 🚀 Project Overview

This project demonstrates how a Retrieval-Augmented Generation (RAG) pipeline works.

Instead of asking the Large Language Model to answer a question only from its pretrained knowledge, the application first searches the provided documents for relevant information.

The retrieved information is then passed as context to the LLM, which generates the final answer based only on the retrieved context.

### RAG Pipeline

```text
PDF Document
     ↓
Document Loader
     ↓
Text Splitting
     ↓
HuggingFace Embeddings
     ↓
Chroma Vector Database
     ↓
Similarity Search
     ↓
Relevant Context
     ↓
Ollama - Qwen3 4B
     ↓
RAG Prompt
     ↓
Final Answer
🛠️ Technologies Used
- Python
- LangChain
- HuggingFace Embeddings
- ChromaDB
- Ollama
- Qwen3 4B
- PyPDF
- Virtual Environment
- Git & GitHub
🧠 What is RAG?
Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with Large Language Models.
The system first retrieves relevant information from a knowledge source and then provides that information to the LLM as context.
Traditional LLM
Question
   ↓
LLM
   ↓
Answer

RAG
Question
   ↓
Vector Search
   ↓
Relevant Documents
   ↓
Context
   ↓
LLM
   ↓
Answer

This helps the model generate answers based on the information available in the provided documents.
📂 Project Structure
Basic-RAG-Project/
│
├── main.py
├── .gitignore
├── README.md
│
├── documents/
│   └── PDF files
│
├── chroma_db/
│   └── Chroma vector database
│
└── ragenv/
    └── Python virtual environment

Note: documents/, chroma_db/, and ragenv/ are excluded from the Git repository using .gitignore.

⚙️ How the Project Works
1. Load the PDF
The application loads the PDF document and extracts its text.
2. Split the Text
The extracted text is divided into smaller chunks so that relevant sections can be efficiently retrieved.
3. Generate Embeddings
HuggingFace embeddings are used to convert the text chunks into numerical vector representations.
4. Store Embeddings
The generated embeddings are stored in a Chroma vector database.
5. Perform Similarity Search
When a question is provided, the application searches the Chroma database and retrieves the most relevant document chunks.
6. Build the Context
The retrieved document chunks are combined to create the context provided to the language model.
7. Generate the Answer
The context and question are passed to the local Qwen3 4B model running through Ollama.
The prompt instructs the model to answer using the information provided in the retrieved context.
💻 Installation
1. Clone the Repository
git clone https://github.com/RadhaVenkataTarun/Basic-RAG-Project.git

Navigate into the project:
cd Basic-RAG-Project

2. Create a Virtual Environment
python -m venv ragenv

Activate the environment on Windows:
ragenv\Scripts\activate

3. Install Required Packages
Install the required Python libraries:
pip install langchain
pip install langchain-community
pip install langchain-huggingface
pip install langchain-chroma
pip install langchain-ollama
pip install chromadb
pip install pypdf
pip install sentence-transformers

🦙 Ollama Setup
This project uses Ollama to run the language model locally.
Install Ollama and then download the Qwen3 4B model:
ollama pull qwen3:4b

Verify that the model is available:
ollama list

You should see:
qwen3:4b

Make sure Ollama is running before executing the Python application.
▶️ Running the Project
Activate the virtual environment:
ragenv\Scripts\activate

Run the application:
python main.py

The application will:
1. Load the document
2. Split the document into chunks
3. Generate embeddings
4. Store/search the embeddings using ChromaDB
5. Retrieve relevant information
6. Send the retrieved context to Qwen3 4B
7. Generate the final RAG answer
📝 Example
Question
How many annual leave days do employees get?

Retrieved Information
The system searches the PDF document and retrieves the relevant sections containing information about employee leave policies.
RAG Answer
18 days

The answer is generated using the retrieved document context.
🔍 Key Components
LangChain
Used to build and connect the different components of the RAG pipeline.
HuggingFace Embeddings
Used to convert text into vector representations for semantic similarity search.
ChromaDB
Used as the vector database for storing and retrieving document embeddings.
Ollama
Used to run the Large Language Model locally.
Qwen3 4B
The local LLM used to generate answers from the retrieved context.
🔐 Privacy
The LLM is run locally using Ollama.
The project does not require sending the document content to a third-party LLM API for generating the answer.
🎯 Learning Objectives
This project demonstrates:
- Understanding of Retrieval-Augmented Generation
- PDF document processing
- Text chunking
- Vector embeddings
- Vector databases
- Semantic similarity search
- Prompt construction
- Local LLM integration
- LangChain pipelines
- Ollama model integration
🚀 Future Enhancements
Possible improvements include:
- Add a Streamlit web interface
- Allow users to upload multiple PDFs
- Support conversational chat history
- Add source citations to generated answers
- Improve document chunking strategies
- Add multiple document collections
- Add configurable similarity-search parameters
- Deploy the application as a web application
👨‍💻 Author
Radha Venkata Tarun Motumarri
GitHub:
https://github.com/RadhaVenkataTarun
📜 License
This project is intended for educational and demonstration purposes.