from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# Load PDF
loader = PyPDFLoader("documents/RAG_Practice_Company_Handbook.pdf")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create/load vector database
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Add documents only when creating the database for the first time
# vector_store.add_documents(chunks)

# Get question from user
question = input("Enter your question: ")

# Search for relevant chunks
results = vector_store.similarity_search(question, k=2)

print("\nRelevant information:\n")

for result in results:
    print(result.page_content)
    print("--------------------")

# Load Ollama model
llm = ChatOllama(
    model="qwen3:4b"
)

# Create context from retrieved chunks
context = "\n\n".join([result.page_content for result in results])

# Create prompt
prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

# Generate answer
response = llm.invoke(prompt)

# Print answer
print("\nRAG ANSWER:")
print(response.content)