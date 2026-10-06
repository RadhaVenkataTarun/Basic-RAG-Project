from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

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

# User question
question = "How many annual leave days do employees get?"

# Search for relevant chunks

results = vector_store.similarity_search(question, k=2)

print("\nRelevant information:\n")

for result in results:
    print(result.page_content)
    print("--------------------")
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen3:4b"
)

context = "\n\n".join([result.page_content for result in results])

question = "How many annual leave days do employees get?"

prompt = f"""
Answer the question using only the information provided in the context.

Context:
{context}

Question:
{question}

Answer:
"""

response = llm.invoke(prompt)

print("\nRAG ANSWER:")
print(response.content)