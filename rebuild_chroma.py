
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PDF_PATH = "documents/RAG_Practice_Company_Handbook.pdf"

# 1. Load the current PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# 2. Split the PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

# 3. Create the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Create a fresh ChromaDB database
vector_store = Chroma(
    collection_name="langchain",
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 5. Remove existing records to avoid stale or duplicate chunks
vector_store.reset_collection()

# 6. Insert the current PDF chunks
vector_store.add_documents(chunks)

print("ChromaDB rebuilt successfully.")
print("Total stored chunks:", vector_store._collection.count())

# 7. Verify that the performance review sentence is indexed
results = vector_store.similarity_search(
    "How often are formal performance reviews conducted?",
    k=4
)

print("\nRetrieved context:")

for doc in results:
    print("\nPage:", doc.metadata.get("page", -1) + 1)
    print(doc.page_content)
