import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document 

# Paths
DATA_PATH = "data/memory_publication.json" 
VECTOR_DB_PATH = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_publications():
    """Loads the publication JSON and converts it into LangChain Documents."""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            publications = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load JSON data. Ensure the file exists and is valid. Error: {e}")
        return []

    docs = []
    for pub in publications:
        enriched_text = f"""
Title: {pub.get('title', 'Unknown')}
Author: {pub.get('username', 'Unknown')}
Description: {pub.get('publication_description', '')}
"""
        docs.append(Document(page_content=enriched_text.strip(), metadata={"id": pub.get("id", "N/A")}))
    return docs

def chunk_and_embed(docs):
    """Chunks documents, generates embeddings, and saves to FAISS."""
    if not docs:
        print("Ingestion aborted due to empty documents list.")
        return

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    print(f"Creating and saving FAISS vector store to {VECTOR_DB_PATH}...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    print(f"✅ Vectorstore saved at {VECTOR_DB_PATH}")

if __name__ == "__main__":
    docs = load_publications()
    chunk_and_embed(docs)