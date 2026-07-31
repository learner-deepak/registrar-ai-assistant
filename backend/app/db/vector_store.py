import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# 1. Load the secret API key (OPENAI_API_KEY) from the .env file
load_dotenv()

def get_embedding_model():
    """
    Initializes the OpenAI Embedding model (text-embedding-3-small).
    It automatically finds the OPENAI_API_KEY in the environment.
    """
    return OpenAIEmbeddings(model="text-embedding-3-small")

def get_vector_store() -> Chroma:
    """
    Connects to our local ChromaDB database.
    If the database doesn't exist yet, Chroma will automatically create it.
    """
    db_dir = os.getenv("CHROMA_DB_DIR", "../data/chroma_db")
    embeddings = get_embedding_model()
    
    vector_store = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name="registrar_documents"
    )
    return vector_store

def add_chunks_to_database(chunks: list[Document]):
    """
    Takes text chunks, generates OpenAI embeddings for them, and saves them to ChromaDB.
    """
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)

# ==========================================
# TEST BLOCK: Putting Phase 2, 3, and 4 together!
# ==========================================
if __name__ == "__main__":
    import sys
    sys.path.append('.') 
    
    from app.services.ingestion import load_university_document
    from app.services.chunking import split_documents_into_chunks
    
    test_file_path = "../data/academic_guidelines.txt"
    
    try:
        print("1. Reading the mock PDF/TXT file...")
        raw_documents = load_university_document(test_file_path)
        
        print("2. Splitting the document into chunks...")
        document_chunks = split_documents_into_chunks(raw_documents, chunk_size=150, chunk_overlap=20)
        
        print("3. Contacting OpenAI to generate embeddings and saving to ChromaDB...")
        print("(This requires internet and might take a few seconds...)")
        
        add_chunks_to_database(document_chunks)
        
        print(f"\n✅ SUCCESS! {len(document_chunks)} chunks were vectorized and saved to your database.")
        
    except Exception as e:
        print(f"\n❌ Error during vector indexing: {e}")