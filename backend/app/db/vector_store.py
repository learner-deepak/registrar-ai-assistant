import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# 1. Load the secret API key from the .env file
load_dotenv()

def get_embedding_model():
    """
    Initializes the Google Gemini Embedding model.
    It automatically finds the GOOGLE_API_KEY in the environment.
    """
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def get_vector_store() -> Chroma:
    """
    Connects to our local ChromaDB database.
    If the database doesn't exist yet, Chroma will automatically create it.
    """
    # Look for the DB path in .env, default to "../data/chroma_db" if not found
    db_dir = os.getenv("CHROMA_DB_DIR", "../data/chroma_db")
    embeddings = get_embedding_model()
    
    # Initialize the database connection
    vector_store = Chroma(
        persist_directory=db_dir,
        embedding_function=embeddings,
        collection_name="registrar_documents" # Think of a collection like a SQL table
    )
    return vector_store

def add_chunks_to_database(chunks: list[Document]):
    """
    Takes text chunks, generates Google embeddings for them, and saves them to ChromaDB.
    """
    vector_store = get_vector_store()
    
    # This single line calls Google's API for the math, then saves it to your hard drive!
    vector_store.add_documents(chunks)

# ==========================================
# TEST BLOCK: Putting Phase 2, 3, and 4 together!
# ==========================================
if __name__ == "__main__":
    import sys
    # This allows our test block to find the other files in the 'app' folder
    sys.path.append('.') 
    
    from app.services.ingestion import load_university_document
    from app.services.chunking import split_documents_into_chunks
    
    test_file_path = "../data/academic_guidelines.txt"
    
    try:
        print("1. Reading the mock PDF/TXT file...")
        raw_documents = load_university_document(test_file_path)
        
        print("2. Splitting the document into chunks...")
        document_chunks = split_documents_into_chunks(raw_documents, chunk_size=150, chunk_overlap=20)
        
        print("3. Contacting Google Gemini to generate embeddings and saving to ChromaDB...")
        print("(This requires internet and might take a few seconds...)")
        
        add_chunks_to_database(document_chunks)
        
        print(f"\n✅ SUCCESS! {len(document_chunks)} chunks were vectorized and saved to your database.")
        
    except Exception as e:
        print(f"\n❌ Error during vector indexing: {e}")