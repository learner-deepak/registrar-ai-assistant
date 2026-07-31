import sys
import os

# Ensure Python can find the 'app' module relative to the backend root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.db.vector_store import get_vector_store

def get_document_retriever(k_results: int = 10):
    """
    Creates a LangChain retriever that searches our ChromaDB database.
    k_results: The number of document chunks to return (default is top 10).
    """
    vector_store = get_vector_store()
    
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Changed from "similarity" to "mmr"
        search_kwargs={
            "k": 10, 
            "fetch_k": 30   # Fetches 30 chunks, then picks the 10 most diverse ones
        }
    )
    
    return retriever

def perform_search(query: str):
    """
    Takes a user question, searches the database, and returns the raw matching text chunks.
    """
    retriever = get_document_retriever()
    relevant_chunks = retriever.invoke(query)
    return relevant_chunks

# ==========================================
# TEST BLOCK: Testing the Search Engine!
# ==========================================
if __name__ == "__main__":
    test_question = "Can I exit the honors track early?"
    
    try:
        print(f"Question: '{test_question}'\n")
        print("Searching the database...")
        
        results = perform_search(test_question)
        
        print("\n Search complete! Here are the most relevant chunks found:\n")
        
        for i, doc in enumerate(results):
            print(f"--- Result {i+1} ---")
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Text: {doc.page_content}\n")
            
    except Exception as e:
        print(f"\n Error during retrieval: {e}")