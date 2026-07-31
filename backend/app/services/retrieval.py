import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.db.vector_store import get_vector_store
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

def perform_search(query: str, k: int = 5):
    """
    Performs a Hybrid Search using both Vector (Chroma) and Keyword (BM25) search.
    Returns a combined, deduplicated, and re-ranked list of top document chunks.
    """
    db = get_vector_store()
    
    # 1. Setup the Vector Retriever (Semantic Meaning)
    vector_retriever = db.as_retriever(search_kwargs={"k": k})
    
    # 2. Setup the BM25 Retriever (Keyword/Alphanumeric Matching)
    # We load the docs directly from the vector store's underlying collection to build the BM25 index
    collection_data = db.get()
    
    # Reconstruct the documents so Langchain's BM25 wrapper can read them
    from langchain_core.documents import Document
    bm25_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(collection_data["documents"], collection_data["metadatas"])
    ]
    
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k  # Set number of results to fetch
    
    # 3. Create the Ensemble Retriever (Hybrid Fusion)
    # weights=[0.5, 0.5] means both vector and keyword search carry equal importance
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )
    
    # 4. Fetch the fused results
    results = ensemble_retriever.invoke(query)
    
    return results

if __name__ == "__main__":
    test_query = "CS-101" # A great test for hybrid search!
    print(f"--- Testing Hybrid Search ---")
    print(f"Query: {test_query}\n")
    
    docs = perform_search(test_query)
    for i, doc in enumerate(docs):
        print(f"Result {i+1} (Source: {doc.metadata.get('source')}):")
        print(f"{doc.page_content[:150]}...\n")