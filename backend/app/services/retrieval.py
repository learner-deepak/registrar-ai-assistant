import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.db.vector_store import get_vector_store
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

def perform_search(query: str, k: int = 5):
    """
    Performs a Hybrid Search using both Vector (Chroma) and Keyword (BM25) search.
    Fuses the results using a custom Reciprocal Rank Fusion (RRF) algorithm to 
    completely bypass LangChain's broken dependency.
    """
    db = get_vector_store()
    
    # 1. Fetch Vector Search Results (Chroma)
    vector_retriever = db.as_retriever(search_kwargs={"k": k})
    vector_docs = vector_retriever.invoke(query)
    
    # 2. Setup & Fetch BM25 Keyword Search Results
    collection_data = db.get()
    bm25_docs = [
        Document(page_content=text, metadata=meta) 
        for text, meta in zip(collection_data["documents"], collection_data["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = k
    keyword_docs = bm25_retriever.invoke(query)
    
    # 3. Custom Reciprocal Rank Fusion (RRF)
    # We manually score and merge the documents to avoid broken imports
    rrf_scores = {}
    doc_lookup = {}
    
    # Score Vector results
    for rank, doc in enumerate(vector_docs):
        content = doc.page_content
        doc_lookup[content] = doc
        # RRF Formula: 1 / (rank + constant)
        rrf_scores[content] = rrf_scores.get(content, 0) + (1 / (rank + 60))
        
    # Score BM25 Keyword results
    for rank, doc in enumerate(keyword_docs):
        content = doc.page_content
        doc_lookup[content] = doc
        rrf_scores[content] = rrf_scores.get(content, 0) + (1 / (rank + 60))
        
    # Sort documents by their combined RRF score (highest to lowest)
    reranked_docs = sorted(doc_lookup.values(), key=lambda d: rrf_scores[d.page_content], reverse=True)
    
    # Return the top 'k' fused results
    return reranked_docs[:k]

if __name__ == "__main__":
    test_query = "CS-101" 
    print(f"--- Testing Custom Hybrid Search ---")
    print(f"Query: {test_query}\n")
    
    docs = perform_search(test_query)
    for i, doc in enumerate(docs):
        print(f"Result {i+1} (Source: {doc.metadata.get('source')}):")
        print(f"{doc.page_content[:150]}...\n")