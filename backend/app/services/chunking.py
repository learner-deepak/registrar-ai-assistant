from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_documents_into_chunks(documents: list[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """
    Takes a list of whole documents and breaks them down into smaller, overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Tries to split by paragraphs first, then lines, then words
    )
    
    # This automatically splits the text but preserves the metadata (like the file path) for every chunk!
    chunks = text_splitter.split_documents(documents)
    return chunks

# ==========================================
# TEST BLOCK: This only runs if we execute this specific file
# ==========================================
if __name__ == "__main__":
    from ingestion import load_university_document
    
    test_file_path = "../data/academic_guidelines.txt"
    
    try:
        # 1. Load the document using our previous function
        raw_documents = load_university_document(test_file_path)
        
        # 2. Split the document into chunks 
        # (We use a tiny chunk size of 150 here just to test that it actually cuts the text)
        document_chunks = split_documents_into_chunks(raw_documents, chunk_size=150, chunk_overlap=20)
        
        print(f"✅ Splitting complete! 1 Document became {len(document_chunks)} chunks.\n")
        
        # 3. Print out each chunk to see the result
        for i, chunk in enumerate(document_chunks):
            print(f"--- Chunk {i+1} ---")
            print(chunk.page_content)
            print(f"Metadata: {chunk.metadata}\n")
            
    except Exception as e:
        print(f"❌ Error chunking document: {e}")