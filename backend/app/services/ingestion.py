import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_core.documents import Document

def load_university_document(file_path: str) -> list[Document]:
    """
    Reads a file (PDF, TXT, or DOCX) and extracts its text into LangChain Document objects.
    """
    # 1. Extract the file extension (e.g., '.pdf', '.txt')
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # 2. Choose the correct LangChain loader based on the extension
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Please upload PDF, TXT, or DOCX.")
    
    # 3. Load and return the extracted text and metadata
    return loader.load()

# ==========================================
# TEST BLOCK: This only runs if we execute this specific file
# ==========================================
if __name__ == "__main__":
    # Since our terminal is in the 'backend' folder, 
    # we only go up ONE level to reach 'data'.
    test_file_path = "../data/academic_guidelines.txt"
    
    try:
        documents = load_university_document(test_file_path)
        
        print("✅ Document successfully loaded!\n")
        print(f"Total Pages/Chunks: {len(documents)}")
        print(f"Citation Metadata: {documents[0].metadata}")
        print("-" * 40)
        print("Text Preview:")
        print(documents[0].page_content[:150] + "...")
        
    except Exception as e:
        print(f"❌ Error loading document: {e}")