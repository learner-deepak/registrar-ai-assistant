import os
import sys
from pathlib import Path

# Ensure Python can find the 'app' module relative to backend root
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BACKEND_DIR))

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    TextLoader
)
from app.services.chunking import split_documents_into_chunks
from app.db.vector_store import add_chunks_to_database

def load_single_file(file_path: Path):
    """
    Routes the file to the appropriate loader based on its extension.
    """
    ext = file_path.suffix.lower()
    
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(str(file_path))
        elif ext == ".csv":
            loader = CSVLoader(str(file_path))
        elif ext in [".xlsx", ".xls"]:
            loader = UnstructuredExcelLoader(str(file_path), mode="elements")
        elif ext in [".txt", ".md"]:
            loader = TextLoader(str(file_path), encoding="utf-8")
        else:
            print(f"  ⚠️ Skipping unsupported file format: {file_path.name}")
            return []

        return loader.load()

    except Exception as e:
        print(f"  ❌ Failed to load {file_path.name}: {e}")
        return []

def fetch_local_documents(folder_path: Path):
    """
    Scans the local directory and loads all supported files.
    """
    if not folder_path.exists():
        print(f"❌ Could not find folder at: {folder_path}")
        return []

    print(f"📖 Scanning local folder: {folder_path}...\n")
    all_docs = []
    
    # Iterate through all files in the folder
    files = [f for f in folder_path.iterdir() if f.is_file()]
    
    if not files:
        print("⚠️ No files found in the docs directory.")
        return []

    for file_path in files:
        loaded_pages = load_single_file(file_path)
        if loaded_pages:
            print(f"  📄 Loaded '{file_path.name}': {len(loaded_pages)} segments")
            all_docs.extend(loaded_pages)

    print(f"\n✅ Successfully loaded {len(all_docs)} total segments across all files!")
    return all_docs

if __name__ == "__main__":
    # Point directly to data/docs
    DOCS_FOLDER = BACKEND_DIR.parent / "data" / "docs"

    try:
        print(f"1. Reading local files from: {DOCS_FOLDER}\n")
        raw_docs = fetch_local_documents(DOCS_FOLDER)

        if not raw_docs:
            print("\n⚠️ No document pages were loaded. Check file formats or extensions.")
            sys.exit(0)

        print("\n2. Splitting documents into text chunks...")
        chunks = split_documents_into_chunks(raw_docs, chunk_size=800, chunk_overlap=150)

        print("3. Generating OpenAI embeddings and saving to ChromaDB...")
        add_chunks_to_database(chunks)

        print(f"\n🎉 SUCCESS! {len(chunks)} document chunks were vectorized and stored in ChromaDB.")
    except Exception as e:
        print(f"\n❌ Error during local ingestion: {e}")