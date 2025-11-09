# ingest.py - Responsible for creating and persisting the vector database
# USES: HuggingFace (MiniLM) for embeddings and ChromaDB for local storage

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from the standard .env file
# This loads the GOOGLE_API_KEY for use in app.py later.
load_dotenv() 

# --- Configuration ---
# Update the path to reflect the 'pdf_qa_project' subfolder
PDF_FILES = [
    "pdf_qa_project/paper1.pdf", 
    "pdf_qa_project/paper2.pdf", 
]
PERSIST_DIRECTORY = './chroma_db'

# Define the free, open-source embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_db():
    """Loads PDFs, splits text, creates embeddings, and saves the Chroma DB."""
    
    # 1. Load the documents
    documents = []
    print("Loading documents...")
    for file_path in PDF_FILES:
        try:
            # PyPDFLoader handles local paths
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        except Exception as e:
            # Error handler for missing PDF files
            print(f"Error loading {file_path}: {e}")
            print("Please ensure you have downloaded the PDFs and named them paper1.pdf and paper2.pdf inside 'pdf_qa_project'.")
            return

    # 2. Split the text into chunks
    print(f"Splitting {len(documents)} pages into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Initialize the Local Embedding Model
    print(f"Initializing local HuggingFace embedding model: {EMBEDDING_MODEL_NAME}")
    
    # HuggingFaceEmbeddings downloads the model weights (the files) once
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME, 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. Create and Persist the Vector Store
    # This step calculates the vectors (embeddings) and saves them locally.
    print("Creating and persisting vector store (this may take a couple of minutes)...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    
    print(f"✅ Vector store created and saved successfully to {PERSIST_DIRECTORY}")
    print(f"You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    create_vector_db()