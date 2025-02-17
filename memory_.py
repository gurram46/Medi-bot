import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

# Load environment variables from the .env file (if present)
load_dotenv(find_dotenv())

# Directory containing your PDF files
DATA_PATH = "data/"

def load_pdf_files(data_path):
    """
    Load PDF files from the specified directory using DirectoryLoader and PyPDFLoader.
    """
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """
    Split loaded documents into smaller text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding_model_instance():
    """
    Initialize and return the HuggingFace embeddings model.
    """
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(text_chunks, embedding_model):
    """
    Create a FAISS vectorstore from text chunks using the provided embeddings model.
    """
    vectorstore = FAISS.from_documents(text_chunks, embedding_model)
    return vectorstore

def main():
    # Define the path where the vectorstore will be saved locally
    DB_FASS_PATH = "vectorstore/db_faiss"
    
    # Check if the vectorstore already exists (by checking for a key file, e.g., "index.faiss")
    index_file = os.path.join(DB_FASS_PATH, "index.faiss")
    if os.path.exists(index_file):
        print("Loading existing vectorstore from disk...")
        embedding_model = get_embedding_model_instance()
        db = FAISS.load_local(DB_FASS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("Vectorstore loaded successfully.")
    else:
        print("Vectorstore not found. Processing PDFs to build vectorstore...")
        documents = load_pdf_files(DATA_PATH)
        print(f"Loaded {len(documents)} PDF document(s).")
        
        text_chunks = create_chunks(documents)
        print(f"Created {len(text_chunks)} text chunk(s).")
        
        embedding_model = get_embedding_model_instance()
        db = build_vectorstore(text_chunks, embedding_model)
        print("Vectorstore built successfully.")
        
        db.save_local(DB_FASS_PATH)
        print(f"Vectorstore saved locally at: {DB_FASS_PATH}")

if __name__ == "__main__":
    main()
