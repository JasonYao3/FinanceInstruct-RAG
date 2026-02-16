import os
# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from datasets import load_dataset

# Configuration
DB_PATH = "./chroma_db"
DATASET_NAME = "Josephgflowers/Finance-Instruct-500k"

def run_ingestion():
    # TODO: 1. Load the dataset
    # Look at datasets.load_dataset()
    # Hint: Start with split="train[:1000]" to save time during dev
    
    # TODO: 2. Prepare the documents
    # The dataset has 'system', 'user', 'assistant' columns.
    # storage_text = f"Question: {row['user']}\nAnswer: {row['assistant']}"
    # Create LangChain Document objects
    
    # TODO: 3. Initialize Embeddings
    # use HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # TODO: 4. Create/Persist VectorStore
    # Chroma.from_documents(...)
    
    print("Ingestion complete!")

if __name__ == "__main__":
    run_ingestion()
