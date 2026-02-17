import pandas as pd
import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ast
import json
from langchain_community.retrievers import BM25Retriever
import pickle

# Configuration
DB_PATH = "./chroma_db"
PARENT_STORE_PATH = "./parent_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed_chunks(
    child_path="data/chunked_dataset.parquet",
    parent_path="data/parent_documents.parquet",
):
    print("Embedding chunks with Parent Document Retrieval...")

    if not os.path.exists(child_path):
        print(f"Child documents not found: {child_path}")
        return

    if not os.path.exists(parent_path):
        print(f"Parent documents not found: {parent_path}")
        return

    # Load data
    child_df = pd.read_parquet(child_path)
    parent_df = pd.read_parquet(parent_path)

    print(f"Loaded {len(child_df)} child docs and {len(parent_df)} parent docs")

    # Initialize embeddings
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Create child documents
    child_documents = []
    for _, row in child_df.iterrows():
        metadata = row.get("metadata", {})

        if isinstance(metadata, str):
            try:
                metadata = ast.literal_eval(metadata)
            except:
                metadata = {}

        if not isinstance(metadata, dict):
            metadata = {}

        # Ensure conversation_id is in metadata
        if "conversation_id" not in metadata:
            metadata["conversation_id"] = row.get("conversation_id", "unknown")

        doc = Document(page_content=row["text"], metadata=metadata)
        child_documents.append(doc)

    # Create parent documents
    parent_documents = []
    parent_id_mapping = {}  # Map conversation_id to parent doc

    for _, row in parent_df.iterrows():
        metadata = row.get("metadata", {})

        if isinstance(metadata, str):
            try:
                metadata = ast.literal_eval(metadata)
            except:
                metadata = {}

        if not isinstance(metadata, dict):
            metadata = {}

        conv_id = row.get("conversation_id", metadata.get("conversation_id", "unknown"))

        doc = Document(page_content=row["text"], metadata=metadata)
        parent_documents.append(doc)
        parent_id_mapping[conv_id] = doc

    print(
        f"Created {len(child_documents)} child docs and {len(parent_documents)} parent docs"
    )

    # Initialize vector store
    if os.path.exists(DB_PATH):
        print(f"Clearing existing database at {DB_PATH}...")
        import shutil

        shutil.rmtree(DB_PATH)

    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Initialize parent document store
    os.makedirs(PARENT_STORE_PATH, exist_ok=True)
    parent_store = LocalFileStore(PARENT_STORE_PATH)

    # Store parent documents in the docstore
    print("Storing parent documents...")
    for conv_id, parent_doc in parent_id_mapping.items():
        # Serialize parent document to JSON string (LocalFileStore expects bytes-like)
        # Filter out non-JSON-serializable metadata (like arrays)
        clean_metadata = {}
        for k, v in parent_doc.metadata.items():
            if isinstance(v, (str, int, float, bool, list)) or v is None:
                clean_metadata[k] = v
            elif isinstance(v, list) and all(isinstance(x, int) for x in v):
                clean_metadata[k] = v

        parent_data = {
            "page_content": parent_doc.page_content,
            "metadata": clean_metadata,
        }
        parent_store.mset([(conv_id, json.dumps(parent_data).encode("utf-8"))])

    # Add child documents to vectorstore with parent references
    print("Indexing child documents...")
    batch_size = 1000
    total_docs = len(child_documents)

    for i in range(0, total_docs, batch_size):
        batch = child_documents[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Indexed {min(i+batch_size, total_docs)}/{total_docs} child documents")

    print(f"\nEmbedding complete!")
    print(f"- VectorStore: {DB_PATH}")
    print(f"- Parent Store: {PARENT_STORE_PATH}")
    print(f"- Child docs in vector DB: {vectorstore._collection.count()}")
    print(f"- Parent docs in store: {len(parent_id_mapping)}")

    # Create BM25 index for hybrid search
    print("\nCreating BM25 index for hybrid search...")
    from langchain_community.retrievers import BM25Retriever
    import pickle

    # Preprocess documents to handle encoding issues
    # BM25 can have issues with special unicode characters
    clean_child_documents = []
    for doc in child_documents:
        # Clean text to ASCII-safe characters
        clean_text = doc.page_content.encode("ascii", "ignore").decode("ascii")
        clean_doc = Document(page_content=clean_text, metadata=doc.metadata)
        clean_child_documents.append(clean_doc)

    bm25_retriever = BM25Retriever.from_documents(clean_child_documents)
    bm25_retriever.k = 10  # Default k for BM25

    # Save BM25 index
    bm25_path = "bm25_index.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"- BM25 Index: {bm25_path}")
    print("\n✅ All indexes created successfully!")


if __name__ == "__main__":
    embed_chunks()
