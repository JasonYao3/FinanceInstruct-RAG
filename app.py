import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
import pickle

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = "./chroma_db"
PARENT_STORE_PATH = "./parent_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Finance RAG Assistant", page_icon="💰")


def get_vectorstore():
    if not os.path.exists(DB_PATH):
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    return vectorstore


def get_bm25_retriever():
    """Load BM25 retriever from disk"""
    bm25_path = "bm25_index.pkl"
    if not os.path.exists(bm25_path):
        return None

    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    return bm25_retriever


def get_parent_retriever():
    """Load parent document store"""
    from langchain.storage import LocalFileStore

    if not os.path.exists(PARENT_STORE_PATH):
        return None

    parent_store = LocalFileStore(PARENT_STORE_PATH)
    return parent_store


def main():
    st.title("💰 Finance RAG Assistant")
    st.markdown(
        "Ask questions about finance concepts and get answers based on the Finance-Instruct-500k dataset."
    )

    # Sidebar for API Key and Filters
    with st.sidebar:
        groq_api_key = st.text_input(
            "Groq API Key", type="password", help="Get one for free at console.groq.com"
        )
        if not groq_api_key:
            groq_api_key = os.getenv("GROQ_API_KEY")

        st.markdown("---")
        st.markdown("### Retrieval Settings")
        k_retrieval = st.slider(
            "Documents to Retrieve", min_value=1, max_value=10, value=3
        )

        st.markdown("### Retrieval Mode")
        use_hybrid = st.checkbox(
            "Enable Hybrid Search (BM25 + Semantic)",
            value=True,
            help="Combines keyword-based (BM25) and semantic search for better results",
        )

        if use_hybrid:
            semantic_weight = st.slider(
                "Semantic Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Higher = more semantic, Lower = more keyword-based",
            )

        st.markdown("### Context Settings")
        use_full_context = st.checkbox(
            "Show full conversation context",
            value=True,
            help="When enabled, displays the entire conversation thread for retrieved Q&A pairs",
        )

    if not groq_api_key:
        st.warning(
            "Please enter your Groq API Key in the sidebar or .env file to continue."
        )
        st.stop()

    # Initialize components
    vectorstore = get_vectorstore()

    if not vectorstore:
        st.error(
            f"Vector Database not found at {DB_PATH}. Please run the ingestion pipeline first."
        )
        st.stop()

    # Get parent store for full conversation retrieval
    parent_store = get_parent_retriever()

    # Create retriever based on mode
    if use_hybrid:
        # Load BM25 retriever
        bm25_retriever = get_bm25_retriever()

        if bm25_retriever:
            st.sidebar.success("✅ Hybrid search enabled")

            # Semantic retriever with MMR
            semantic_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": k_retrieval * 3},  # Get more for fusion
            )

            # Set BM25 k
            bm25_retriever.k = k_retrieval * 3

            # Combine with EnsembleRetriever
            bm25_weight = 1.0 - semantic_weight
            retriever = EnsembleRetriever(
                retrievers=[semantic_retriever, bm25_retriever],
                weights=[semantic_weight, bm25_weight],
            )
        else:
            st.sidebar.warning("⚠️ BM25 index not found. Using semantic only.")
            retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": k_retrieval}
            )
    else:
        st.sidebar.info("ℹ️ Using semantic search only")
        retriever = vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k_retrieval}
        )

    # Use Groq LLaMA 3.3
    llm = ChatGroq(
        groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.1
    )

    # Create Chain
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and relevant to the financial context.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    # User Input
    user_query = st.text_input(
        "Ask a financial question:",
        placeholder="e.g., What is the difference between stocks and bonds?",
    )

    if user_query:
        with st.spinner("Analyzing financial data..."):
            try:
                response = qa_chain.invoke({"query": user_query})

                st.markdown("### Answer")
                st.write(response["result"])

                with st.expander("View Retrieved Context & Metadata"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.markdown(f"**Source {i+1}:**")

                        # Display child document (what matched)
                        st.info(doc.page_content)

                        # Display metadata
                        metadata = doc.metadata
                        conv_id = metadata.get("conversation_id", None)

                        if metadata:
                            st.caption("**Metadata:**")
                            cols = st.columns(2)

                            if "conversation_id" in metadata:
                                cols[0].metric(
                                    "Conversation ID", metadata["conversation_id"]
                                )
                            if (
                                "system_prompt" in metadata
                                and metadata["system_prompt"]
                            ):
                                cols[1].caption(
                                    f"System: {metadata['system_prompt'][:50]}..."
                                )
                            if "row_id" in metadata:
                                cols[0].caption(f"Row ID: {metadata['row_id']}")
                            if "source" in metadata:
                                cols[1].caption(f"Source: {metadata['source']}")

                        # Fetch and display full conversation if enabled
                        if use_full_context and parent_store and conv_id:
                            try:
                                parent_data_list = parent_store.mget([conv_id])
                                if parent_data_list and parent_data_list[0]:
                                    # Deserialize from JSON
                                    import json

                                    parent_data = json.loads(
                                        parent_data_list[0].decode("utf-8")
                                    )

                                    st.markdown("**📝 Full Conversation Context:**")
                                    st.text_area(
                                        f"Conversation {conv_id}",
                                        parent_data["page_content"],
                                        height=200,
                                        key=f"conv_{i}",
                                    )
                                    st.caption(
                                        f"Turns: {parent_data['metadata'].get('num_turns', 'N/A')}"
                                    )
                            except Exception as e:
                                st.caption(f"Could not load full conversation: {e}")

                        st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
