import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain.storage import LocalFileStore
from langchain.memory import ConversationBufferMemory
import pickle
import json

# Load environment variables
load_dotenv()

# Configuration
DB_PATH = "./chroma_db"
PARENT_STORE_PATH = "./parent_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

st.set_page_config(page_title="Finance Instruct RAG", page_icon="💰")


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
    if not os.path.exists(PARENT_STORE_PATH):
        return None

    parent_store = LocalFileStore(PARENT_STORE_PATH)
    return parent_store


def main():
    st.title("💰 Finance Instruct RAG")
    st.caption(
        "A financial assistant that uses the Finance-Instruct-500k dataset to answer questions with full conversation context."
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

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()

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
        bm25_retriever = get_bm25_retriever()
        if bm25_retriever:
            st.sidebar.success("✅ Hybrid search enabled")
            semantic_retriever = vectorstore.as_retriever(
                search_type="mmr", search_kwargs={"k": k_retrieval * 3}
            )
            bm25_retriever.k = k_retrieval * 3
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

    # --- CHAT HISTORY SETUP ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Store history for LangChain
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Conversational RAG Chain
    # 1. Condense Question Prompt (Generator)
    condense_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
    
    Chat History:
    {chat_history}
    
    Follow Up Input: {question}
    
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(condense_template)

    # 2. Answer Prompt (Reader)
    qa_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and relevant to the financial context.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate.from_template(qa_template)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        verbose=True,
    )

    # User Input
    if prompt := st.chat_input("Ask a financial question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run Chain
                    response = qa_chain.invoke(
                        {
                            "question": prompt,
                            "chat_history": st.session_state.chat_history,
                        }
                    )

                    answer = response["answer"]
                    st.markdown(answer)

                    # Store interaction in history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    # Keep last 5 turns to manage token limit
                    st.session_state.chat_history.append((prompt, answer))
                    if len(st.session_state.chat_history) > 5:
                        st.session_state.chat_history.pop(0)

                    # Display Context in Expander (optional, to keep chat clean)
                    with st.expander("View Retrieved Context & Metadata"):
                        for i, doc in enumerate(response["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.info(doc.page_content)

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
                                        parent_data = json.loads(
                                            parent_data_list[0].decode("utf-8")
                                        )
                                        st.markdown("**📝 Full Conversation Context:**")
                                        st.text_area(
                                            f"Conversation {conv_id}",
                                            parent_data["page_content"],
                                            height=200,
                                            key=f"conv_{i}_{len(st.session_state.messages)}",
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
