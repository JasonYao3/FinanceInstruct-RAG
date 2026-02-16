import streamlit as st

# from langchain_community.vectorstores import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings

# Configuration
DB_PATH = "./chroma_db"


def main():
    st.title("Finance RAG Assistant")

    # TODO: 1. Initialize Embeddings (same model as ingest.py)

    # TODO: 2. Load VectorStore
    # db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)

    # TODO: 3. Create Retriever
    # retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

    user_query = st.text_input("Ask a financial question:")

    if user_query:
        # TODO: 4. Perform Retrieval
        # docs = retriever.invoke(user_query)

        # TODO: 5. Generate Answer (Mock for now, or connect LLM)
        st.write("Retrieved Context:")
        # for doc in docs:
        #     st.info(doc.page_content)

        st.write("Answer:")
        st.write("(Connect LLM here)")


if __name__ == "__main__":
    main()
