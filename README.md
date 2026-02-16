# Finance RAG

A Retrieval-Augmented Generation (RAG) system specialized in finance, using the [Finance-Instruct-500k](https://huggingface.co/datasets/Josephgflowers/Finance-Instruct-500k) dataset.

## Setup

1.  **Create a virtual environment**:

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up environment variables**:
    Create a `.env` file with your API keys (e.g., `GROQ_API_KEY` or `OPENAI_API_KEY`).

## Project Structure

- `ingest.py`: Script to load data and populate the vector database.
- `app.py`: Streamlit application for the user interface.
- `explore_data.py`: Script to inspect the dataset structure.
