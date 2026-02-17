import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_dataset(input_path="data/cleaned_dataset.parquet"):
    print("Chunking dataset...")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_parquet(input_path)

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    chunked_data = []

    for _, row in df.iterrows():
        chunks = text_splitter.split_text(row["text"])
        for chunk in chunks:
            chunked_data.append({"text": chunk, "source": row["source"]})

    chunked_df = pd.DataFrame(chunked_data)

    output_path = "data/chunked_dataset.parquet"
    chunked_df.to_parquet(output_path)
    print(f"Chunked dataset saved to {output_path} with {len(chunked_df)} chunks")
    return output_path


if __name__ == "__main__":
    chunk_dataset()
