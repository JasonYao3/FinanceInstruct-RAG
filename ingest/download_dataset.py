import os
from datasets import load_dataset
import pandas as pd
import argparse


def download_dataset(limit=None):
    print("Downloading dataset...")
    # Load dataset from Hugging Face
    ds = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")

    if limit:
        print(f"Limiting to {limit} examples")
        ds = ds.select(range(limit))

    # Convert to pandas for easier handling in next steps (or just save to disk)
    df = ds.to_pandas()

    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)

    output_path = "data/raw_dataset.parquet"
    df.to_parquet(output_path)
    print(f"Dataset saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of examples")
    args = parser.parse_args()

    download_dataset(limit=args.limit)
