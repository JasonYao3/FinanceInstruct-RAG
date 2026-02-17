import pandas as pd
import os
import hashlib
import json


def clean_dataset(input_path="data/raw_dataset.parquet"):
    print("Cleaning dataset...")
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return

    df = pd.read_parquet(input_path)

    # Filter out empty rows
    df = df.dropna(subset=["user", "assistant"])

    # Group by conversation ID to create parent documents
    conversations = {}
    child_docs = []

    for idx, row in df.iterrows():
        # Generate conversation ID
        conv_key = str(row.get("system", "")) + str(row["user"])[:100]
        conv_id = hashlib.md5(conv_key.encode()).hexdigest()[:12]

        # Create child document (individual Q&A)
        child_metadata = {
            "source": "Finance-Instruct-500k",
            "row_id": idx,
            "user_question": row["user"],
            "assistant_answer": row["assistant"],
            "conversation_id": conv_id,
            "doc_type": "child",
        }

        if "system" in row and pd.notna(row["system"]):
            child_metadata["system_prompt"] = row["system"]
        else:
            child_metadata["system_prompt"] = None

        child_content = f"Q: {row['user']}\nA: {row['assistant']}"

        child_docs.append(
            {
                "text": child_content,
                "metadata": child_metadata,
                "conversation_id": conv_id,
            }
        )

        # Group for parent documents
        if conv_id not in conversations:
            conversations[conv_id] = {
                "system": row.get("system", ""),
                "turns": [],
                "row_ids": [],
            }

        conversations[conv_id]["turns"].append(
            {"user": row["user"], "assistant": row["assistant"]}
        )
        conversations[conv_id]["row_ids"].append(idx)

    # Create parent documents (full conversations)
    parent_docs = []
    for conv_id, conv_data in conversations.items():
        # Build full conversation text
        turns_text = []
        for i, turn in enumerate(conv_data["turns"], 1):
            turns_text.append(f"Turn {i}:\nQ: {turn['user']}\nA: {turn['assistant']}")

        parent_content = "\n\n".join(turns_text)

        parent_metadata = {
            "source": "Finance-Instruct-500k",
            "conversation_id": conv_id,
            "doc_type": "parent",
            "num_turns": len(conv_data["turns"]),
            "row_ids": conv_data["row_ids"],
        }

        if conv_data["system"]:
            parent_metadata["system_prompt"] = conv_data["system"]

        parent_docs.append(
            {
                "text": parent_content,
                "metadata": parent_metadata,
                "conversation_id": conv_id,
            }
        )

    # Save both child and parent documents
    child_df = pd.DataFrame(child_docs)
    parent_df = pd.DataFrame(parent_docs)

    child_output = "data/cleaned_dataset.parquet"
    parent_output = "data/parent_documents.parquet"

    child_df.to_parquet(child_output)
    parent_df.to_parquet(parent_output)

    print(f"Child documents saved to {child_output} ({len(child_df)} docs)")
    print(f"Parent documents saved to {parent_output} ({len(parent_df)} conversations)")
    print(f"Average turns per conversation: {len(child_df) / len(parent_df):.2f}")

    return child_output, parent_output


if __name__ == "__main__":
    clean_dataset()
