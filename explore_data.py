from datasets import load_dataset


def explore():
    print("Loading dataset metadata...")
    # Load streaming to avoid downloading the whole thing just for a peek
    dataset = load_dataset(
        "Josephgflowers/Finance-Instruct-500k", split="train", streaming=True
    )

    print("\n--- First 3 Examples ---")
    for i, example in enumerate(dataset.take(3)):
        print(f"\nExample {i+1}:")
        print(f"System: {example.get('system', '')[:100]}...")
        print(f"User: {example.get('user', '')[:100]}...")
        print(f"Assistant: {example.get('assistant', '')[:100]}...")
        print("-" * 50)


if __name__ == "__main__":
    explore()
