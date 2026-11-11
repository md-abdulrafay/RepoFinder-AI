import json
import os


def save_documents(documents, file_path="data/repositories.json"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(documents, file, indent=4, ensure_ascii=False)

    print(f"Saved {len(documents)} documents to {file_path}")


def load_documents(file_path="data/repositories.json"):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)