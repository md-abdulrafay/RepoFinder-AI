import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_repository_documents(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        repositories = json.load(file)

    documents = []

    for repo in repositories:

        document = Document(
            page_content=repo["content"],
            metadata={
                "id": repo["id"],
                "full_name": repo["full_name"],
                "description": repo["description"],
                "language": repo["language"],
                "stars": repo["stars"],
                "pushed_at": repo.get("pushed_at"),
                "archived": repo.get("archived", False),
                "url": repo["url"],
                "topics": repo["topics"]
            }
        )

        documents.append(document)

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return chunks