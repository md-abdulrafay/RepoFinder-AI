
import os
import json
import hashlib
import uuid
from backend.vector_store import (
    create_vector_store,
    load_vector_store,
    save_vector_store
)
from backend.github_services import (
    search_repositories,
    get_repository_readme
)
from backend.text_processing import clean_readme
from backend.document_builder import build_repository_document
from backend.storage import save_documents, load_documents
from backend.document_processing import (
    load_repository_documents,
    split_documents
)


DATA_PATH = "data/repositories.json"
INDEX_PATH = "data/faiss_index"
MANIFEST_PATH = "data/index_manifest.json"


def search_github(queries, pages=1):

    repositories = []
    seen_ids = set()

    for query in queries:

        for page in range(1, pages + 1):

            results = search_repositories(query, page)

            for repo in results:

                if repo["id"] not in seen_ids:
                    seen_ids.add(repo["id"])
                    repositories.append(repo)

    return repositories


def get_document_hash(document):

    content = json.dumps(
        document,
        sort_keys=True,
        ensure_ascii=False
    )

    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def get_semantic_hash(document):

    semantic_fields = {
        "full_name": document.get("full_name"),
        "description": document.get("description"),
        "language": document.get("language"),
        "topics": document.get("topics"),
        "content": document.get("content")
    }

    return get_document_hash(semantic_fields)


def create_manifest_from_index(vector_store):

    manifest = {}

    for vector_id in vector_store.index_to_docstore_id.values():

        doc = vector_store.docstore.search(vector_id)

        if not hasattr(doc, "metadata"):
            continue

        repo_id = str(doc.metadata["id"])

        manifest.setdefault(repo_id, []).append(vector_id)

    return manifest


def load_manifest(vector_store):

    if os.path.exists(MANIFEST_PATH):

        with open(MANIFEST_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    print("Creating manifest from existing FAISS index...")

    return create_manifest_from_index(vector_store)


def save_manifest(manifest):

    os.makedirs("data", exist_ok=True)

    with open(MANIFEST_PATH, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)


def build_index(queries, pages=1):

    print("\nSearching GitHub...")

    repositories = search_github(queries, pages)

    print(f"Discovered repositories: {len(repositories)}")

    new_documents = []

    for repo in repositories:

        owner, repo_name = repo["full_name"].split("/")

        readme = get_repository_readme(owner, repo_name)

        repo["clean_readme"] = clean_readme(readme)

        document = build_repository_document(repo)

        document["content_hash"] = get_semantic_hash(document)

        new_documents.append(document)

    # Load existing dataset
    try:
        existing_documents = load_documents(DATA_PATH)
    except FileNotFoundError:
        existing_documents = []

    # Merge repositories using GitHub ID
    repositories_by_id = {
        str(repo["id"]): repo
        for repo in existing_documents
    }

    changed_ids = set()
    dataset_changed = False

    for document in new_documents:

        repo_id = str(document["id"])

        old_document = repositories_by_id.get(repo_id)

        if old_document:
            old_hash = get_semantic_hash(old_document)
        else:
            old_hash = None

        if old_hash != document["content_hash"]:
            changed_ids.add(repo_id)

        if old_document != document:
            dataset_changed = True


        repositories_by_id[repo_id] = document

    all_documents = list(repositories_by_id.values())

    print(f"Total repositories in dataset: {len(all_documents)}")
    print(f"New or changed repositories: {len(changed_ids)}")

    if dataset_changed or not os.path.exists(DATA_PATH):

        save_documents(all_documents, DATA_PATH)

    else:

        print("No dataset changes. Skipping JSON save.")

    # First-time index creation
    if not os.path.exists(INDEX_PATH):

        print("\nCreating initial FAISS index...")

        documents = load_repository_documents(DATA_PATH)
        chunks = split_documents(documents)

        vector_store = create_vector_store(chunks, INDEX_PATH)

        manifest = create_manifest_from_index(vector_store)

        save_manifest(manifest)

        print(f"Total chunks: {len(chunks)}")
        print("Initial indexing completed!")

        return

    # Load existing FAISS index
    vector_store = load_vector_store(INDEX_PATH)

    manifest = load_manifest(vector_store)

    # Remove old vectors for changed repositories
    ids_to_delete = []

    for repo_id in changed_ids:

        old_vector_ids = manifest.get(repo_id, [])

        ids_to_delete.extend(old_vector_ids)

        manifest[repo_id] = []

    if ids_to_delete:

        vector_store.delete(ids=ids_to_delete)

        print(f"Removed old vectors: {len(ids_to_delete)}")

    # Create chunks only for new or changed repositories
    if changed_ids:

        all_langchain_documents = load_repository_documents(DATA_PATH)

        changed_documents = [
            doc for doc in all_langchain_documents
            if str(doc.metadata["id"]) in changed_ids
        ]

        chunks = split_documents(changed_documents)

        # Generate unique IDs for the new chunks
        chunk_ids = [
            str(uuid.uuid4())
            for _ in chunks
        ]

        if chunks:

            vector_store.add_documents(
                documents=chunks,
                ids=chunk_ids
            )

            for chunk, chunk_id in zip(chunks, chunk_ids):

                repo_id = str(chunk.metadata["id"])

                manifest.setdefault(repo_id, []).append(chunk_id)

        print(f"New chunks added: {len(chunks)}")

    else:

        print("No new or changed repositories. No embeddings needed.")

    # Save updated FAISS index and manifest
    save_vector_store(vector_store, INDEX_PATH)
    save_manifest(manifest)

    print("\nIncremental indexing completed!")



def add_live_repositories(repositories, vector_store):

    # Load existing repository dataset
    try:
        existing_documents = load_documents(DATA_PATH)
    except FileNotFoundError:
        existing_documents = []

    repositories_by_id = {
        str(repo["id"]): repo
        for repo in existing_documents
    }

    changed_ids = set()
    dataset_changed = False
    updated_documents = []

    print("\nChecking discovered repositories...")

    for repo in repositories:

        repo_id = str(repo["id"])

        old_document = repositories_by_id.get(repo_id)

        # Fetch latest README
        owner, repo_name = repo["full_name"].split("/")

        readme = get_repository_readme(owner, repo_name)

        repo["clean_readme"] = clean_readme(readme)

        # Build updated repository document
        document = build_repository_document(repo)

        document["content_hash"] = get_semantic_hash(document)

        # Compare semantic content, not metadata like stars
        if old_document:

            old_hash = get_semantic_hash(old_document)

            if old_hash != document["content_hash"]:
                changed_ids.add(repo_id)

            if old_document != document:
                dataset_changed = True

                changed_fields = [
                    key for key in document
                    if old_document.get(key) != document.get(key)
                ]

                print(f"Metadata/content changes in {repo['full_name']}: {changed_fields}")

        else:
            changed_ids.add(repo_id)
            dataset_changed = True

        # Always refresh metadata in JSON
        repositories_by_id[repo_id] = document

        updated_documents.append(document)

    print(f"Repositories checked: {len(updated_documents)}")
    print(f"New or changed repositories: {len(changed_ids)}")

    # Save dataset only when something actually changed
    if dataset_changed:

        all_documents = list(repositories_by_id.values())
        save_documents(all_documents, DATA_PATH)

    else:

        print("No dataset changes. Skipping JSON save.")

    if not changed_ids:

        print("All discovered repositories are up to date.")

        return vector_store

    # Load manifest
    manifest = load_manifest(vector_store)

    # Delete old FAISS chunks for changed repositories
    ids_to_delete = []

    for repo_id in changed_ids:

        old_vector_ids = manifest.get(repo_id, [])

        ids_to_delete.extend(old_vector_ids)

        manifest[repo_id] = []

    if ids_to_delete:

        vector_store.delete(ids=ids_to_delete)

        print(f"Removed old chunks: {len(ids_to_delete)}")

    # Convert only changed repositories into LangChain documents
    documents = load_repository_documents(DATA_PATH)

    changed_documents = [
        doc for doc in documents
        if str(doc.metadata["id"]) in changed_ids
    ]

    # Split changed documents
    chunks = split_documents(changed_documents)

    print(f"New chunks created: {len(chunks)}")

    # Add updated chunks to FAISS
    chunk_ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    if chunks:

        vector_store.add_documents(
            documents=chunks,
            ids=chunk_ids
        )

        for chunk, chunk_id in zip(chunks, chunk_ids):

            repo_id = str(chunk.metadata["id"])

            manifest.setdefault(repo_id, []).append(chunk_id)

    # Save updated FAISS and manifest
    save_vector_store(vector_store, INDEX_PATH)

    save_manifest(manifest)

    print("Live repository synchronization completed!")

    return vector_store

