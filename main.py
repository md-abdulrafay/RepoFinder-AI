
from backend.search import load_search_engine
from backend.live_search import live_search
from backend.query_understanding import understand_query


# Load existing FAISS index
vector_store = load_search_engine()


# Get natural-language request
user_query = input("\nWhat are you looking for? ")

# Understand the query using Gemini or local Qwen
query_data = understand_query(user_query)

if query_data:
    semantic_query = query_data["semantic_query"]
    github_query = query_data["github_query"]

    print(f"\nLLM provider: {query_data['provider']}")
    print(f"Semantic query: {semantic_query}")
    print(f"GitHub query: {github_query}")

else:
    # Preserve manual search if both LLMs fail
    print("\nUsing manual search mode.")

    semantic_query = user_query
    github_query = input("GitHub search query: ")


# Run existing hybrid search
results = live_search(
    vector_store=vector_store,
    semantic_query=semantic_query,
    github_query=github_query,
    k=5,
    pages=1
)


print("\nFinal Search Results:\n")

for doc, score in results:

    print(f"Repository: {doc.metadata['full_name']}")
    print(f"Description: {doc.metadata['description']}")
    print(f"Stars: {doc.metadata['stars']}")
    print(f"URL: {doc.metadata['url']}")
    print(f"Hybrid score: {score}")

    print("-" * 50)
