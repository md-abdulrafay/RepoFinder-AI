
import os
import re
import hashlib
import json
from backend.storage import load_documents
from rank_bm25 import BM25Okapi
import math
from datetime import datetime, timezone

from backend.vector_store import load_vector_store
from backend.document_processing import (
    load_repository_documents,
    split_documents
)


INDEX_PATH = "data/faiss_index"
DATA_PATH = "data/repositories.json"


# In-memory BM25 cache
_bm25_index = None
_bm25_chunks = None
_bm25_content_hash = None



def load_search_engine():

    vector_store = load_vector_store(INDEX_PATH)

    return vector_store


def tokenize(text):

    return re.findall(r"\b\w+\b", text.lower())


def get_bm25_index():
    global _bm25_index
    global _bm25_chunks
    global _bm25_content_hash

    repositories = load_documents(DATA_PATH)

    # Only fingerprint the content used by BM25
    content_data = [
        {
            "id": repo["id"],
            "content": repo["content"]
        }
        for repo in repositories
    ]

    current_hash = hashlib.sha256(
        json.dumps(
            content_data,
            sort_keys=True,
            ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()

    # Rebuild only if repository content actually changes
    if _bm25_index is None or current_hash != _bm25_content_hash:

        print("\nBuilding BM25 index...")

        documents = load_repository_documents(DATA_PATH)
        chunks = split_documents(documents)

        if not chunks:
            _bm25_index = None
            _bm25_chunks = []
            _bm25_content_hash = current_hash
            return None, []

        tokenized_chunks = [
            tokenize(chunk.page_content)
            for chunk in chunks
        ]

        _bm25_index = BM25Okapi(tokenized_chunks)
        _bm25_chunks = chunks
        _bm25_content_hash = current_hash

        print(f"BM25 indexed chunks: {len(chunks)}")

    return _bm25_index, _bm25_chunks

def get_popularity_score(stars):
    stars = max(0, int(stars or 0))

    # 10,000 stars is our fixed popularity reference.
    reference_stars = 10_000

    score = math.log1p(stars) / math.log1p(reference_stars)

    return min(score, 1.0)


def get_activity_score(pushed_at):
    if not pushed_at:
        return 0.5

    try:
        pushed_date = datetime.fromisoformat(
            pushed_at.replace("Z", "+00:00")
        )

        if pushed_date.tzinfo is None:
            pushed_date = pushed_date.replace(
                tzinfo=timezone.utc
            )

        age_days = (
            datetime.now(timezone.utc) - pushed_date
        ).days

        age_days = max(0, age_days)

        return math.exp(-age_days / 365)

    except (ValueError, TypeError, AttributeError):
        return 0.5


def hybrid_search(vector_store, query, k=5, return_diagnostics=False):

    # Latest repository metadata from JSON
    latest_repositories = {
        str(repo["id"]): repo
        for repo in load_documents(DATA_PATH)
    }

    # Retrieve a wider pool before final ranking
    candidate_k = max(k * 10, 50)

    # 1. FAISS semantic retrieval
    faiss_results = vector_store.similarity_search_with_score(
        query,
        k=candidate_k * 3
    )

    faiss_documents = []
    faiss_seen = set()

    for doc, score in faiss_results:
        repo_id = str(doc.metadata["id"])

        if repo_id not in faiss_seen:
            faiss_seen.add(repo_id)
            faiss_documents.append(doc)

        if len(faiss_documents) >= candidate_k:
            break

    # 2. BM25 keyword retrieval
    bm25, chunks = get_bm25_index()
    bm25_documents = []

    if bm25 is not None:
        bm25_scores = bm25.get_scores(tokenize(query))

        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )

        bm25_seen = set()

        for i in bm25_indices:
            if bm25_scores[i] <= 0:
                continue

            doc = chunks[i]
            repo_id = str(doc.metadata["id"])

            if repo_id not in bm25_seen:
                bm25_seen.add(repo_id)
                bm25_documents.append(doc)

            if len(bm25_documents) >= candidate_k:
                break

    # 3. Reciprocal Rank Fusion
    rrf_scores = {}
    repository_documents = {}

    def add_ranked_results(results):
        seen_repositories = set()

        for doc in results:
            repo_id = str(doc.metadata["id"])

            if repo_id in seen_repositories:
                continue

            seen_repositories.add(repo_id)
            rank = len(seen_repositories)

            rrf_scores[repo_id] = (
                rrf_scores.get(repo_id, 0)
                + 1 / (60 + rank)
            )

            repository_documents[repo_id] = doc

    add_ranked_results(faiss_documents)
    add_ranked_results(bm25_documents)

    # Refresh candidate metadata using the latest JSON records
    for repo_id, doc in repository_documents.items():

        latest = latest_repositories.get(repo_id)

        if latest:
            doc.metadata.update({
                "full_name": latest.get("full_name", ""),
                "description": latest.get("description", ""),
                "language": latest.get("language", ""),
                "stars": latest.get("stars", 0),
                "pushed_at": latest.get("pushed_at"),
                "archived": latest.get("archived", False),
                "url": latest.get("url", ""),
                "topics": latest.get("topics", [])
            })

    # 4. Calculate different ranking strategies
    max_rrf = max(rrf_scores.values(), default=1)

    ranking_scores = {
        "rrf_only": {},
        "rrf_popularity": {},
        "rrf_activity": {},
        "current": {},
        "rrf_activity_archive": {}
    }

    score_details = {}

    for repo_id, rrf_score in rrf_scores.items():

        doc = repository_documents[repo_id]
        metadata = doc.metadata

        relevance_score = rrf_score / max_rrf

        popularity_score = get_popularity_score(
            metadata.get("stars", 0)
        )

        activity_score = get_activity_score(
            metadata.get("pushed_at")
        )

        archived_penalty = (
            0.0 if metadata.get("archived", False) else 1.0
        )

        # Strategy 1: Pure RRF
        ranking_scores["rrf_only"][repo_id] = (
            relevance_score
        )

        ranking_scores["rrf_activity_archive"][repo_id] = (
            0.90 * relevance_score
            + 0.08 * activity_score
            + 0.02 * archived_penalty
        )

        # Strategy 2: RRF + popularity
        ranking_scores["rrf_popularity"][repo_id] = (
            0.90 * relevance_score
            + 0.10 * popularity_score
        )

        # Strategy 3: RRF + activity
        ranking_scores["rrf_activity"][repo_id] = (
            0.92 * relevance_score
            + 0.08 * activity_score
        )

        # Strategy 4: Current formula
        ranking_scores["current"][repo_id] = (
            0.80 * relevance_score
            + 0.10 * popularity_score
            + 0.08 * activity_score
            + 0.02 * archived_penalty
        )

        score_details[repo_id] = {
            "name": metadata.get("full_name", "Unknown"),
            "rrf_score": rrf_score,
            "relevance": relevance_score,
            "popularity": popularity_score,
            "activity": activity_score,
            "archived_score": archived_penalty,
            "final_score": ranking_scores["current"][repo_id]
        }

    # 5. Sort results using each strategy
    rrf_ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    ranked_strategies = {}

    for strategy, scores in ranking_scores.items():

        ranked_strategies[strategy] = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

    # Keep current strategy as the default search ranking
    final_ranked = ranked_strategies["current"]

    results = [
        (repository_documents[repo_id], score)
        for repo_id, score in final_ranked[:k]
    ]

    # 5. Sort results at each stage
    rrf_ranked = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    final_ranked = ranked_strategies["current"]

    results = [
        (repository_documents[repo_id], score)
        for repo_id, score in final_ranked[:k]
    ]

    if not return_diagnostics:
        return results

    def get_names(documents):
        return [
            doc.metadata.get("full_name", "Unknown")
            for doc in documents
        ]

    diagnostics = {
        "faiss": get_names(faiss_documents),

        "bm25": get_names(bm25_documents),

        "rrf": [
            repository_documents[repo_id].metadata.get(
                "full_name", "Unknown"
            )
            for repo_id, score in rrf_ranked
        ],

        "final": get_names([
            repository_documents[repo_id]
            for repo_id, score in final_ranked
        ]),

        "ranking_variants": {
            strategy: [
                repository_documents[repo_id].metadata.get(
                    "full_name", "Unknown"
                )
                for repo_id, score in ranked
            ]
            for strategy, ranked in ranked_strategies.items()
        },

        "score_details": [
            score_details[repo_id]
            for repo_id, score in final_ranked
        ]
    }

    return results, diagnostics