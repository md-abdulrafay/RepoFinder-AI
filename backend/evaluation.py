
import json
from pathlib import Path

from backend.search import (
    load_search_engine,
    hybrid_search
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "evaluation_queries.json"

K = 5


def normalize_name(name):
    return name.strip().lower()

def calculate_metrics(names, relevant, k=5):

    normalized_names = [
        normalize_name(name)
        for name in names[:k]
    ]

    hits = set(normalized_names).intersection(relevant)

    precision = len(hits) / k if k else 0

    recall = len(hits) / len(relevant) if relevant else 0

    mrr = 0

    for rank, name in enumerate(normalized_names, start=1):
        if name in relevant:
            mrr = 1 / rank
            break

    return {
        "precision": precision,
        "recall": recall,
        "mrr": mrr
    }


def evaluate_query(vector_store, test_case, k=5):

    query = test_case["query"]

    relevant = {
        normalize_name(name)
        for name in test_case["relevant"]
    }

    # Get search results and diagnostics
    results, diagnostics = hybrid_search(
        vector_store,
        query,
        k=k,
        return_diagnostics=True
    )

    returned_names = [
        doc.metadata.get("full_name", "")
        for doc, score in results
    ]

    normalized_results = [
        normalize_name(name)
        for name in returned_names
    ]

    # Detect duplicate repositories
    unique_results = set(normalized_results)
    duplicates = len(normalized_results) - len(unique_results)

    # Calculate relevant results
    hits = unique_results.intersection(relevant)

    precision = len(hits) / k if k else 0

    recall = len(hits) / len(relevant) if relevant else 0

    # Calculate reciprocal rank
    reciprocal_rank = 0

    for rank, name in enumerate(normalized_results, start=1):
        if name in relevant:
            reciprocal_rank = 1 / rank
            break

    # Calculate candidate recall before final ranking
    retrieved_names = {
        normalize_name(name)
        for name in diagnostics["rrf"]
    }

    candidate_hits = retrieved_names.intersection(relevant)

    candidate_recall = (
        len(candidate_hits) / len(relevant)
        if relevant else 0
    )

    # Evaluate all ranking strategies
    strategy_metrics = {}

    for strategy, names in diagnostics["ranking_variants"].items():

        strategy_metrics[strategy] = calculate_metrics(
            names,
            relevant,
            k
        )

    return {
        "query": query,
        "results": returned_names,
        "precision": precision,
        "recall": recall,
        "mrr": reciprocal_rank,
        "duplicates": duplicates,
        "hits": len(hits),
        "diagnostics": diagnostics,
        "candidate_recall": candidate_recall,
        "strategy_metrics": strategy_metrics
    }


def run_evaluation():

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    print("\nLoading FAISS search engine...")
    vector_store = load_search_engine()

    all_results = []

    print("\nStarting evaluation...")

    for index, test_case in enumerate(test_cases, start=1):

        print("\n" + "=" * 65)
        print(f"TEST {index}: {test_case['query']}")

        result = evaluate_query(
            vector_store,
            test_case,
            K
        )

        all_results.append(result)

        # Final search results
        print("\nFINAL RESULTS:")

        for rank, name in enumerate(result["results"], start=1):
            print(f"{rank}. {name}")

        print(f"\nRelevant results: {result['hits']}")
        print(f"Precision@{K}: {result['precision']:.2f}")
        print(f"Recall@{K}: {result['recall']:.2f}")
        print(
              f"Candidate Recall: "
              f"{result['candidate_recall']:.2f}"
              )
        print(f"MRR@{K}: {result['mrr']:.2f}")
        print(f"Duplicates: {result['duplicates']}")

        # Retrieval diagnostics
        diagnostics = result["diagnostics"]

        print("\n--- RETRIEVAL DIAGNOSTICS ---")

        for stage in ["faiss", "bm25", "rrf", "final"]:

            print(f"\n{stage.upper()}:")

            for rank, name in enumerate(
                diagnostics[stage][:5],
                start=1
            ):
                print(f"{rank}. {name}")

        # Find relevant repositories that were missed
        relevant = {
            normalize_name(name)
            for name in test_case["relevant"]
        }

        returned = {
            normalize_name(name)
            for name in result["results"]
        }

        retrieved = {
            normalize_name(name)
            for name in diagnostics["rrf"]
        }

        missed = relevant - returned
        never_retrieved = relevant - retrieved

        print("\nRelevant repositories missing from final top 5:")
        print(missed or "None")

        print("\nRelevant repositories never retrieved by FAISS or BM25:")
        print(never_retrieved or "None")

        print("\n--- RELEVANT REPOSITORY SCORE BREAKDOWN ---")

        relevant = {
            normalize_name(name)
            for name in test_case["relevant"]
        }

        for item in diagnostics["score_details"]:
            if normalize_name(item["name"]) in relevant:
                print(f"\nRepository: {item['name']}")
                print(f"RRF score: {item['rrf_score']:.5f}")
                print(f"Relevance: {item['relevance']:.3f}")
                print(f"Popularity: {item['popularity']:.3f}")
                print(f"Activity: {item['activity']:.3f}")
                print(f"Not archived: {item['archived_score']:.1f}")
                print(f"Final score: {item['final_score']:.3f}")

    # Overall evaluation
    total = len(all_results)
    

    if total == 0:
        print("\nNo evaluation queries found.")
        return

    avg_precision = sum(
        r["precision"] for r in all_results
    ) / total

    avg_recall = sum(
        r["recall"] for r in all_results
    ) / total

    avg_mrr = sum(
        r["mrr"] for r in all_results
    ) / total

    total_duplicates = sum(
        r["duplicates"] for r in all_results
    )

    strategies = [
    "rrf_only",
    "rrf_popularity",
    "rrf_activity",
    "rrf_activity_archive",
    "current"
    ]

    print("\n" + "=" * 75)
    print("RANKING STRATEGY COMPARISON")
    print("=" * 75)

    print(
        f"{'Strategy':<20}"
        f"{'Precision@5':<15}"
        f"{'Recall@5':<15}"
        f"{'MRR@5':<15}"
    )

    for strategy in strategies:

        avg_p = sum(
            r["strategy_metrics"][strategy]["precision"]
            for r in all_results
        ) / total

        avg_r = sum(
            r["strategy_metrics"][strategy]["recall"]
            for r in all_results
        ) / total

        avg_mrr = sum(
            r["strategy_metrics"][strategy]["mrr"]
            for r in all_results
        ) / total

        print(
            f"{strategy:<20}"
            f"{avg_p:<15.3f}"
            f"{avg_r:<15.3f}"
            f"{avg_mrr:<15.3f}"
        )

    avg_candidate_recall = sum(
        r["candidate_recall"] for r in all_results) / total

    print("\n" + "=" * 65)
    print("FINAL EVALUATION REPORT")
    print("=" * 65)
    
    print(f"Total queries: {total}")
    print(f"Precision@{K}: {avg_precision:.3f}")
    print(f"Recall@{K}: {avg_recall:.3f}")
    print(f"MRR@{K}: {avg_mrr:.3f}")
    print(f"Total duplicates: {total_duplicates}")
    print(f"Candidate Recall: {avg_candidate_recall:.3f}")

    print("\nEvaluation completed.")


if __name__ == "__main__":
    run_evaluation()
