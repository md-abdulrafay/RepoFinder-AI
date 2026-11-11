
from backend.github_services import (
    search_repositories,
    GitHubAPIError
)

from backend.indexing import add_live_repositories
from backend.search import hybrid_search

from backend.requirement_evaluation import (
    evaluate_repositories
)


def live_search(
    vector_store,
    semantic_query,
    github_query,
    requirements=None,
    k=5,
    pages=1,
    per_page=10
):

    requirements = requirements or []

    print("\nSearching GitHub live...")

    live_repositories = []
    seen_ids = set()

    try:

        for page in range(1, pages + 1):

            results = search_repositories(
                github_query,
                page,
                per_page=per_page
            )

            for repo in results:

                if repo["id"] not in seen_ids:
                    seen_ids.add(repo["id"])
                    live_repositories.append(repo)

    except GitHubAPIError as error:

        print(f"\nLive search failed: {error}")
        print("Continuing with the existing repository database.")

    print(
        f"GitHub repositories discovered: "
        f"{len(live_repositories)}"
    )

    if live_repositories:

        vector_store = add_live_repositories(
            live_repositories,
            vector_store
        )

    print("\nSearching updated database...")

    # Retrieve more candidates than the number displayed.
    candidate_count = min(max(k, 15), 20)

    candidate_results = hybrid_search(
        vector_store,
        semantic_query,
        k=candidate_count
    )

    if not candidate_results:
        return []

    if not requirements:
        return candidate_results[:k]

    candidate_documents = [
        doc for doc, score in candidate_results
    ]

    try:

        print("\nEvaluating repository requirements...")

        candidate_documents = evaluate_repositories(
            requirements,
            candidate_documents
        )

        # Preserve original hybrid scores.
        original_scores = {
            str(doc.metadata["id"]): score
            for doc, score in candidate_results
        }

        # Identify high-importance requirements.
        high_importance_ids = {
            requirement["id"]
            for requirement in requirements
            if requirement.get("importance") == "high"
        }

        eligible_results = []
        excluded_count = 0

        for doc in candidate_documents:

            repo_id = str(doc.metadata["id"])

            hybrid_score = original_scores.get(repo_id)

            if hybrid_score is None:
                continue

            checks = doc.metadata.get(
                "requirement_checks",
                {}
            )

            # Exclude repositories that explicitly fail
            # any high-importance requirement.
            failed_high_requirements = [
                requirement_id
                for requirement_id in high_importance_ids
                if checks.get(
                    requirement_id,
                    {}
                ).get("status", "unknown") == "mismatch"
            ]

            if failed_high_requirements:

                excluded_count += 1

                doc.metadata["excluded_requirements"] = (
                    failed_high_requirements
                )

                print(
                    f"Excluded {doc.metadata.get('full_name', repo_id)}: "
                    f"failed high-importance requirements "
                    f"{failed_high_requirements}"
                )

                continue

            requirement_score = doc.metadata.get(
                "requirement_score",
                0.5
            )

            final_score = (
                0.65 * hybrid_score
                + 0.35 * requirement_score
            )

            doc.metadata["hybrid_score"] = hybrid_score
            doc.metadata["final_score"] = final_score

            eligible_results.append((doc, final_score))

        eligible_results.sort(
            key=lambda item: item[1],
            reverse=True
        )

        print(
            f"\nExcluded repositories: {excluded_count}"
        )

        print(
            f"Eligible repositories: {len(eligible_results)}"
        )

        print("\nRequirement-aware ranking completed.")

        return eligible_results[:k]

    except Exception as error:

        print(
            f"\nRequirement evaluation failed: {error}"
        )

        print("Using original hybrid ranking instead.")

        return candidate_results[:k]
