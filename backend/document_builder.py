def build_repository_document(repo):
    topics = repo.get("topics", [])

    content = f"""
    Repository: {repo.get('full_name', '')}
    Description: {repo.get('description') or ''}
    Language: {repo.get('language') or ''}
    Topics: {', '.join(topics)}

    README:
    {repo.get('clean_readme', '')}
    """

    return {
        "id": repo["id"],
        "full_name": repo.get("full_name", ""),
        "description": repo.get("description") or "",
        "language": repo.get("language") or "",
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "pushed_at": repo.get("pushed_at"),
        "archived": repo.get("archived", False),
        "url": repo.get("html_url", ""),
        "topics": topics,
        "content": content.strip()
    }