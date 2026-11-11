import requests, base64, os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}


class GitHubAPIError(Exception):
    pass


def search_repositories(query, page, per_page=10):

    url = "https://api.github.com/search/repositories"

    params = {
        "q": query,
        "per_page": per_page,
        "page": page
    }

    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=15
        )

    except requests.RequestException as error:
        raise GitHubAPIError(
            f"Could not connect to GitHub: {error}"
        ) from error

    if response.status_code != 200:

        if response.status_code in (403, 429):
            message = "GitHub rate limit or access restriction."
        else:
            message = f"GitHub returned HTTP {response.status_code}."

        raise GitHubAPIError(message)

    return response.json()["items"]

def get_repository_readme(owner, repo_name):

    url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"

    response = requests.get(url, headers=HEADERS)

    if response.status_code == 200:
        data = response.json()

        content = base64.b64decode(data["content"]).decode("utf-8")

        return content

    elif response.status_code == 404:
        return "README_NOT_FOUND"

    else:
        print(
            f"README request failed for {owner}/{repo_name}: "
            f"{response.status_code} - {response.text[:200]}"
        )
        return ""