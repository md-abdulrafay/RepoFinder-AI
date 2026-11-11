import json
import re
import os

from backend.llm_providers import (
    generate_with_gemini,
    generate_with_local
)


def build_prompt(user_query):
    return f"""
Understand this GitHub repository search request.

User request:
{user_query}

Return ONLY valid JSON using this exact structure:

{{
    "semantic_query": "Enriched search description",
    "github_query": "Concise GitHub repository search query",
    "requirements": [
        {{
            "id": "short_unique_identifier",
            "description": "Specific requirement to evaluate",
            "importance": "high"
        }}
    ]
}}

Rules:
- Preserve the user's original intent.
- Extract explicit and clearly implied requirements.
- Do not invent requirements.
- Include technical requirements, budget constraints,
  hardware constraints, and feature constraints when relevant.
- Use importance values: high, medium, low.
- Use unique, short IDs.
- Include an empty requirements list if no constraints exist.
- GitHub query must use GitHub repository search syntax.
- Return JSON only. No Markdown or explanations.
"""


def parse_response(response):

    if isinstance(response, list):
        response = "".join(
            item.get("text", "")
            for item in response
            if isinstance(item, dict)
        )

    if not isinstance(response, str):
        raise ValueError("Model returned an invalid response")

    response = response.strip()

    response = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        response
    )

    start = response.find("{")
    end = response.rfind("}")

    if start == -1 or end == -1:
        raise ValueError("No JSON object found")

    data = json.loads(response[start:end + 1])

    semantic_query = data.get("semantic_query")
    github_query = data.get("github_query")
    requirements = data.get("requirements", [])

    if not isinstance(semantic_query, str) or not semantic_query.strip():
        raise ValueError("Missing semantic_query")

    if not isinstance(github_query, str) or not github_query.strip():
        raise ValueError("Missing github_query")

    if not isinstance(requirements, list):
        raise ValueError("Requirements must be a list")

    validated_requirements = []

    for requirement in requirements:

        if not isinstance(requirement, dict):
            continue

        requirement_id = requirement.get("id")
        description = requirement.get("description")
        importance = requirement.get("importance", "medium")

        if not isinstance(requirement_id, str) or not requirement_id.strip():
            continue

        if not isinstance(description, str) or not description.strip():
            continue

        if importance not in ("high", "medium", "low"):
            importance = "medium"

        validated_requirements.append({
            "id": requirement_id.strip(),
            "description": description.strip(),
            "importance": importance
        })

    return {
        "semantic_query": semantic_query.strip(),
        "github_query": github_query.strip(),
        "requirements": validated_requirements
    }


def understand_query(user_query):

    prompt = build_prompt(user_query)

    primary_model = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash"
    )

    gemini_models = [
        primary_model,
        "gemini-3.6-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite"
    ]

    gemini_models = list(dict.fromkeys(gemini_models))

    for model_name in gemini_models:

        try:
            print(f"\nTrying Gemini model: {model_name}...")

            response = generate_with_gemini(
                prompt,
                model_name
            )

            result = parse_response(response)

            result["provider"] = f"Gemini ({model_name})"
            result["model_name"] = model_name

            print(f"Query understood using {model_name}")

            return result

        except Exception as error:
            print(f"{model_name} failed: {error}")

    try:
        print("\nAll Gemini models failed.")
        print("Switching to Local Qwen...")

        response = generate_with_local(prompt)

        result = parse_response(response)

        result["provider"] = "Local Qwen"
        result["model_name"] = None

        print("Query understood using Local Qwen")

        return result

    except Exception as error:
        print(f"Local Qwen failed: {error}")

    print("\nAll LLM providers failed.")

    return None