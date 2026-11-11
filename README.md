# RepoFinder AI

**Natural-language GitHub repository discovery using hybrid search and
requirement-aware ranking.**

RepoFinder AI helps users find GitHub repositories by describing what
they want to build in ordinary language. It combines semantic retrieval,
keyword retrieval, GitHub's live repository search, and an LLM-assisted
requirement evaluation stage.

The project is designed to run locally wherever practical. Repository
embeddings and vector search use local components; Gemini is used for
query understanding and requirement evaluation when configured, with a
locally hosted Qwen model available as a fallback.

> **Project status:** Working prototype / academic engineering project.
> Search quality has been evaluated on a 14-query dataset. Results and
> limitations are documented in
> [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

## Features

-   **Natural-language search:** Describe a project in detail instead of
    manually constructing a GitHub search query.
-   **LLM query understanding:** Converts a request into a semantic
    query, a GitHub API query, and structured requirements.
-   **Live GitHub discovery:** Searches GitHub repositories and
    retrieves repository README content.
-   **Local semantic search:** Uses sentence-transformer embeddings and
    FAISS.
-   **Keyword search:** Uses BM25 to find exact terms and technology
    names.
-   **Hybrid retrieval:** Combines FAISS and BM25 rankings using
    Reciprocal Rank Fusion (RRF).
-   **Repository-aware ranking:** Considers relevance, stars, recent
    activity, and archived status.
-   **Requirement-aware evaluation:** Evaluates candidate repositories
    against the user's extracted requirements.
-   **Streamlit interface:** Provides search-depth and result-count
    controls, repository details, scores, and requirement explanations.
-   **Local LLM fallback:** Can use a local Qwen GGUF model when Gemini
    is unavailable, provided the model file and runtime dependencies are
    configured.

## How it works

``` text
User's natural-language request
             |
             v
      Query understanding
       Gemini / local Qwen
             |
             v
  +----------+-----------+
  |                      |
  v                      v
Semantic query       GitHub query
  |                      |
  |                      v
  |               GitHub REST API
  |                      |
  |               New repository data
  |                      |
  |               Document processing
  |                      |
  |               Incremental indexing
  |                      |
  +----------+-----------+
             |
             v
      Hybrid retrieval
       /           \
      v             v
    FAISS          BM25
       \           /
        \         /
             v
       Reciprocal Rank
          Fusion
             |
             v
      Candidate repositories
             |
             v
   Requirement evaluation
       Gemini / fallback
             |
             v
  High-importance mismatch filter
             |
             v
   Requirement-aware score
             |
             v
       Ranked results
             |
             v
       Streamlit UI
```

### Retrieval and ranking

The hybrid search stage retrieves candidates independently from FAISS
and BM25, then combines their ranks using Reciprocal Rank Fusion.

The current repository-ranking formula is:

``` text
hybrid_score =
    0.80 * relevance_score
  + 0.10 * popularity_score
  + 0.08 * activity_score
  + 0.02 * archived_status_score
```

Where:

-   `relevance_score` is the normalized RRF score.
-   `popularity_score` is calculated from GitHub stars using a
    logarithmic scale.
-   `activity_score` decays with the age of the last push.
-   `archived_status_score` is `1.0` for non-archived repositories and
    `0.0` for archived repositories.

The later requirement-aware stage uses:

``` text
final_score =
    0.65 * hybrid_score
  + 0.35 * requirement_score
```

A repository is excluded when it explicitly mismatches a high-importance
requirement. Unknown evidence does not automatically exclude a
repository; it receives partial credit in requirement scoring.

These are two distinct stages: the first ranks retrieval candidates, and
the second adjusts their order according to the user's stated
requirements.

## Technology stack

  ------------------------------------------------------------------------------
  Area                                Technology
  ----------------------------------- ------------------------------------------
  User interface                      Streamlit

  Backend language                    Python

  GitHub data                         GitHub REST API

  Query understanding and evaluation  Gemini through LangChain integration

  Local LLM fallback                  Qwen GGUF model

  Embeddings                          `sentence-transformers/all-MiniLM-L6-v2`

  Vector search                       FAISS

  Keyword retrieval                   BM25 (`rank_bm25`)

  Document processing                 LangChain document utilities and text
                                      splitters

  Persistent repository data          JSON

  Evaluation                          Custom Python evaluation scripts
  ------------------------------------------------------------------------------

The local embedding model is used for vector representations; it is not
the generative LLM that interprets a user's request.

## Project structure

The following describes the active application modules. Generated data
and the local model are runtime assets, not Python source modules.

``` text
RepoFinder AI/
├── app.py
├── main.py
├── index.py
├── backend/
│   ├── document_builder.py
│   ├── document_processing.py
│   ├── evaluation.py
│   ├── github_services.py
│   ├── indexing.py
│   ├── live_search.py
│   ├── llm_providers.py
│   ├── query_understanding.py
│   ├── requirement_evaluation.py
│   ├── search.py
│   ├── storage.py
│   ├── text_processing.py
│   └── vector_store.py
├── data/
│   ├── repositories.json
│   ├── index_manifest.json
│   ├── evaluation_queries.json
│   └── faiss_index/
├── models/
│   └── qwen2.5-1.5b-instruct-q4_k_m.gguf
├── .env
├── README.md
└── PROJECT_REPORT.md
```

`evidence_collector.py` was moved out of the project after searches
found no references to its module name or its public collection
functions in the other Python files. It is not part of the documented
active workflow.

## Setup

### 1. Requirements

-   Python 3.11 is the tested Python version for the current development
    environment.
-   GitHub API access. A GitHub token can be configured to improve API
    rate limits.
-   A Gemini API key for the configured Gemini workflow.
-   Sufficient disk space for the local embedding model, FAISS data, and
    optional Qwen GGUF model.

The Qwen model is approximately 1 GB in the development setup. It is a
local asset and should not be committed to a normal source-code
repository.

### 2. Create and activate a virtual environment

macOS / Linux:

``` bash
cd "RepoFinder AI"
python3.11 -m venv repo
source repo/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies

Install the project's pinned dependencies from `requirements.txt` if
that file is present:

``` bash
python -m pip install -r requirements.txt
```

If dependency pins have not yet been generated, create a reproducible
dependency file from the tested environment before sharing the project.
Do not assume that installing unpinned latest versions will reproduce
the development environment.

### 4. Configure environment variables

Create a `.env` file in the project root. Add the variables expected by
the provider configuration in `backend/llm_providers.py` and
query/evaluation modules.

Typical configuration keys used during development:

``` dotenv
GOOGLE_API_KEY=your_gemini_api_key
GITHUB_TOKEN=your_github_token
GEMINI_MODEL=gemini-3.8-flash
LOCAL_MODEL_PATH=models/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

Use your own credentials. Never commit `.env` or paste secret values
into issue reports. The configured Gemini model may time out; the
application has model fallback attempts, and local Qwen is the final
fallback when configured.

`HF_TOKEN` was present in the development environment but was not part
of the active hosted-inference workflow described here.

### 5. Prepare runtime data

The application expects its repository JSON data and FAISS index at the
paths configured in the backend:

``` text
data/repositories.json
data/faiss_index/
```

If these assets are not supplied, run the indexing workflow after
configuring GitHub access. The exact collection queries and indexing
parameters are defined in `index.py` and `backend/indexing.py`; review
them before starting a large collection.

## Run the application

From the project root with the virtual environment activated:

``` bash
streamlit run app.py
```

Streamlit normally opens at `http://localhost:8501`. If that port is
already occupied, Streamlit may select another available port and print
it in the terminal.

The project also includes a command-line entry point:

``` bash
python main.py
```

The indexing entry point is:

``` bash
python index.py
```

Review the configured GitHub queries and page limits in `index.py`
before running indexing.

## Using the Streamlit interface

1.  Start the application.
2.  Enter a detailed description of the software repository you want to
    find.
3.  Choose a search depth. The interface maps this user-facing setting
    to GitHub page and per-page limits.
4.  Choose how many final results to display.
5.  Start the search.
6.  Review repository descriptions, language, stars, topics, GitHub
    links, and scores.
7.  Expand the match explanation to inspect requirement statuses and the
    evidence returned by the evaluator.

Detailed prompts are useful because they give query understanding and
requirement evaluation more context. However, an LLM-generated `match`
is not independent proof that a repository implements a feature. Check
the repository README, source code, issues, and license before relying
on it.

## Evaluation

The project includes a custom evaluation dataset at
`data/evaluation_queries.json` and evaluation logic in
`backend/evaluation.py`.

The reported 14-query baseline was:

  Metric                Result
  ------------------- --------
  Precision@5            0.600
  Recall@5               0.658
  MRR@5                  0.889
  Candidate recall       1.000
  Duplicate results          0

A ranking-strategy comparison found the current hybrid formula achieved
the best Precision@5 and Recall@5 among the tested variants. These
values describe the project's evaluation dataset, not a guarantee of
performance on arbitrary GitHub searches.

## Important limitations

-   Repository search coverage depends on GitHub API results, API rate
    limits, selected pages, and the content already indexed.
-   README-based evidence can be incomplete or outdated.
-   LLM requirement judgments can be too generous when a requirement
    contains multiple features. A broad description may support only
    part of a compound requirement.
-   `unknown` means the available repository evidence was insufficient;
    it does not mean the feature is absent.
-   A repository being open source does not automatically establish that
    every dependency, hosted service, or model it uses is free.
-   A claim that a project runs locally does not establish that it is
    lightweight enough for a low-spec computer.
-   Search scores are ranking signals, not probabilities that a
    repository is a perfect match.
-   Gemini requests can time out. Fallback models may differ in quality
    and output consistency.
-   The Qwen fallback requires a compatible local model file and runtime
    support; it consumes local storage and memory.

## Troubleshooting

### Streamlit reports that port 8501 is in use

Use the port printed by Streamlit, or specify another port:

``` bash
streamlit run app.py --server.port 8502
```

### Python package installation causes NumPy / PyTorch compatibility errors

The development environment encountered a NumPy 2.x compatibility
problem with its PyTorch build. The tested compatible versions were:

``` bash
python -m pip install "numpy==1.26.4" "pandas==2.2.3"
```

Do not apply these pins blindly to a different environment; check the
installed PyTorch and package constraints first.

### Gemini times out

The application attempts configured fallback Gemini models. If Gemini
remains unavailable, local Qwen can be used when the model path and
local inference dependencies are correctly configured. Check the
terminal logs and environment variables without exposing API keys.

### FAISS index cannot be loaded

Check that `data/faiss_index/` contains the expected index files and
that the index was created with a compatible embedding configuration. If
the index is missing or incompatible, follow the indexing workflow
rather than manually editing the FAISS files.

### Search results do not reflect newly fetched repositories

Check whether live GitHub collection succeeded, whether new documents
were saved, and whether indexing completed. Also check the GitHub API
response and rate-limit messages.

## Security and repository hygiene

Before publishing:

-   Keep `.env` out of version control.
-   Keep virtual environments, Python caches, and local secrets out of
    version control.
-   Do not commit the approximately 1 GB local GGUF model unless there
    is a deliberate distribution plan.
-   Decide whether generated FAISS files and repository JSON data should
    be included. They may be reproducible, large, or subject to
    GitHub/API terms and project-specific data policies.
-   Provide a dependency file that matches the tested environment.
-   Review all files for credentials before pushing.

## Further documentation

See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the detailed
engineering history, design decisions, problems encountered, fixes,
experiments, and known limitations.
