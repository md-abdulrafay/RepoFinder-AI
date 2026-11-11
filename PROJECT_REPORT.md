# RepoFinder AI --- Detailed Engineering Project Report

**Project:** RepoFinder AI\
**Type:** Natural-language GitHub repository discovery system\
**Status:** Working prototype with a measured retrieval baseline\
**Primary interface:** Streamlit\
**Backend language:** Python

------------------------------------------------------------------------

## 1. Executive summary

RepoFinder AI is a repository-discovery application that accepts a
natural-language description of a software project and searches GitHub
for repositories that may satisfy the request.

A user may describe not only a project category, but also its required
capabilities, preferred programming languages, frameworks, databases,
deployment model, licensing expectations, hardware constraints, and
other preferences. The application attempts to convert that description
into a search query and a structured list of requirements.

The retrieval engine combines two complementary approaches:

1.  Semantic retrieval using local sentence-transformer embeddings and
    FAISS.
2.  Keyword retrieval using BM25.

Reciprocal Rank Fusion (RRF) combines the two ranked lists. A
repository-ranking formula then considers relevance, popularity,
activity, and archived status. A separate LLM-assisted
requirement-evaluation stage checks candidate repositories against the
requirements extracted from the user's request.

The system has a Streamlit interface, a JSON repository store, a FAISS
index, a custom evaluation dataset, Gemini integration, and a local Qwen
fallback.

The project is a prototype, not a guarantee that every returned
repository implements every requested feature. The experiments
identified an important limitation: an LLM can mark a compound
requirement as matched even when the available README evidence supports
only part of it. This report distinguishes measured results from
qualitative observations and records that limitation explicitly.

## 2. Problem statement

GitHub contains a very large number of repositories. Finding a useful
project is difficult when the user knows what they want to build but
does not know the exact repository name, keywords, or GitHub search
syntax.

Traditional keyword search has several weaknesses:

-   Users may describe a capability using different words from those
    used in a repository README.
-   A search term may match a repository that is only loosely related to
    the intended project.
-   A repository may mention a technology without implementing the
    requested workflow.
-   A user may have many constraints that cannot be expressed clearly in
    one short keyword query.
-   Search ranking based only on popularity can favor famous projects
    over more relevant projects.

The project addresses these issues by combining semantic and lexical
retrieval, live GitHub discovery, structured requirements, and a
second-stage candidate evaluation.

## 3. Objectives

The principal objectives were:

1.  Accept detailed natural-language repository requests.
2.  Convert a request into a semantic search query and a
    GitHub-compatible query.
3.  Extract explicit requirements from the request.
4.  Collect GitHub repository metadata and README content.
5.  Build a local searchable document collection.
6.  Support semantic and keyword retrieval.
7.  combine retrieval rankings rather than depending on one retrieval
    method.
8.  Include repository popularity and activity as smaller ranking
    signals.
9.  Evaluate candidate repositories against user requirements.
10. Provide a practical Streamlit interface.
11. Measure retrieval quality using a fixed evaluation dataset.
12. Prefer free or locally runnable components where practical.

## 4. Scope and non-goals

### In scope

-   GitHub repository discovery.
-   Repository metadata and README collection.
-   Local document processing and embedding.
-   FAISS semantic retrieval.
-   BM25 keyword retrieval.
-   RRF fusion and repository ranking.
-   LLM-based query understanding and requirement evaluation.
-   Gemini with a local Qwen fallback.
-   Streamlit search interface.
-   Offline evaluation of ranking strategies.

### Not established by the current implementation

The following should not be claimed as completed unless separately
implemented and tested:

-   Exhaustive indexing of all GitHub repositories.
-   Verification of source-code behavior for every returned repository.
-   Guaranteed correctness of license, pricing, or hardware claims.
-   A production-grade multi-user service with authentication and
    authorization.
-   A guarantee that every result is free, self-hosted, or compatible
    with a user's hardware.
-   A guarantee that every requested feature exists simply because the
    evaluator returns `match`.

## 5. High-level architecture

``` text
                         USER
                           |
                           v
                    Streamlit app.py
                           |
                           v
                 Natural-language query
                           |
                           v
                 Query understanding
                   Gemini / Qwen
                           |
               +-----------+-----------+
               |                       |
               v                       v
         Semantic query           GitHub query
               |                       |
               |                       v
               |                GitHub REST API
               |                       |
               |                Repository metadata
               |                       |
               |                  README fetch
               |                       |
               |                Document builder
               |                       |
               |                JSON repository store
               |                       |
               |                Incremental indexing
               |                       |
               +-----------+-----------+
                           |
                           v
                    Hybrid retrieval
                     /           \
                    v             v
                  FAISS          BM25
                    \             /
                     \           /
                           v
                    RRF combination
                           |
                           v
                  Candidate repositories
                           |
                           v
                 Requirement evaluation
                     Gemini / fallback
                           |
                           v
                 High-importance filter
                           |
                           v
                 Requirement-aware ranking
                           |
                           v
                    Ranked results
                           |
                           v
                    Streamlit display
```

The diagram represents the intended active search flow. Initial
collection/indexing can also be run through the separate indexing entry
point.

## 6. Technology choices

  ------------------------------------------------------------------------------------------
  Component               Technology                                 Purpose
  ----------------------- ------------------------------------------ -----------------------
  Programming language    Python                                     Backend and application
                                                                     logic

  UI                      Streamlit                                  Search form and result
                                                                     display

  Repository source       GitHub REST API                            Search and repository
                                                                     data

  Text embeddings         `sentence-transformers/all-MiniLM-L6-v2`   Convert text into
                                                                     vectors

  Vector index            FAISS                                      Semantic
                                                                     nearest-neighbor
                                                                     retrieval

  Lexical retrieval       BM25 via `rank_bm25`                       Keyword relevance

  Fusion                  Reciprocal Rank Fusion                     Combine ranked
                                                                     retrieval lists

  Document processing     LangChain document utilities               Document loading and
                                                                     chunking

  Primary LLM provider    Gemini                                     Query understanding and
                                                                     requirement evaluation

  Local LLM fallback      Qwen GGUF                                  Local generation when
                                                                     configured

  Repository storage      JSON                                       Persistent repository
                                                                     records

  Evaluation              Custom Python scripts                      Ranking metrics and
                                                                     comparisons
  ------------------------------------------------------------------------------------------

### Why use both embeddings and BM25?

Semantic retrieval can find conceptually similar descriptions even if
the exact words differ. Keyword retrieval can identify exact framework
names, database names, and technical terms.

Neither method is sufficient for every query. Combining their rankings
gives the system two different relevance signals.

### Why use a local embedding model?

The embedding model is downloaded and run locally. It creates vectors
for repository text and queries. This avoids requiring a paid hosted
embedding endpoint for each search.

The embedding model is not the same thing as Gemini or Qwen: it
generates vector representations, not natural-language explanations.

### Why use Gemini and local Qwen?

Gemini provides query interpretation and requirement-evaluation
capabilities without running a large generative model on the development
machine. However, API calls can time out or become unavailable.

A local Qwen GGUF model provides a fallback path. It avoids a hosted
inference subscription but requires a local model file, compatible
inference dependencies, memory, and CPU time. Local inference is not
resource-free.

## 7. Repository collection and document creation

### 7.1 GitHub API search

`backend/github_services.py` contains the GitHub repository search and
README retrieval functions.

The search function accepts a GitHub query, page number, and page size.
The implementation uses the GitHub REST API and a request timeout. API
failures and rate-limit responses are represented through the project's
GitHub API error handling.

The live-search stage can continue with the locally stored collection if
GitHub retrieval fails.

### 7.2 README retrieval

For each discovered repository, the application attempts to retrieve its
README. README content supplies context that is often missing from the
short GitHub description.

README retrieval is not equivalent to inspecting and executing the
repository's source code. It can be absent, incomplete, outdated, or
inaccurate.

### 7.3 Document building

`backend/document_builder.py` combines repository information into a
document. The stored record includes fields such as:

-   Repository ID and full name.
-   Description.
-   Main language.
-   Stars and forks.
-   Last push date.
-   Archived status.
-   GitHub URL.
-   Topics.
-   Combined searchable content.

This creates one consistent representation for later storage, indexing,
retrieval, and display.

### 7.4 Text cleaning and document splitting

The text-processing and document-processing modules prepare repository
content for indexing.

The current text splitter uses:

-   Chunk size: 1,000 characters.
-   Chunk overlap: 200 characters.

Overlap helps preserve context when a relevant passage falls near a
chunk boundary. Chunking also makes it possible to retrieve a relevant
part of a long README rather than treating the entire README as one
indivisible unit.

## 8. Storage and indexing

### 8.1 JSON repository store

`backend/storage.py` provides the JSON persistence functions. The main
repository data file is:

``` text
data/repositories.json
```

JSON stores the repository records and searchable content used by
document processing and metadata refresh.

### 8.2 FAISS vector index

`backend/vector_store.py` loads and manages the FAISS vector store. The
current index path is:

``` text
data/faiss_index/
```

The development environment contained:

``` text
data/faiss_index/index.faiss
data/faiss_index/index.pkl
```

The vector index must remain compatible with the embedding model and
document representation used to create it.

### 8.3 Incremental indexing

The indexing implementation tracks repository content using a content
fingerprint/manifest. It can identify repositories whose searchable
content has changed and update their indexed chunks instead of always
rebuilding the entire collection.

The intended benefits are:

-   Avoid re-embedding unchanged repository content.
-   Reduce unnecessary index work.
-   Keep the searchable index synchronized with changed records.
-   Make live discovery more practical.

The implementation uses repository identity and content hashes to decide
whether an update is needed. The JSON store and FAISS index should be
treated as related runtime assets; manually editing one without updating
the other can cause inconsistencies.

## 9. Search engine implementation

The main retrieval and ranking implementation is in `backend/search.py`.

### 9.1 Loading the search engine

`load_search_engine()` loads the existing FAISS vector store from the
configured index path.

### 9.2 Candidate pool

The search function retrieves a wider set of candidates before returning
the final `k` results:

``` python
candidate_k = max(k * 10, 50)
```

This gives the ranking stage more candidates than the number ultimately
displayed.

### 9.3 FAISS semantic retrieval

FAISS searches the vector index using the query. The implementation
retrieves a wider number of chunks, then keeps the first occurrence of
each repository ID until it has enough unique repositories.

This deduplication matters because a repository may have several chunks
that are individually similar to the query. The result list should rank
repositories, not repeatedly show the same repository because multiple
chunks matched.

### 9.4 BM25 keyword retrieval

The BM25 path:

1.  Loads repository records.
2.  Calculates a SHA-256 fingerprint of repository IDs and content.
3.  Rebuilds the in-memory BM25 index only when the content fingerprint
    changes.
4.  Splits repository documents using the same document-processing
    pipeline.
5.  Tokenizes chunk text.
6.  Scores chunks against query tokens.
7.  Keeps unique repositories with positive scores.

The cache avoids rebuilding BM25 on every search when repository content
has not changed.

### 9.5 Reciprocal Rank Fusion

FAISS and BM25 return separate ranked lists. RRF combines them using:

\[ RRF(d)=`\sum`{=tex}\_{r
`\in `{=tex}rankings}`\frac{1}{60+rank_r(d)}`{=tex} \]

Here, (rank_r(d)) is the rank of document/repository (d) in a retrieval
list. The constant 60 reduces the effect of small rank differences.

The implementation deduplicates repositories within each ranked list
before assigning ranks. A repository appearing in both lists receives
contributions from both.

RRF combines ranks, not raw FAISS distances and BM25 scores. This avoids
directly adding scores whose scales and meanings differ.

### 9.6 Repository metadata refresh

Before final scoring, the search implementation refreshes metadata from
the latest JSON records. This allows fields such as stars, description,
language, push date, archived status, URL, and topics to reflect the
stored latest repository record rather than stale chunk metadata.

### 9.7 Ranking signals

The implementation calculates four signals.

**Relevance**

The RRF score is normalized by the maximum RRF score among candidates:

\[ Relevance(d)=`\frac{RRF(d)}{\max_{x}RRF(x)}`{=tex} \]

**Popularity**

Popularity is calculated from stars using a logarithmic scale with
10,000 stars as a reference:

\[ Popularity(d)=`\min`{=tex}`\left`{=tex}(
`\frac{\log(1+stars_d)}{\log(1+10000)}`{=tex},1 `\right`{=tex}) \]

A logarithmic transformation prevents very large star counts from
dominating linearly.

**Activity**

Activity is based on the age in days of the last push:

\[ Activity(d)=e\^{-ageDays/365} \]

If the date is missing or cannot be parsed, the implementation uses a
neutral fallback value of 0.5.

**Archived status**

The archived-status score is 1.0 for a repository that is not archived
and 0.0 for an archived repository.

### 9.8 Ranking strategy experiments

The search implementation calculates several strategies for comparison:

-   Pure RRF.
-   RRF plus popularity.
-   RRF plus activity.
-   RRF plus activity and archive status.
-   The current combined strategy.

The current formula is:

\[
```{=tex}
\begin{aligned}
HybridScore(d)={}&0.80 \times Relevance(d)\\
&+0.10 \times Popularity(d)\\
&+0.08 \times Activity(d)\\
&+0.02 \times ArchivedStatus(d)
\end{aligned}
```
\]

The formula gives the largest weight to relevance. Popularity and
activity are supporting signals rather than replacements for relevance.

The tested baseline comparison favored this current strategy on
Precision@5 and Recall@5. The formula was therefore frozen for the
reported baseline rather than continually changing weights without new
evidence.

## 10. Natural-language query understanding

`backend/query_understanding.py` converts a user's request into
structured information.

The expected result contains:

``` json
{
  "semantic_query": "A search-oriented description of the user's intent",
  "github_query": "A GitHub-compatible repository search query",
  "requirements": [
    {
      "id": "requirement_1",
      "description": "A specific requested capability or constraint",
      "importance": "high"
    }
  ]
}
```

The actual values depend on the user's request.

### 10.1 Semantic query

The semantic query preserves the user's intent in a form suitable for
embedding-based retrieval. It can include capabilities, technologies,
and context that may not be suitable as literal GitHub search syntax.

### 10.2 GitHub query

The GitHub query is used for live repository discovery. GitHub search
has its own syntax and behavior, so this query is not interchangeable
with the semantic query.

### 10.3 Structured requirements

The model extracts individual requirements and assigns importance
levels:

-   `high`
-   `medium`
-   `low`

The IDs allow later evaluation results to be associated with the correct
requirement.

### 10.4 Provider fallback

The implementation tries the configured Gemini model and fallback Gemini
model names. If those attempts fail, it attempts local Qwen when
configured.

This fallback is important because Gemini requests timed out during
development. The fallback does not guarantee identical quality:
different models can interpret the same prompt differently.

## 11. Live search orchestration

`backend/live_search.py` coordinates the query and retrieval components.

The high-level sequence is:

1.  Receive the vector store, semantic query, GitHub query, result
    count, page count, page size, and extracted requirements.
2.  Request repository pages from GitHub.
3.  Deduplicate discovered repositories using repository identity.
4.  Handle GitHub API errors without necessarily discarding the existing
    local collection.
5.  Build and save documents for newly discovered or changed
    repositories.
6.  Update the searchable index when new/changed records are available.
7.  Retrieve a candidate pool using hybrid search.
8.  If no requirements were extracted, return the hybrid results.
9.  Otherwise, evaluate the candidates against the requirements.
10. Exclude a candidate if it explicitly mismatches a high-importance
    requirement.
11. Combine the hybrid score and requirement score.
12. Sort eligible candidates and return the requested number of results.

The requirement-aware score is:

\[ FinalScore(d)=0.65 `\times `{=tex}HybridScore(d)+0.35
`\times `{=tex}RequirementScore(d) \]

This formula belongs to the second-stage ranking process. It does not
replace the hybrid formula inside `backend/search.py`.

If requirement evaluation fails, the current orchestration has a
fallback that returns the original hybrid candidate results. This makes
search more resilient, but it also means a result list can be returned
without successful requirement-aware scoring. The UI checks for the
absence of requirement scores and can warn the user.

## 12. Requirement evaluation

`backend/requirement_evaluation.py` evaluates candidate repositories
against the structured requirements.

### 12.1 Evidence provided to the evaluator

The evaluator uses available repository information, including
description, topics, and README-derived content. The prompt limits the
model to the supplied repository evidence and treats README content as
untrusted data rather than instructions.

The evaluator does not independently execute the repository or prove
that a feature works.

### 12.2 Status values

Each requirement receives one of three statuses:

  Status       Meaning                                           Numeric value
  ------------ ----------------------------------------------- ---------------
  `match`      Supplied evidence supports the requirement                  1.0
  `unknown`    Evidence is insufficient to decide                          0.5
  `mismatch`   Supplied evidence contradicts the requirement               0.0

### 12.3 Importance weights

  Importance     Weight
  ------------ --------
  High                3
  Medium              2
  Low                 1

The requirement score is a weighted average of the status values:

\[ RequirementScore =
`\frac{\sum_i weight_i \times statusValue_i}`{=tex} {`\sum`{=tex}\_i
weight_i} \]

A missing evaluation is treated as unknown rather than silently treated
as a match.

### 12.4 High-importance mismatch filtering

After evaluation, live search checks high-importance requirements. If a
repository explicitly mismatches any high-importance requirement, it is
excluded.

An unknown status does not automatically exclude a repository. It
reduces the requirement score instead.

This distinction is deliberate: absence of evidence in a README is not
necessarily evidence that a feature does not exist.

### 12.5 Important evaluation limitation

A requirement can contain several separate capabilities. For example:

> Support video lessons, quizzes, assignment submissions, progress
> tracking, and role-based access control.

A model may see that the README describes a learning platform and mark
the entire requirement as a match, even though the README only proves
course management or enrollment.

This is an observed weakness from project testing. A `match` should
therefore be interpreted as an LLM judgment based on supplied text, not
as verified implementation evidence. The system would need more granular
requirement decomposition and stronger evidence validation to reduce
this risk.

## 13. Streamlit interface

`app.py` provides the user-facing application.

The interface includes:

-   Page title and layout configuration.
-   A cached search-engine load.
-   A natural-language query input.
-   A search-depth selector.
-   A final result-count selector.
-   A search action.
-   Query-understanding output and requirement count.
-   Repository result cards/sections.
-   Repository metadata and GitHub links.
-   Hybrid and final scores.
-   Expandable requirement explanations.

### Search-depth settings

The interface hides raw GitHub API variables from the user and exposes
understandable choices. The current mapping is:

  User-facing depth     Pages   Results per page
  ------------------- ------- ------------------
  Quick                     1                 10
  Balanced                  2                 10
  Deep                      3                 20
  Very Deep                 4                 30
  Exhaustive                5                 50

The selected values control how many GitHub search results are
requested. Higher depth can increase API usage and search time. It does
not guarantee better relevance.

## 14. Problems encountered and solutions

This section records issues encountered during development and the
response taken. It distinguishes resolved environment issues from
limitations that remain.

### 14.1 GitHub API and live-search reliability

**Problem:** Live repository collection can fail because of API errors
or rate limits.

**Response:** The GitHub service layer has explicit error handling, and
live search can continue with the local repository collection when
GitHub retrieval fails.

**Remaining limitation:** Local fallback can only search data already
collected. It cannot discover repositories that were never fetched.

### 14.2 Duplicate repositories from multiple chunks

**Problem:** A repository can have several README chunks, and several
chunks may appear near the top of a retrieval list. Returning
chunk-level results directly could show the same repository repeatedly.

**Solution:** FAISS and BM25 results are deduplicated by repository ID
before their ranks are fused. RRF also avoids counting duplicate
repository entries within one retrieval list.

**Outcome:** The evaluation baseline reported zero duplicate results.

### 14.3 Semantic search alone is not enough

**Problem:** Embedding similarity can find conceptually related projects
but may miss exact technology names or specific terms.

**Solution:** BM25 keyword retrieval was added alongside FAISS semantic
retrieval.

**Outcome:** The system now has both semantic and lexical candidate
lists, combined by RRF.

### 14.4 Combining incompatible retrieval scores

**Problem:** FAISS distances and BM25 scores have different scales.
Directly adding them would not provide a principled combination.

**Solution:** Reciprocal Rank Fusion combines the positions of
repositories in ranked lists rather than adding raw scores.

### 14.5 Rebuilding BM25 unnecessarily

**Problem:** Rebuilding a lexical index on every search would add
avoidable work.

**Solution:** The application hashes the repository IDs and content used
by BM25. It rebuilds the cached index only if the hash changes.

### 14.6 Keeping FAISS and repository records synchronized

**Problem:** Repository content can change after initial collection.
Stale chunks can remain in the index if updates are not handled.

**Solution:** The indexing workflow uses repository identity and content
fingerprints to identify changed content and update the corresponding
vectors.

**Remaining caution:** Keep the JSON records, index manifest, and FAISS
files together as related runtime assets. Do not manually edit only one
of them.

### 14.7 Gemini model timeouts

**Problem:** The configured Gemini model returned 504 timeouts during
development. A request to one model could therefore prevent query
understanding or requirement evaluation from completing.

**Solution:** The provider/query modules try fallback Gemini model
names. Local Qwen is also available as a fallback when its model file
and runtime dependencies are configured.

**Remaining limitation:** Fallback models can produce different outputs
and have different latency. A fallback does not guarantee the same
quality as the primary model.

### 14.8 Local LLM versus hosted inference

**Problem:** Hosted inference services can introduce subscriptions,
usage credits, or external dependencies. The project goal favored free
or locally runnable components.

**Decision:** Use Gemini as the primary generative provider and a
locally downloaded Qwen GGUF model as a fallback. The embedding model
also runs locally.

**Trade-off:** Local inference avoids a hosted inference bill but uses
local disk, memory, CPU, and setup effort. It is not necessarily fast on
a low-spec machine.

### 14.9 Python package compatibility: NumPy and PyTorch

**Problem:** Installing a newer Streamlit dependency set pulled NumPy
2.x, while the installed PyTorch build had been compiled against NumPy
1.x. This caused compatibility problems when converting tensors to NumPy
arrays.

**Solution in the development environment:**

``` bash
python -m pip install "numpy==1.26.4"
python -m pip install "pandas==2.2.3"
```

**Verification recorded:**

``` text
NumPy: 1.26.4
Pandas: 2.2.3
PyTorch: 2.2.2
Tensor: [1 2 3]
FAISS: OK
```

This fixed the tested environment. It should not be applied blindly to
every future environment; dependency constraints should be checked
together.

### 14.10 Streamlit port already occupied

**Problem:** Port 8501 was already in use during development.

**Response:** Streamlit selected another available port (8502 was used).
The application can also be started with an explicit port:

``` bash
streamlit run app.py --server.port 8502
```

### 14.11 Requirement evaluation can overstate evidence

**Problem:** In detailed tests, the evaluator sometimes marked a
compound requirement as matched when the repository description or
README supported only part of it.

**Examples observed:**

-   An LMS repository was credited for a broad collection of features
    even though the supplied evidence clearly supported only some of
    them.
-   A generic inventory description did not prove every requested
    feature, such as low-stock notifications or analytics.
-   A project described as local or free was not necessarily proven
    lightweight or free of paid external services.

**Response during the project:** These results were recorded as
limitations rather than treated as proof of full compatibility. The
evaluation prompt already asks the model to use supplied evidence,
distinguish unknown from mismatch, and avoid assumptions about licensing
and cost.

**Remaining limitation:** The evaluator still needs more granular
requirements and evidence for each capability. This report does not
claim that this issue has been fully solved.

### 14.12 Incomplete YouTube automation prompt during testing

**Problem:** An early test prompt ended before the user's complete
requirements were stated. The resulting requirements did not strongly
enforce long-form video generation, and several Shorts-focused
repositories appeared.

**Response:** The test was repeated with a complete prompt explicitly
requiring videos of at least 8--10 minutes and distinguishing long-form
generation from Shorts-only tools.

**Outcome:** The controlled prompt produced a more relevant top result
and better exposed the difference between long-form and Shorts-focused
projects.

**Lesson:** Query quality and completeness affect query understanding,
GitHub retrieval, and requirement evaluation. Test prompts should state
the actual constraints clearly.

### 14.13 Unused evidence collector module

**Problem:** `backend/evidence_collector.py` contained functions for
collecting additional evidence such as license, dependencies, homepage,
and pricing information, but the active application did not appear to
call it.

**Investigation:** Two repository-wide searches were run: one for the
module name and another for its public collection function names. Both
returned no references outside the module.

**Action:** The file was moved out of the project as a backup rather
than permanently deleted.

**Documentation decision:** It is not described as an active feature or
dependency of the current search pipeline.

## 15. Experiments with detailed natural-language prompts

The system was tested with realistic, long-form requests because users
may specify multiple capabilities and constraints rather than entering a
few keywords.

These examples are qualitative test observations. They are not a
statistically large benchmark and should not be presented as proof that
every returned repository fully satisfies its requirements.

### 15.1 AI document assistant

**Prompt summary:** An open-source personal assistant that understands
PDFs, answers questions with RAG, maintains conversation history,
supports multiple users, prefers Python/FastAPI/LangChain/vector
storage, and should run locally without expensive APIs.

**Observed result:** The top results included projects describing RAG,
PDF processing, FastAPI, LangChain, vector storage, conversation/session
support, and local model options.

**What worked:** The system found repositories matching several
technical and functional aspects of the request.

**What needed caution:** A repository with an OpenAI default model may
still require external API credentials even if the codebase can be
self-hosted. A local deployment option does not automatically prove that
all functionality is free.

### 15.2 Small-business inventory system

**Prompt summary:** Product management, stock tracking, sales records,
low-stock notifications, analytics dashboard, Python/Django or FastAPI,
PostgreSQL, self-hosted, and free.

**Observed result:** InvenTree appeared first, followed by Django
inventory/POS projects.

**What worked:** The system identified relevant inventory-management
projects and frameworks.

**Limitation:** The top repository did not have clear supplied evidence
for every requested feature. Some feature statuses were unknown. This is
a useful example of why a high overall score should not be interpreted
as proof of complete feature coverage.

### 15.3 Long-form YouTube automation

**Prompt summary:** Open-source automation for complete long-form videos
of at least 8--10 minutes, including script, voiceover, visuals,
subtitles, rendering, thumbnails, and uploads, with free tools and
lightweight local models.

**Initial test:** The prompt was incomplete. Several Shorts-focused
projects ranked highly.

**Controlled retest:** The complete prompt explicitly said "not just
YouTube Shorts" and listed the full workflow.

**Observed result:** A project describing both long- and short-form
workflows ranked first. Other results were less complete, including
Shorts-oriented projects and a non-Python project.

**Lesson:** Explicit negative constraints such as "not Shorts-only"
help, but the evaluator still needs direct evidence for duration,
hardware requirements, pricing, and each automation stage.

### 15.4 Online learning platform backend

**Prompt summary:** Open-source backend for student/teacher accounts,
courses, video lessons, quizzes, assignments, progress tracking, RBAC,
Python/FastAPI or Django, PostgreSQL, REST API, Docker, and tests.

**Observed result:** A FastAPI learning-platform API ranked first, with
other LMS and generic FastAPI projects also appearing.

**What worked:** The retrieval stage found repositories in the
LMS/backend domain.

**Limitation:** The first repository's available evidence supported some
backend and course-related features, but did not clearly prove every
requested feature such as video lessons, quizzes, assignments, and
progress tracking. Some generic backend templates also appeared.

**Lesson:** A project name or broad domain description should not be
enough to mark a full bundle of requirements as matched.

## 16. Quantitative retrieval evaluation

The project uses a 14-query evaluation dataset in:

``` text
data/evaluation_queries.json
```

The evaluation code is in:

``` text
backend/evaluation.py
```

### 16.1 Metrics

**Precision@5**

The fraction of the top five returned results judged relevant, averaged
over evaluation queries.

\[ Precision@5=`\frac{\text{Relevant results in top 5}}{5}`{=tex} \]

**Recall@5**

The fraction of known relevant repositories retrieved in the top five.

\[
Recall@5=`\frac{\text{Relevant results in top 5}}{\text{Total known relevant results}}`{=tex}
\]

**Mean Reciprocal Rank at 5 (MRR@5)**

Measures how early the first relevant result appears.

\[
MRR@5=`\frac{1}{N}`{=tex}`\sum`{=tex}\_{q=1}\^{N}`\frac{1}{rank_q}`{=tex}
\]

If no relevant result appears in the top five for a query, its
reciprocal-rank contribution is zero.

**Candidate recall**

Measures whether the candidate-generation stage includes the known
relevant repositories before final ranking.

**Duplicate count**

Counts repeated repositories in the returned results.

### 16.2 Ranking strategy comparison

The recorded experiment:

  Strategy                     Precision@5    Recall@5   MRR@5
  -------------------------- ------------- ----------- -------
  RRF only                           0.557       0.628   0.964
  RRF + popularity                   0.586       0.644   0.929
  RRF + activity                     0.586       0.650   0.964
  RRF + activity + archive           0.586       0.650   0.917
  Current combined formula       **0.600**   **0.658**   0.889

### 16.3 Selected baseline

The current combined formula achieved the highest Precision@5 and
Recall@5 among the tested strategies. It was retained as the default
despite a lower MRR@5 than some alternatives.

This is a trade-off: the chosen formula improved the average number of
relevant results in the top five, while some other strategies placed the
first relevant result earlier on average.

### 16.4 Final recorded metrics

  Metric                Recorded result
  ------------------- -----------------
  Number of queries                  14
  Precision@5                     0.600
  Recall@5                        0.658
  MRR@5                           0.889
  Candidate recall                1.000
  Duplicate results                   0

These metrics are tied to the dataset, relevance labels, and evaluation
procedure used at the time. They should be recalculated if the dataset,
indexing corpus, retrieval code, or relevance judgments change.

## 17. Design decisions and trade-offs

### 17.1 Keep hybrid retrieval

FAISS and BM25 solve different retrieval problems. Keeping both is more
defensible than relying on semantic similarity alone.

### 17.2 Keep RRF rather than adding raw retrieval scores

RRF is used because it fuses ranks without assuming that FAISS and BM25
scores are on the same scale.

### 17.3 Keep the tested hybrid ranking formula frozen

The selected formula had the best Precision@5 and Recall@5 in the
recorded comparison. It should not be changed without a new controlled
evaluation.

### 17.4 Keep requirement-aware ranking separate

The requirement-aware score is a second-stage adjustment. Keeping it
separate makes it possible to inspect baseline retrieval quality
independently from LLM judgments.

### 17.5 Treat unknown evidence differently from mismatch

A README may omit a feature even when the code implements it. Unknown
therefore receives partial credit and does not automatically cause
exclusion. Explicit contradiction on a high-importance requirement is
treated more strictly.

### 17.6 Prefer local components where practical

Local embeddings and local Qwen reduce dependence on hosted inference
for those operations. Gemini remains an external provider in the current
configuration, so the whole application should not be described as fully
offline.

## 18. Current project structure

The active source layout observed during development:

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
│   ├── evaluation_queries.json
│   ├── index_manifest.json
│   ├── repositories.json
│   └── faiss_index/
├── models/
│   └── local Qwen GGUF model
├── .env
├── README.md
└── PROJECT_REPORT.md
```

The exact files included in a public release may differ. In particular,
local models, `.env`, virtual environments, and generated index files
should be reviewed before publishing.

## 19. Reproducibility and operational notes

### Environment

The development environment used macOS, Python 3.11.2, and a local
virtual environment. The machine used for development had an Intel
MacBook Pro with 8 GB RAM.

### Start Streamlit

``` bash
cd "/Users/macbook/Desktop/RepoFinder AI"
source repo/bin/activate
streamlit run app.py
```

### Run command-line search

``` bash
python main.py
```

### Run indexing

``` bash
python index.py
```

Review the configured queries and page settings before running indexing.

### Run evaluation

The evaluation implementation and dataset are in `backend/evaluation.py`
and `data/evaluation_queries.json`. Use the project's actual evaluation
entry point and environment. Preserve the evaluation dataset and
relevance judgments when comparing future changes.

### Secrets

Do not commit `.env`, API keys, or tokens. Do not include secret values
in screenshots, logs, or documentation.

### Large assets

The development `models/` directory was approximately 1 GB. The `data/`
directory was approximately 9.1 MB at the time of inspection. A public
repository should make an explicit decision about generated data and
local model distribution instead of committing them automatically.

## 20. Future improvements

The following are possible improvements, not claims about current
functionality:

1.  Decompose compound requirements into smaller atomic requirements
    before evaluation.
2.  Require a separate evidence passage for each atomic requirement.
3.  Add a manual review option for high-impact claims such as license,
    pricing, and hardware compatibility.
4.  Inspect repository files such as dependency manifests and
    configuration files when README evidence is insufficient.
5.  Add automated tests for API errors, empty README files, malformed
    LLM output, duplicate repositories, and index-update failures.
6.  Record provider/model information and evaluation failure reasons in
    structured logs.
7.  Expand the relevance-labeled evaluation dataset with more detailed
    real-world prompts.
8.  Evaluate requirement-aware ranking separately from baseline
    retrieval.
9.  Measure latency, GitHub API usage, and local memory consumption at
    each search depth.
10. Provide a reproducible dependency lock or tested `requirements.txt`.
11. Add a clear setup check for missing FAISS files, model files, and
    environment variables.

These should be prioritized based on measured user impact rather than
added all at once.

## 21. Conclusion

RepoFinder AI combines natural-language query understanding, live GitHub
search, local semantic retrieval, BM25 keyword retrieval, Reciprocal
Rank Fusion, repository metadata ranking, and LLM-assisted requirement
evaluation.

The measured 14-query baseline achieved Precision@5 of 0.600, Recall@5
of 0.658, MRR@5 of 0.889, candidate recall of 1.000, and zero duplicate
results. The ranking comparison supported retaining the current hybrid
scoring formula.

The detailed prompt experiments showed that the system can retrieve
repositories from broad technical descriptions, but they also exposed a
significant limitation: LLM-based requirement evaluation can overstate
what repository documentation proves, particularly for compound
requirements.

The most accurate description of the project is therefore a **hybrid
GitHub repository discovery prototype with requirement-aware ranking**,
not an automated verifier of repository functionality. Its strongest
current contributions are the combination of semantic and keyword
retrieval, the separation of baseline and requirement-aware ranking, and
the evaluation work used to make ranking decisions measurable.
