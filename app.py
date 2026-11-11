
import streamlit as st

from backend.search import load_search_engine
from backend.live_search import live_search
from backend.query_understanding import understand_query


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RepoFinder AI",
    page_icon="🔎",
    layout="wide"
)


# --------------------------------------------------
# Load FAISS search engine once
# --------------------------------------------------

@st.cache_resource
def get_vector_store():
    return load_search_engine()


# --------------------------------------------------
# Search depth configuration
# --------------------------------------------------

SEARCH_DEPTHS = {
    "Quick": {
        "pages": 1,
        "per_page": 10
    },
    "Balanced": {
        "pages": 2,
        "per_page": 10
    },
    "Deep": {
        "pages": 3,
        "per_page": 20
    },
    "Very Deep": {
        "pages": 4,
        "per_page": 30
    },
    "Exhaustive": {
        "pages": 5,
        "per_page": 50
    }
}


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Search Settings")

search_depth = st.sidebar.select_slider(
    "Search depth",
    options=list(SEARCH_DEPTHS.keys()),
    value="Balanced",
    help="Explore more repositories by increasing search depth."
)

result_count = st.sidebar.select_slider(
    "Results to show",
    options=[5, 10, 15, 20],
    value=5
)

st.sidebar.divider()

st.sidebar.caption(
    "Higher search depth explores more GitHub repositories "
    "but may take longer."
)


# --------------------------------------------------
# Main interface
# --------------------------------------------------

st.title("🔎 RepoFinder AI")

st.markdown(
    "Discover GitHub repositories using natural language "
    "and hybrid AI-powered search."
)

st.markdown("### What are you looking for?")

user_query = st.text_area(
    "Describe your project requirements",
    placeholder=(
        "Example: FastAPI backend with PostgreSQL "
        "authentication and JWT"
    ),
    height=100,
    label_visibility="collapsed"
)

search_button = st.button(
    "Search repositories",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# Search execution
# --------------------------------------------------

if search_button:

    if not user_query.strip():

        st.warning("Please enter what you are looking for.")

    else:

        settings = SEARCH_DEPTHS[search_depth]

        try:

            # Load search engine
            with st.spinner("Loading search engine..."):
                vector_store = get_vector_store()

            # Understand user query
            with st.spinner("Understanding your requirements..."):

                query_data = understand_query(user_query)

            if query_data:

                semantic_query = query_data["semantic_query"]
                github_query = query_data["github_query"]
                requirements = query_data.get("requirements", [])

                st.success(
                    f"Query understood using {query_data['provider']}."
                )

                if requirements:
                    st.caption(
                        f"AI extracted {len(requirements)} requirements."
                    )

            else:

                semantic_query = user_query
                github_query = user_query
                requirements = []

                st.warning(
                    "AI query understanding was unavailable. "
                    "Searching using your original query."
                )

            # Search information
            st.info(f"Search depth: {search_depth}")

            with st.spinner(
                "Searching GitHub and ranking repositories..."
            ):

                results = live_search(
                    vector_store=vector_store,
                    semantic_query=semantic_query,
                    github_query=github_query,
                    requirements=requirements,
                    k=result_count,
                    pages=settings["pages"],
                    per_page=settings["per_page"]
                )

            # Display results
            st.divider()

            st.subheader("Search Results")

            st.caption(
                f"Found {len(results)} matching repositories."
            )

            # Check whether requirement evaluation failed
            if (
                requirements
                and results
                and all(
                    "requirement_score" not in doc.metadata
                    for doc, _ in results
                )
            ):

                st.warning(
                    "Gemini requirement evaluation was unavailable. "
                    "Showing results using the original hybrid ranking."
                )

            if not results:

                st.info("No matching repositories found.")

            else:

                for rank, (doc, score) in enumerate(results, start=1):

                    metadata = doc.metadata

                    with st.container(border=True):

                        col1, col2 = st.columns([5, 1])

                        with col1:

                            st.markdown(
                                f"### {rank}. "
                                f"{metadata.get('full_name', 'Unknown')}"
                            )

                        with col2:

                            st.metric(
                                "Final score",
                                f"{score:.3f}"
                            )

                            hybrid_score = metadata.get("hybrid_score")

                            if hybrid_score is not None:

                                st.caption(
                                    f"Original hybrid score: {hybrid_score:.3f}"
                                )

                        # Repository description
                        description = metadata.get("description")

                        st.write(
                            description or "No description available."
                        )

                        # Repository information
                        language = metadata.get("language") or "Unknown"
                        stars = metadata.get("stars", 0)

                        st.caption(
                            f"💻 {language}  |  "
                            f"⭐ {stars:,} stars"
                        )

                        # Repository topics
                        topics = metadata.get("topics", [])

                        if topics:

                            st.write(
                                " ".join(
                                    f"`{topic}`" for topic in topics[:8]
                                )
                            )

                        # GitHub link
                        url = metadata.get("url")

                        if url:

                            st.link_button(
                                "View on GitHub",
                                url
                            )

                        # Requirement evaluation
                        checks = metadata.get(
                            "requirement_checks",
                            {}
                        )

                        if requirements:

                            with st.expander(
                                "Why this repository matches"
                            ):

                                st.write(
                                    metadata.get(
                                        "requirement_evaluation",
                                        "No evaluation summary available."
                                    )
                                )

                                for requirement in requirements:

                                    requirement_id = requirement["id"]

                                    result = checks.get(
                                        requirement_id,
                                        {
                                            "status": "unknown",
                                            "evidence": (
                                                "No evaluation evidence returned."
                                            )
                                        }
                                    )

                                    status = result.get(
                                        "status",
                                        "unknown"
                                    )

                                    icons = {
                                        "match": "✅",
                                        "mismatch": "❌",
                                        "unknown": "❔"
                                    }

                                    st.markdown(
                                        f"**{icons.get(status, '❔')} "
                                        f"{requirement['description']}** "
                                        f"— `{status}`"
                                    )

                                    evidence = result.get("evidence", "")

                                    if evidence:
                                        st.caption(evidence)

        except Exception as error:

            st.error(f"Search failed: {error}")

            st.caption(
                "Check your API configuration and backend setup."
            )
