
import os
from pathlib import Path

import streamlit as st
from google import genai
from sentence_transformers import SentenceTransformer

from filinglens_graph import (
    EMBED_MODEL_NAME,
    build_graph,
    load_index,
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="FilingLens",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "chart_images"

st.title("📊 FilingLens")
st.caption(
    "AI-powered financial filing analysis using SEC filings, "
    "hybrid retrieval, chart analysis, and Gemini."
)

# --------------------------------------------------
# Initialize ML pipeline
# --------------------------------------------------
@st.cache_resource(show_spinner="Loading the FilingLens model and index...")
def initialize_filinglens(api_key):
    client = genai.Client(api_key=api_key)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings, entries = load_index()
    graph = build_graph(embed_model, embeddings, entries, client)

    return graph, len(entries)


# Read API key from Streamlit Cloud Secrets or environment
try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    api_key = ""

api_key = api_key or os.environ.get("GEMINI_API_KEY", "")

if not api_key:
    st.error(
        "Gemini API key is missing. Add GEMINI_API_KEY "
        "under App settings → Secrets."
    )
    st.stop()

try:
    graph, index_count = initialize_filinglens(api_key)
except Exception as exc:
    st.error("FilingLens could not initialize.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Check full_index/embeddings.npy, "
        "full_index/entries.pkl, dependencies, and available memory."
    )
    st.stop()

st.success(f"FilingLens is ready · {index_count:,} indexed entries")

# --------------------------------------------------
# Question input
# --------------------------------------------------
with st.form("question_form"):
    question = st.text_area(
        "Ask a question about a company's financial filings",
        placeholder=(
            "Examples:\n"
            "• How did Apple's revenue change between fiscal years?\n"
            "• Show Apple's stock performance chart from its 2024 10-K.\n"
            "• What risks did the company report?"
        ),
        max_chars=500,
    )

    submitted = st.form_submit_button(
        "Analyze filings",
        type="primary",
        use_container_width=False,
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def resolve_chart_path(image_reference):
    """Resolve an indexed chart reference to a local image file."""
    if not image_reference:
        return None

    reference = str(image_reference).strip()

    # Support absolute paths and project-relative paths.
    candidates = [
        Path(reference),
        BASE_DIR / reference.lstrip("/"),
    ]

    # Handle references such as /chart_images/company/file.jpg.
    normalized = reference.lstrip("/")
    if normalized.startswith("chart_images/"):
        candidates.append(BASE_DIR / normalized)

    # Handle references stored relative to the chart_images directory.
    candidates.append(CHART_DIR / normalized)

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            if resolved.is_file() and resolved.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".webp"
            }:
                return resolved
        except (OSError, ValueError):
            continue

    return None


def render_source(source, index):
    """Display source text, structured tables, and available chart images."""
    filing = source.get("filing", "Unknown filing")
    section = source.get("section", "Unknown section")
    company = source.get("company", "Unknown company")
    source_type = source.get("type", "Unknown")

    title = f"{index}. {company} — {filing} — {section}"

    with st.expander(title):
        st.caption(f"Source type: {source_type}")

        # Display chart if the retrieved entry has an image reference.
        image_reference = source.get("image")
        chart_path = resolve_chart_path(image_reference)

        if chart_path:
            st.image(
                str(chart_path),
                caption=f"{company} | {filing} | {section}",
                use_container_width=True,
            )
        elif image_reference:
            st.warning(
                f"Chart image was referenced but not found locally: "
                f"{image_reference}"
            )

        # Display the retrieved financial text.
        display_text = source.get("display_text") or source.get("text")
        if display_text:
            st.markdown("**Retrieved filing content**")
            st.write(display_text)

        # Display structured financial tables.
        table = source.get("table")
        if table:
            st.markdown("**Financial table**")
            if isinstance(table, dict):
                columns = table.get("columns")
                rows = table.get("rows")

                if columns and rows:
                    import pandas as pd

                    table_rows = []
                    for row in rows:
                        values = row.get("values", [])
                        record = {"Metric": row.get("label", "Value")}
                        for column, value in zip(columns, values):
                            record[str(column)] = value
                        table_rows.append(record)

                    if table_rows:
                        st.dataframe(
                            pd.DataFrame(table_rows),
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.json(table)
                else:
                    st.json(table)
            else:
                st.write(table)

        if not display_text and not table and not chart_path:
            st.caption("No additional source details are available.")


# --------------------------------------------------
# Run the retrieval and generation graph
# --------------------------------------------------
if submitted:
    if len(question.strip()) < 3:
        st.warning("Please enter a question with at least 3 characters.")
    else:
        try:
            with st.spinner(
                "Searching SEC filings and generating your answer..."
            ):
                result = graph.invoke({
                    "query": question.strip(),
                    "route": "",
                    "retrieved": [],
                    "answer": "",
                    "verification": [],
                    "model": "",
                })

            # Answer
            st.subheader("Answer")
            st.markdown(
                result.get("answer") or "No answer was returned."
            )

            # Retrieval metadata
            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Retrieval route",
                    result.get("route") or "Unknown",
                )

            with col2:
                st.metric(
                    "Gemini model",
                    result.get("model") or "Unknown",
                )

            # Verification
            verification = result.get("verification", [])

            if verification:
                unverified = sum(
                    1
                    for item in verification
                    if not item.get("verified", False)
                )

                st.subheader("Answer verification")
                st.metric("Unverified claims", unverified)

                with st.expander("View verification details"):
                    for item in verification:
                        verified = item.get("verified", False)

                        if verified:
                            st.success("Verified claim")
                        else:
                            st.warning("Unverified claim")

                        st.json(item)

            # Sources
            sources = result.get("retrieved", [])

            st.subheader(f"Retrieved sources ({len(sources)})")

            if not sources:
                st.info("No supporting sources were returned.")
            else:
                for i, source in enumerate(sources, start=1):
                    render_source(source, i)

            # Helpful note if no retrieved source has a chart reference.
            chart_sources = [
                source for source in sources
                if source.get("image")
            ]

            if not chart_sources:
                st.info(
                    "No chart image references were returned by retrieval "
                    "for this question. Try asking specifically about the "
                    "stock performance graph in a company's 10-K filing."
                )

        except Exception as exc:
            st.error("The question could not be processed.")
            st.code(f"{type(exc).__name__}: {exc}")
            st.info(
                "Check the terminal logs, Gemini API quota, and "
                "the retrieved index entries."
            )
