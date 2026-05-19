# --------------------------------------------------------------------------------
# UI Layer
# Handling user interaction, triggering RAG pipeline, and displaying responses
# --------------------------------------------------------------------------------

import streamlit as st
import os

from services.rag_service import RAGService
from config import DATA_DIR


st.set_page_config(
    page_title="NHS Medical RAG Assistant",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-image: radial-gradient(circle, rgba(255,255,255,0.12) 1px, transparent 1px);
    background-size: 18px 18px;
    background-color: #0e1117;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Hero title */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    color: white;
    margin-bottom: 0.5rem;
}

/* Hero subtitle */
.hero-subtitle {
    font-size: 1.1rem;
    color: #b0b3b8;
    line-height: 1.7;
    margin-bottom: 2rem;
}

/* Disclaimer */
.disclaimer {
    color: #ffb3b3;
    font-size: 0.9rem;
    line-height: 1.6;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)


# Initializing RAG service for handling queries
rag_service = RAGService()


# --------------------------------------------------------------------------------
# Fetching available disease conditions from data directory
# Converting file names into readable condition names
# --------------------------------------------------------------------------------
def get_available_conditions(data_dir=DATA_DIR):
    return sorted(
        f.replace("_", " ").replace(".pdf", "")
        for f in os.listdir(data_dir)
        if f.endswith(".pdf")
    )


conditions = get_available_conditions()


# --------------------------------------------------------------------------------
# Sidebar
# Showing available dataset coverage and project limitations
# --------------------------------------------------------------------------------
with st.sidebar:
    st.header("Dataset Coverage")

    st.metric("Available Conditions", len(conditions))

    st.markdown("### Included NHS Conditions")

    for condition in conditions:
        st.markdown(f"- {condition}")

    st.divider()

    st.markdown("### System Notes")

    st.caption(
        """
        This system answers questions strictly from retrieved NHS documents.

        Some diseases or symptom combinations may not exist in the dataset.
        In those cases, the system intentionally refuses instead of guessing.
        """
    )


# --------------------------------------------------------------------------------
# Hero Section
# Introducing system purpose and grounding behavior
# --------------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-title">
        NHS Medical RAG Assistant
    </div>

    <div class="hero-subtitle">
        Ask questions about NHS medical conditions, symptoms, treatments,
        and self-care guidance using retrieved NHS documents.<br><br>

        The system uses retrieval grounding, validation guardrails,
        and refusal handling to reduce unsupported medical responses.
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------------------------------------
# Query Input Section
# User query with examples and wider layout
# --------------------------------------------------------------------------------
st.markdown("### Ask a Question")

user_query = st.text_input(
    label="",
    placeholder="Example: What are the symptoms of asthma?"
)

# --------------------------------------------------------------------------------
# Query Submission
# Running retrieval and generation pipeline with loading state
# --------------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    submit = st.button("Submit Query")

if submit and user_query:

    with st.spinner("Retrieving NHS context and validating response..."):

        try:
            response = rag_service.query(user_query)

            # Handling empty responses
            if not response or not response.strip():

                st.warning(
                    "I couldn't find enough reliable NHS information for this query."
                )

            else:
                st.markdown("### Response")

                with st.container():

                    with st.container(border=True):
                        st.markdown(response)

                    st.markdown(
                        """
                        <div class="disclaimer">
                            This response is for informational purposes only and
                            should not replace professional medical advice,
                            diagnosis, or treatment.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"Error: {str(e)}")