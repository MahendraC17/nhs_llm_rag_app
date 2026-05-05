# --------------------------------------------------------------------------------
# UI Layer
# Handling user interaction, triggering RAG pipeline, and displaying responses
# --------------------------------------------------------------------------------

import streamlit as st
import os

from services.rag_service import RAGService
from config import DATA_DIR

# os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]

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

st.set_page_config(page_title="NHS Disease Information Chatbot", layout="centered")
st.title("🔍 NHS Disease Information Chatbot")

# --------------------------------------------------
# Sidebar Available Conditions
# --------------------------------------------------

conditions = get_available_conditions()

with st.sidebar:
    st.header("Available Conditions")
    st.metric("Total conditions", len(conditions))

    st.markdown("**Covered diseases & conditions:**")
    for condition in conditions:
        st.markdown(f"- {condition}")

    st.caption(
        "The application only includes some diseases & conditions because scraping them was done manually and was tedious, also "
        "it serves purpose for testing and deploying with limitation of token strength for embeddings and responses."
    )

# --------------------------------------------------
# Main Query UI
# --------------------------------------------------

user_query = st.text_input("Enter your question:")

# Handling query submission and triggering pipeline
if st.button("Submit") and user_query:
    with st.spinner("Fetching answer..."):
        try:
            response = rag_service.query(user_query)

            # Handling empty responses
            if not response or not response.strip():
                st.warning(
                    "I couldn’t find anything relevant to that in the NHS documents."
                )
            else:
                st.subheader("Answer")
                st.write(response)

                st.markdown(
                    """
                    <span style='color: #ff4b4b; font-size: 0.9em;'>
                    ⚠️ This response is for informational purposes only and is not a substitute
                    for professional medical advice. Please consult a healthcare provider
                    for personal medical concerns.
                    </span>
                    """,
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")