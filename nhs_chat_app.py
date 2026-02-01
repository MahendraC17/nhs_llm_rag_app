import os
import streamlit as st
from rag_pipeline import chat_chain
from config.settings import Settings

settings = Settings()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def get_available_conditions(data_dir=settings.data_dir):
    return sorted(
        f.replace("_", " ").replace(".pdf", "")
        for f in os.listdir(data_dir)
        if f.endswith(".pdf")
    )

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(page_title="NHS Disease Information Chatbot", layout="centered")
st.title("🔍 NHS Disease Information Chatbot")

# --------------------------------------------------
# Sidebar: Available Conditions
# --------------------------------------------------
conditions = get_available_conditions()

with st.sidebar:
    st.header("📊 Available Conditions")
    st.metric("Total conditions", len(conditions))

    st.markdown("**Covered diseases / conditions:**")
    for condition in conditions:
        st.markdown(f"- {condition}")

# --------------------------------------------------
# Main Query UI
# --------------------------------------------------
user_query = st.text_input("Enter your question:")

if st.button("Submit") and user_query:
    with st.spinner("Fetching answer..."):
        try:
            response = chat_chain.invoke(user_query)

            if not response or not response.strip():
                st.warning(
                    "🤖 I couldn’t find anything relevant to that in the NHS documents."
                )
            else:
                st.subheader("🧠 Answer")
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
