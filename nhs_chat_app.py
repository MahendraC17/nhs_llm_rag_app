import streamlit as st
from rag_pipeline import qa_chain

st.set_page_config(page_title="NHS Health A -Z", layout="centered")
st.title("🔍 Ask Your Medical Question")

user_query = st.text_input("Enter your question:")

if st.button("Submit") and user_query:
    with st.spinner("Fetching answer..."):
        try:
            response = qa_chain.invoke({"question": user_query})
            st.subheader("🧠 Answer")
            st.write(response["answer"])

            st.subheader("📄 Sources")
            st.write(response["sources"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
