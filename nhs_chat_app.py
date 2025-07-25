import streamlit as st
from rag_pipeline import chat_chain

st.set_page_config(page_title="Test NHS Test", layout="centered")
st.title("🔍 Test NHS Test")

user_query = st.text_input("Enter your question:")

if st.button("Submit") and user_query:
    with st.spinner("Fetching answer..."):
        try:
            response = chat_chain.invoke({"question": user_query})
            
            if not response["source_documents"]:
                print("🤖 I couldn’t find anything related to that in the NHS documents.")
            else:
                st.subheader("🧠 Answer")
                st.write(response["answer"])

                st.subheader("📄 Sources")
                st.write(response["sources"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
