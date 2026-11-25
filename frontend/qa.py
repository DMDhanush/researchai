import streamlit as st
from auth import check_session
from rag.qa_engine import QAEngine

def show():
    st.title("Ask ResearchAI")
    if not check_session(): return
    
    question = st.text_input("Ask a question about your research literature")
    if st.button("Ask"):
        with st.spinner("Analyzing..."):
            qa = QAEngine()
            result = qa.ask(question, st.session_state.user_id)
            st.write(result['answer'])
            st.write("Sources:", result['sources'])
