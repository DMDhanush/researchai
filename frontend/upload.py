import streamlit as st
from auth import check_session
import time

def show():
    st.title("Upload Papers")
    if not check_session(): return
    
    uploaded_file = st.file_uploader("Upload PDF / Text file", type=['pdf', 'txt', 'md'])
    if uploaded_file and st.button("Upload"):
        with st.spinner("Processing..."):
            # Placeholder for ingestion logic invocation
            time.sleep(2)
            st.success("Uploaded!")
