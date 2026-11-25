import streamlit as st
from auth import check_session

def show():
    st.title("Dashboard")
    user = check_session()
    if not user: return
    st.write(f"Welcome, {user.username}!")
    st.metric("Papers", 0)
    st.metric("Analyses", 0)
