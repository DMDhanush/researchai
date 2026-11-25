"""ResearchAI Login and Registration Page"""
import streamlit as st
from auth import authenticate_user, register_user

def show():
    st.set_page_config(page_title="ResearchAI - Login", layout="centered")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# 📚 ResearchAI")
        st.markdown("### Academic Research Intelligence Platform")
        st.divider()

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")

            if st.button("🔐 Login", use_container_width=True):
                if not username or not password:
                    st.error("Please enter both username and password.")
                else:
                    try:
                        user = authenticate_user(username, password)
                        if user:
                            st.session_state.user_id = user.id
                            st.session_state.username = user.username
                            st.success(f"Welcome back, {username}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
                    except Exception as e:
                        st.error(f"Login failed: {str(e)}")

        with tab2:
            st.subheader("Create a New Account")
            reg_username = st.text_input("Choose a Username", key="reg_username")
            reg_email = st.text_input("Email Address", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")

            if st.button("📝 Register", use_container_width=True):
                if not all([reg_username, reg_email, reg_password, reg_confirm]):
                    st.error("Please fill in all fields.")
                elif reg_password != reg_confirm:
                    st.error("Passwords do not match.")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        new_user = register_user(reg_username, reg_email, reg_password)
                        st.success("Account created! Please log in.")
                        st.session_state.user_id = new_user.id
                        st.session_state.username = new_user.username
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Registration failed: {str(e)}")

        st.divider()
        st.markdown("### About ResearchAI")
        st.markdown("""
        ResearchAI is an academic research intelligence platform that uses:
        - **Generative AI** for intelligent analysis
        - **Retrieval-Augmented Generation (RAG)** for grounded answers
        - **Semantic Search** to find relevant research
        - **Multi-Agent Architecture** for specialized tasks

        Upload your research papers and ask questions about them in natural language.
        """)
