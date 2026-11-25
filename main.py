import streamlit as st
from frontend import login, dashboard, upload, library, qa, summarizer, comparison, gaps, literature, insights, evaluation, settings

def main():
    st.set_page_config(
        page_title="ResearchAI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None
    )

    # Dark theme CSS
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'conversation_id' not in st.session_state:
        st.session_state.conversation_id = None

    # Check authentication
    if not st.session_state.user_id:
        login.show()
    else:
        # Authenticated - show sidebar navigation
        with st.sidebar:
            st.title("📚 ResearchAI")
            st.divider()

            page = st.radio("Navigation", [
                "Dashboard",
                "Upload Papers",
                "Paper Library",
                "Ask ResearchAI",
                "Paper Summarizer",
                "Compare Papers",
                "Research Gaps",
                "Literature Review",
                "Research Insights",
                "Evaluation",
                "Settings",
                "Logout"
            ])

            st.divider()
            st.caption(f"Logged in as: {st.session_state.username}")

        if page == "Dashboard":
            dashboard.show()
        elif page == "Upload Papers":
            upload.show()
        elif page == "Paper Library":
            library.show()
        elif page == "Ask ResearchAI":
            qa.show()
        elif page == "Paper Summarizer":
            summarizer.show()
        elif page == "Compare Papers":
            comparison.show()
        elif page == "Research Gaps":
            gaps.show()
        elif page == "Literature Review":
            literature.show()
        elif page == "Research Insights":
            insights.show()
        elif page == "Evaluation":
            evaluation.show()
        elif page == "Settings":
            settings.show()
        elif page == "Logout":
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.conversation_id = None
            st.rerun()

if __name__ == "__main__":
    main()
