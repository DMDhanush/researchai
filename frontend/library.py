"""ResearchAI Paper Library"""
import streamlit as st
from database.database import get_session
from database.models import Paper, PaperChunk
import os

def show():
    st.title("📚 Paper Library")

    user_id = st.session_state.user_id

    try:
        with get_session() as db:
            papers = db.query(Paper).filter(Paper.user_id == user_id).order_by(Paper.created_at.desc()).all()

        if not papers:
            st.info("No papers uploaded yet. Start by uploading a research paper!")
            return

        st.subheader(f"Your Papers ({len(papers)})")

        # Filters
        col1, col2 = st.columns(2)
        with col1:
            status_filter = st.selectbox("Filter by status", ["All", "ready", "processing", "failed", "pending"])
        with col2:
            sort_by = st.selectbox("Sort by", ["Newest", "Oldest", "Title A-Z"])

        # Filter papers
        filtered_papers = papers
        if status_filter != "All":
            filtered_papers = [p for p in papers if p.status == status_filter]

        # Sort papers
        if sort_by == "Oldest":
            filtered_papers.sort(key=lambda p: p.created_at)
        elif sort_by == "Title A-Z":
            filtered_papers.sort(key=lambda p: p.title or p.filename)

        st.divider()

        # Display papers
        for paper in filtered_papers:
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 1, 1])

                with col1:
                    st.subheader(paper.title or paper.filename)
                    if paper.authors:
                        st.caption(f"👥 {paper.authors}")
                    cols = st.columns(4)
                    with cols[0]:
                        st.caption(f"📅 {paper.pub_year or 'Year unknown'}")
                    with cols[1]:
                        st.caption(f"📑 {paper.page_count} pages")
                    with cols[2]:
                        st.caption(f"📦 {paper.chunk_count} chunks")
                    with cols[3]:
                        if paper.status == "ready":
                            st.caption("✅ Ready")
                        elif paper.status == "processing":
                            st.caption("⏳ Processing")
                        elif paper.status == "failed":
                            st.caption("❌ Failed")
                        else:
                            st.caption("⏸️ Pending")

                with col2:
                    if st.button("🔄 Re-index", key=f"reindex_{paper.id}"):
                        reindex_paper(paper.id, user_id)

                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{paper.id}"):
                        delete_paper(paper.id, user_id)

                # Show abstract if available
                if paper.abstract:
                    with st.expander("📝 Abstract"):
                        st.write(paper.abstract)

                if paper.error_message and paper.status == "failed":
                    st.error(f"Error: {paper.error_message}")

    except Exception as e:
        st.error(f"Error loading library: {str(e)}")

def reindex_paper(paper_id: int, user_id: int):
    """Re-index a paper's vectors"""
    try:
        from rag.embedding import EmbeddingEngine
        from rag.vector_store import VectorStore

        with get_session() as db:
            chunks = db.query(PaperChunk).filter(
                PaperChunk.paper_id == paper_id,
                PaperChunk.user_id == user_id
            ).all()

            if not chunks:
                st.error("No chunks found for this paper.")
                return

            # Re-embed and re-index
            embedder = EmbeddingEngine()
            chunk_texts = [c.content for c in chunks]
            embeddings = embedder.encode(chunk_texts)

            vdb = VectorStore()
            chunk_dicts = [{
                'paper_id': c.paper_id,
                'user_id': c.user_id,
                'chunk_index': c.chunk_index,
                'page_number': c.page_number,
                'section': c.section,
                'content': c.content,
                'word_count': c.word_count
            } for c in chunks]
            vdb.add_chunks(chunk_dicts, embeddings)

        st.success(f"✅ Re-indexed {len(chunks)} chunks successfully!")
    except Exception as e:
        st.error(f"Error re-indexing: {str(e)}")

def delete_paper(paper_id: int, user_id: int):
    """Delete a paper and all associated data"""
    try:
        with get_session() as db:
            paper = db.query(Paper).filter(
                Paper.id == paper_id,
                Paper.user_id == user_id
            ).first()

            if not paper:
                st.error("Paper not found.")
                return

            # Delete file
            if os.path.exists(paper.file_path):
                os.remove(paper.file_path)

            # Delete from database (cascade will handle chunks, pages, etc)
            db.delete(paper)
            db.commit()

        st.success(f"✅ Deleted '{paper.title or paper.filename}' and all associated data.")
        st.rerun()
    except Exception as e:
        st.error(f"Error deleting paper: {str(e)}")
