"""
ResearchAI — SQLAlchemy 2.x Database Models
"""

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    String, Integer, Float, Text, Boolean, DateTime,
    ForeignKey, func, Index
)
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, relationship
)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    salt: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    papers: Mapped[List["Paper"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["PaperChunk"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    questions: Mapped[List["ResearchQuestion"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    gap_records: Mapped[List["ResearchGapRecord"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    comparison_records: Mapped[List["ComparisonRecord"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    literature_reviews: Mapped[List["LiteratureReviewRecord"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    evaluation_records: Mapped[List["EvaluationRecord"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    agent_logs: Mapped[List["AgentLog"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    authors: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    abstract: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pub_year: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    keywords: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    full_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, onupdate=func.now(), nullable=True)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="papers")
    pages: Mapped[List["PaperPage"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    chunks: Mapped[List["PaperChunk"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    analysis_results: Mapped[List["AnalysisResult"]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_papers_user_status", "user_id", "status"),
    )


class PaperPage(Base):
    __tablename__ = "paper_pages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    word_count: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    paper: Mapped["Paper"] = relationship(back_populates="pages")

    __table_args__ = (
        Index("ix_paper_pages_paper_page", "paper_id", "page_number"),
    )


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    paper: Mapped["Paper"] = relationship(back_populates="chunks")
    user: Mapped["User"] = relationship(back_populates="chunks")

    __table_args__ = (
        Index("ix_paper_chunks_user_paper", "user_id", "paper_id"),
    )


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    paper_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=True)
    analysis_type: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="analysis_results")
    paper: Mapped[Optional["Paper"]] = relationship(back_populates="analysis_results")


class ResearchQuestion(Base):
    __tablename__ = "research_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sources: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    confidence: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    paper_scope: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON list of paper IDs
    conversation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="questions")
    retrieval_logs: Mapped[List["RetrievalLog"]] = relationship(
        back_populates="question", cascade="all, delete-orphan"
    )


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[int] = mapped_column(Integer, ForeignKey("research_questions.id", ondelete="CASCADE"), nullable=False, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_count: Mapped[int] = mapped_column(Integer, default=0)
    retrieved_count: Mapped[int] = mapped_column(Integer, default=0)
    top_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    question: Mapped["ResearchQuestion"] = relationship(back_populates="retrieval_logs")


class ResearchGapRecord(Base):
    __tablename__ = "research_gap_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    paper_ids: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    gaps: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="gap_records")


class ComparisonRecord(Base):
    __tablename__ = "comparison_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    paper_ids: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    comparison: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="comparison_records")


class LiteratureReviewRecord(Base):
    __tablename__ = "literature_review_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    paper_ids: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    review_type: Mapped[str] = mapped_column(String(20), nullable=False)  # short, medium, detailed
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="literature_reviews")


class EvaluationRecord(Base):
    __tablename__ = "evaluation_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="evaluation_records")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    input_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_data: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    user: Mapped["User"] = relationship(back_populates="agent_logs")
