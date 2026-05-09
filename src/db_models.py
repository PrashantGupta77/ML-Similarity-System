from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime
)

from datetime import datetime

from src.database import Base


class TextEmbedding(Base):
    __tablename__ = "text_embeddings"

    id = Column(Integer, primary_key=True, index=True)

    text = Column(String, nullable=False)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )


class SearchHistory(Base):
    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, index=True)

    query = Column(String, nullable=False)

    top_result = Column(String, nullable=True)

    top_score = Column(Float, nullable=True)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )