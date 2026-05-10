from sqlalchemy.orm import Session

from src.db_models import (
    TextEmbedding,
    SearchHistory
)


def save_text(
    db: Session,
    text: str
):

    # Prevent duplicate text storage
    existing = db.query(TextEmbedding).filter(
        TextEmbedding.text == text
    ).first()

    if existing:
        return existing

    record = TextEmbedding(text=text)

    db.add(record)

    db.commit()

    db.refresh(record)

    return record


def save_search_history(
    db: Session,
    query: str,
    result: dict
):

    history = SearchHistory(
        query=query,
        top_result=result.get("text"),
        top_score=result.get("score")
    )

    db.add(history)

    db.commit()