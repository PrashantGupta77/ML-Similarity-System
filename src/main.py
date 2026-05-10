from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    BackgroundTasks
)
from sqlalchemy.orm import Session

from src.models import StoreInput, SearchInput, SearchResponse
from src.retrieval import store_text, search_similar
from src.database import Base, engine, SessionLocal
from src.repository import save_text, save_search_history

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="ML Similarity System",
    version="2.0.0"
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/store")
def store(
    input_data: StoreInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):

    if not input_data.text.strip():

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    # Save in database first
    record = save_text(
        db=db,
        text=input_data.text
    )

    # Run embedding generation in background
    background_tasks.add_task(
        store_text,
        input_data.text,
        record.id
    )

    return {
        "message": "Processing started",
        "id": record.id
    }

@app.post("/search", response_model=SearchResponse)
def search(
    input_data: SearchInput,
    db: Session = Depends(get_db)
):

    if not input_data.query.strip():

        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    results = search_similar(
        db=db,
        query=input_data.query,
        top_k=input_data.top_k
    )

    if results:

        save_search_history(
            db=db,
            query=input_data.query,
            result=results[0]
        )

    return {
        "results": results
    }