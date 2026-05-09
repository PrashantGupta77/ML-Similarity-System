from fastapi import FastAPI, HTTPException, Depends
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
def store(input_data: StoreInput, db: Session = Depends(get_db)):

    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # 1. Save in DB first
    record = save_text(db=db, text=input_data.text)

    # 2. Save in vector store with SAME ID
    store_text(
        text=input_data.text,
        record_id=record.id
    )

    return {"message": "Stored successfully", "id": record.id}


@app.post("/search", response_model=SearchResponse)
def search(input_data: SearchInput, db: Session = Depends(get_db)):

    if not input_data.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    results = search_similar(
        query=input_data.query,
        top_k=input_data.top_k
    )

    if results:
        save_search_history(
            db=db,
            query=input_data.query,
            result=results[0]
        )

    return {"results": results}