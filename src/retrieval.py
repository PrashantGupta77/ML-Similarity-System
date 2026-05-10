from sqlalchemy.orm import Session

from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore
from src.db_models import TextEmbedding

embedding_service = EmbeddingService()

vector_store = VectorStore()


def store_text(
    text: str,
    record_id: int
):

    embedding = embedding_service.generate_embedding(text)

    vector_store.add_vector(
        embedding=embedding,
        record_id=record_id
    )

    return {
        "message": "Stored successfully"
    }


def search_similar(
    db: Session,
    query: str,
    top_k: int = 5
):

    embedding = embedding_service.generate_embedding(query)

    faiss_results = vector_store.search(
        embedding=embedding,
        top_k=top_k
    )

    final_results = []

    for result in faiss_results:

        record = db.query(TextEmbedding).filter(
            TextEmbedding.id == result["id"]
        ).first()

        if record:

            final_results.append({
                "text": record.text,
                "score": result["score"]
            })

    return final_results