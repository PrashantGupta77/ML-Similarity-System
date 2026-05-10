from src.embedding_service import EmbeddingService
from src.vector_store import VectorStore

embedding_service = EmbeddingService()
vector_store = VectorStore()


def store_text(text: str, record_id: int):
    embedding = embedding_service.generate_embedding(text)

    vector_store.add_vector(
        embedding=embedding,
        text=text,
        record_id=record_id
    )

    return {"message": "Text stored successfully"}


def search_similar(query: str, top_k: int = 5):
    embedding = embedding_service.generate_embedding(query)

    return vector_store.search(
        embedding=embedding,
        top_k=top_k
    )