from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

    def generate_embedding(
        self,
        text: str
    ):

        embedding = self.model.encode(
            text,
            convert_to_numpy=True
        )

        return np.array(
            embedding,
            dtype=np.float32
        )