import faiss
import numpy as np
import os
from typing import List, Dict


class VectorStore:

    def __init__(
        self,
        dim: int = 384,
        path: str = "faiss.index"
    ):

        self.dim = dim
        self.path = path

        # Store vectors with custom IDs
        self.index = faiss.IndexIDMap(
            faiss.IndexFlatIP(dim)
        )

        self._load()

    def add_vector(
        self,
        embedding: np.ndarray,
        record_id: int
    ):

        embedding = embedding.reshape(1, -1)

        # Normalize for cosine similarity
        faiss.normalize_L2(embedding)

        # Prevent duplicate IDs
        existing_ids = faiss.vector_to_array(
            self.index.id_map
        )

        if record_id in existing_ids:
            return

        ids = np.array(
            [record_id],
            dtype=np.int64
        )

        self.index.add_with_ids(
            embedding,
            ids
        )

        self._save()

    def search(
        self,
        embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict]:

        if self.index.ntotal == 0:
            return []

        embedding = embedding.reshape(1, -1)

        # Normalize query embedding
        faiss.normalize_L2(embedding)

        scores, ids = self.index.search(
            embedding,
            top_k
        )

        results = []

        for score, record_id in zip(
            scores[0],
            ids[0]
        ):

            if record_id == -1:
                continue

            results.append({
                "id": int(record_id),
                "score": float(score)
            })

        return results

    def _save(self):

        faiss.write_index(
            self.index,
            self.path
        )

    def _load(self):

        if os.path.exists(self.path):

            self.index = faiss.read_index(
                self.path
            )