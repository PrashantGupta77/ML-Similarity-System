import faiss
import numpy as np
import os
from typing import List, Dict


class VectorStore:
    def __init__(self, dim: int = 384, path: str = "faiss.index"):
        self.dim = dim
        self.path = path

        self.index = faiss.IndexFlatIP(dim)

        self.texts = []
        self.ids = []

        self._load()

    def add_vector(self, embedding: np.ndarray, text: str, record_id: int):
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)

        self.index.add(embedding)

        self.texts.append(text)
        self.ids.append(record_id)

        self._save()

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        embedding = embedding.reshape(1, -1)
        faiss.normalize_L2(embedding)

        scores, indices = self.index.search(embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "score": float(score)
            })

        return results

    def _save(self):
        faiss.write_index(self.index, self.path)

    def _load(self):
        if os.path.exists(self.path):
            self.index = faiss.read_index(self.path)