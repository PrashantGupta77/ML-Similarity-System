import faiss
import numpy as np
import os
import pickle
from typing import List, Dict


class VectorStore:
    def __init__(self, dim: int = 384, path: str = "faiss.index"):
        self.dim = dim
        self.path = path

        # metadata file
        self.meta_path = "metadata.pkl"

        self.index = faiss.IndexFlatIP(dim)

        self.texts = []
        self.ids = []

        self._load()

    def add_vector(self, embedding: np.ndarray, text: str, record_id: int):

        embedding = embedding.reshape(1, -1).astype("float32")

        faiss.normalize_L2(embedding)

        self.index.add(embedding)

        self.texts.append(text)
        self.ids.append(record_id)

        self._save()

    def search(self, embedding: np.ndarray, top_k: int = 5) -> List[Dict]:

        # safe empty search
        if self.index.ntotal == 0:
            return []

        # avoid top_k issues
        top_k = min(top_k, self.index.ntotal)

        embedding = embedding.reshape(1, -1).astype("float32")

        faiss.normalize_L2(embedding)

        scores, indices = self.index.search(embedding, top_k)

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            # protection against mismatch
            if idx >= len(self.ids):
                continue

            results.append({
                "id": self.ids[idx],
                "text": self.texts[idx],
                "score": float(score)
            })

        return results

    def _save(self):

        # save FAISS index
        faiss.write_index(self.index, self.path)

        # save metadata
        with open(self.meta_path, "wb") as f:
            pickle.dump({
                "texts": self.texts,
                "ids": self.ids
            }, f)

    def _load(self):

        # load FAISS index
        if os.path.exists(self.path):
            self.index = faiss.read_index(self.path)

        # load metadata if exists
        if os.path.exists(self.meta_path):

            with open(self.meta_path, "rb") as f:

                metadata = pickle.load(f)

                self.texts = metadata.get("texts", [])
                self.ids = metadata.get("ids", [])

        # safety check
        if self.index.ntotal != len(self.ids):

            print("Index mismatch detected. Resetting vector store.")

            self.index = faiss.IndexFlatIP(self.dim)

            self.texts = []
            self.ids = []

            self._save()