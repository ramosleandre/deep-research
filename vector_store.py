import faiss, numpy as np
from typing import List, Tuple

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []

    def add(self, vecs: np.ndarray, texts: List[str]):
        self.index.add(vecs)
        self.texts.extend(texts)

    def search(self, vector: np.ndarray, k: int = 10):
        D, I = self.index.search(vector, k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            # Faiss renvoie -1 si aucun voisin → on ignore
            if idx == -1 or idx >= len(self.texts):
                continue
            results.append((self.texts[idx], float(dist)))
        return results