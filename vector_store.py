import faiss, numpy as np
from typing import List, Tuple

class VectorStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.texts: List[str] = []

    def add(self, vecs: np.ndarray, texts: List[str]):
        self.index.add(vecs)
        self.texts.extend(texts)

    def search(self, vec: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        D, I = self.index.search(vec, k)
        return [(self.texts[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.texts)]