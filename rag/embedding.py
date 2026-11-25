from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class EmbeddingEngine:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)

    def encode_single(self, text: str) -> np.ndarray:
        return self.model.encode(text)
