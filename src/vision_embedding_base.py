from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np


class VisionEmbeddingBase(ABC):

    @staticmethod
    def normalize(v):
        v = np.array(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-10)

    def __init__(self):
        self._model = None

    @abstractmethod
    def embed_image(self, path: Path) -> list[float]:
        pass
