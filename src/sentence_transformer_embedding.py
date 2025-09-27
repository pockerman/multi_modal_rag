from sentence_transformers import SentenceTransformer
from pathlib import Path
import numpy as np
from PIL import Image


class SentenceTransformerEmbeddings:

    @staticmethod
    def normalize(v):
        v = np.array(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-10)

    def __init__(self, model_name: str="clip-ViT-L-14"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, path: Path) -> list[float]:
        image = Image.open(path)
        image = image.convert('RGB')
        img_embedding = SentenceTransformerEmbeddings.normalize(self.model.encode(image))
        img_embedding = img_embedding.tolist()
        return img_embedding

