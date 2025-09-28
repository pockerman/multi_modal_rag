from sentence_transformers import SentenceTransformer
from pathlib import Path
from PIL import Image

from .vision_embedding_base import VisionEmbeddingBase


class SentenceTransformerEmbeddings(VisionEmbeddingBase):

    def __init__(self, model_name: str="clip-ViT-L-14"):
        super().__init__()
        self.model = SentenceTransformer(model_name)

    @property
    def embedder_id(self) -> str:
        return "SentenceTransformerEmbeddings"

    def embed_image(self, path: Path) -> list[float]:
        image = Image.open(path)
        image = image.convert('RGB')
        img_embedding = VisionEmbeddingBase.normalize(self.model.encode(image))
        img_embedding = img_embedding.tolist()
        return img_embedding

