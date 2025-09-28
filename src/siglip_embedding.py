import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModel
from .vision_embedding_base import VisionEmbeddingBase


class SigLipEmbeddings(VisionEmbeddingBase):
    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__()
        self.model_name = model_name
        self.preprocess = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    @property
    def embedder_id(self) -> str:
        return "SigLipEmbeddings"

    def embed_image(self, path: Path) -> list[float]:
        image = Image.open(path)
        image = image.convert('RGB')
        image_inputs = self.preprocess(images=image, return_tensors="pt")
        with torch.no_grad():
            embeddings = self.model.get_image_features(**image_inputs)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings[0].cpu().numpy()
