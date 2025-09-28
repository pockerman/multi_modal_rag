import torch
from pathlib import Path
from PIL import Image
import open_clip

from .vision_embedding_base import VisionEmbeddingBase


class OpenCipEmbeddings(VisionEmbeddingBase):
    def __init__(self, *, model_name: str = 'ViT-H-14',
                 pretrained: str = 'laion2b_s32b_b79k', device: str = 'cuda'):
        super().__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.device = device
        self.model = self.model.to(device)

    @property
    def embedder_id(self) -> str:
        return "OpenCipEmbeddings"

    def embed_image(self, path: Path) -> list[float]:
        image = Image.open(path)
        image = image.convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_image(image)
        feats /= feats.norm(dim=-1, keepdim=True)  # normalize
        return feats.cpu().numpy()[0]
