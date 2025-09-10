from typing import Self, Callable, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from .vector_db_protocol import VectorDBProtocol


class HybridWeightedFusion:
    def __init__(self):
        pass

    def query(self):
        pass

    def embed_image_from_file(self, img_path: Path, image_id: str, image_metadata: list[dict],
                              documents: list,
                              image_encoder: Callable,
                              db_protocol: VectorDBProtocol, *,
                              to_rgb: bool = False,
                              normalizer: Optional[Callable]=None,
                              repository_name: str='images') -> Self:

        # load the image
        image = Image.open(img_path)

        if to_rgb:
            image = image.convert('RGB')

        if normalizer:
            img_embedding = normalizer(image_encoder.encode(image))

            db_protocol.add(
                repository_name=repository_name,
                ids=[image_id],
                embeddings=[img_embedding.tolist()],
                metadatas=image_metadata,
                documents=documents  # store path for reference
            )

        else:
            img_embedding = image_encoder.encode(image)

            db_protocol.add(
                repository_name=repository_name,
                ids=[image_id],
                embeddings=[img_embedding],
                metadatas=image_metadata,
                documents=documents  # store path for reference
            )

        return self