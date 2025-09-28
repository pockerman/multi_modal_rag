import uuid
from typing import Callable
from pathlib import Path
import os
import numpy as np
import json

from src.utils import read_json
from src.db_models import ImageModel, DefectTextModel
from src.chromadb_wrapper import ChromaDBHttpWrapper
from src.openclip_embedding import OpenCipEmbeddings
from src.sentence_transformer_embedding import SentenceTransformerEmbeddings
from src.siglip_embedding import SigLipEmbeddings


def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def embed_image_from_file(img_data: dict, image_encoder, chroma_db: ChromaDBHttpWrapper, *,
                          repository_name: str) -> str:
    img_model = ImageModel.create_from(data=img_data,
                                       image_encoder=image_encoder)

    chroma_db.add(
        repository_name=repository_name,
        ids=[img_model.uid],
        embeddings=[img_model.img_embedding],
        metadatas={'defect_label': img_model.defect_label, 'severity': img_model.severity,
                   'hull_material': img_model.hull_material},
        documents=[img_model.description]
    )

    return img_model.uid


if __name__ == '__main__':

    DATA_PATH = Path('./data')
    IMAGES_PATH = DATA_PATH / "hull_defects_imgs"

    hull_defects = read_json(DATA_PATH / "hull_defects_short.json")

    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)
    embeders = [OpenCipEmbeddings(device='cpu'),
                SentenceTransformerEmbeddings(model_name="clip-ViT-L-14"),
                SigLipEmbeddings(model_name="google/siglip-base-patch16-224")]

    chroma_db_repos = ['defects_images_' + embeder.embedder_id
                       for embeder in embeders]

    # access all the defects
    defects = hull_defects['defects']

    for repo, embeder in zip(chroma_db_repos, embeders):
        try:
            chromadb_wrapper.delete_collection(repo)
        except Exception as e:
            print(f'Repository name {repo} was not found skipping...')

        print(f'Embedder id ={embeder.embedder_id}')
        chromadb_wrapper.create_collection(repo)

        # get the directories for hull_defects_imgs
        dirs = os.listdir(IMAGES_PATH)
        for defect in defects:

            # get the data first access the label
            # as the directories are named as such
            label = defect['label']

            print(f'Processing dir={label}')
            dir_path = IMAGES_PATH / label

            if not dir_path.is_dir():
                print(f'No defect_info found for {label} embedding just the text')
                continue

            synonyms = defect['synonyms']
            material = defect['material']
            defect_description = defect['description']
            images = defect['images']

            if images:

                image_ids = []
                for img in images:
                    variant = img['variant']
                    source = img['source']
                    severity = img['severity']
                    img_description = img['description']

                    img_data = {
                        'img_path': IMAGES_PATH / source,
                        'variant': variant,
                        'source': source,
                        'severity': severity,
                        'label': label,
                        'description': img_description
                    }

                    img_id = embed_image_from_file(img_data=img_data, image_encoder=embeder,
                                                   chroma_db=chromadb_wrapper,
                                                   repository_name=repo,
                                                   )

                    image_ids.append(img_id)
