from sentence_transformers import SentenceTransformer
import uuid
from pathlib import Path
import os
import numpy as np
from PIL import Image
import json

from src.hybrid_weighted_fusion import HybridWeightedFusion
from src.utils import read_json
from src.chromadb_wrapper import ChromaDBHttpWrapper


def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def find_defect(defects: list[dict], label: str) -> dict | None:
    for defect in defects:
        if defect['label'] == label:
            return defect
    return None


def find_image_description(images: list[dict], img_source: str) -> str:
    for img in images:
        if img["source"] == img_source:
            return img["description"]

    return None


def embed_image_from_file(img_path: Path, image_id: str, image_metadata: list[dict],
                          documents: list, image_encoder, chroma_db: ChromaDBHttpWrapper, *,
                          to_rgb: bool = True, normalize_img: bool = True,
                          repository_name: str = 'images') -> None:
    # load the image
    image = Image.open(img_path)

    if to_rgb:
        image = image.convert('RGB')

    if normalize_img:
        img_embedding = normalize(image_encoder.encode(image))

        chroma_db.add(
            repository_name=repository_name,
            ids=[image_id],
            embeddings=[img_embedding.tolist()],
            metadatas=image_metadata,
            documents=documents  # store path for reference
        )

    else:
        img_embedding = image_encoder.encode(image)

        image_encoder.add(
            repository_name=repository_name,
            ids=[image_id],
            embeddings=[img_embedding],
            metadatas=image_metadata,
            documents=documents  # store path for reference
        )


def embed_text(text: str, text_id: str, text_encoder, chroma_db: ChromaDBHttpWrapper,
               text_metadata, documents,
               repository_name: str = 'defects_texts') -> None:
    txt_emb = normalize(text_encoder.encode(text))
    chroma_db.add(repository_name=repository_name,
                  ids=[text_id],
                  embeddings=[txt_emb],
                  metadatas=text_metadata,
                  documents=documents)


if __name__ == '__main__':
    DATA_PATH = Path('./data')

    hull_defects = read_json(DATA_PATH / "hull_defects.json")
    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)

    chromadb_wrapper.create_collection('defects_images')
    chromadb_wrapper.create_collection('defects_texts')

    # CLIP model for both text & images
    clip_model = SentenceTransformer("clip-ViT-L-14")

    defects = hull_defects['defects']

    # get the images
    dirs = os.listdir(DATA_PATH / "hull_defects_imgs")

    for dir in dirs:

        print(f'Processing dir={dir}')
        dir_path = DATA_PATH / dir

        # get the defect description e.t.c for the
        # images in this directory
        defect_info = find_defect(defects, dir)

        if not defect_info:
            print(f'No defect_info found for {dir}')
            continue

        if dir_path.is_dir():
            imgs = os.listdir(dir_path)

            # add each image to the DB
            for img in imgs:

                img_id = uuid.uuid4().hex
                img_path = dir_path / img

                image_metadata = [
                    {"label": defect_info["label"],
                     "name": defect_info["name"],
                     "synonyms": defect_info["synonyms"],
                     "material": defect_info["material"],
                     "description": defect_info["description"],
                     "visual-cues": defect_info["visual-cues"],
                     "recommendations": defect_info["recommendations"]
                     }
                ]

                images = defect_info["images"]
                image_documents = []

                if images:
                    img_description = find_image_description(images=images, img_source=dir + "/" + img)
                    image_documents.append(img_description)

                    text_meta=[{
                        "description": defect_info["description"],
                        "visual-cues": defect_info["visual-cues"],
                        "recommendations": defect_info["recommendations"]
                    }]
                    embed_text(text_id=img_id, text=img_description,
                               text_encoder=clip_model, repository_name='defects_texts',
                               chroma_db=chromadb_wrapper, documents=[],
                               text_metadata=text_meta)

                embed_image_from_file(image_id=img_id, img_path=img_path, image_encoder=clip_model,
                                      repository_name='defects_images', image_metadata=image_metadata,
                                      chroma_db=chromadb_wrapper, documents=image_documents)


