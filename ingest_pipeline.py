from sentence_transformers import SentenceTransformer
import uuid
from pathlib import Path
import os
import numpy as np
from PIL import Image

from src.utils import read_json
from src.chromadb_wrapper import ChromaDBHttpWrapper


def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def find_defect(defects: list[dict], label: str) -> dict | None:
    for defect in defects:
        if defect['label'] == label:
            return defect
    return 'INVALID'


def find_image_description(images: list[dict], img_source: str) -> str:
    for img in images:
        if img["source"] == img_source:
            return img["description"]

    return 'INVALID'


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
        img_embedding = img_embedding.tolist()
    else:
        img_embedding = image_encoder.encode(image)

    chroma_db.add(
            repository_name=repository_name,
            ids=[image_id],
            embeddings=[img_embedding],
            metadatas=image_metadata,
            documents=documents
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
    IMAGES_PATH = DATA_PATH / "hull_defects_imgs"

    hull_defects = read_json(DATA_PATH / "hull_defects.json")
    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)

    chromadb_wrapper.delete_collection('defects_images')
    chromadb_wrapper.delete_collection('defects_texts')

    chromadb_wrapper.create_collection('defects_images')
    chromadb_wrapper.create_collection('defects_texts')

    # CLIP model for both text & images
    clip_model = SentenceTransformer("clip-ViT-L-14")

    defects = hull_defects['defects']

    # get the images
    dirs = os.listdir(IMAGES_PATH)

    for dir in dirs:

        print(f'Processing dir={dir}')
        dir_path = IMAGES_PATH / dir

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

                recommendations = ",".join(defect_info["recommendations"]) if defect_info[
                                                                                  "recommendations"] is not None else "INVALID"

                synonyms = ",".join(defect_info["synonyms"]) if defect_info["synonyms"] else "INVALID"
                material = ",".join(defect_info["material"]) if defect_info["material"] else "INVALID"

                image_metadata = [
                    {"label": defect_info["label"],
                     "name": defect_info["name"],
                     "synonyms": synonyms,
                     "material": material,
                     "description": defect_info["description"] if defect_info["description"] else "INVALID",
                     "visual-cues": defect_info["visual-cues"] if defect_info["visual-cues"] else "INVALID",
                     "recommendations": recommendations
                     }
                ]

                images = defect_info["images"]
                image_documents = []

                if images:
                    img_description = find_image_description(images=images, img_source=dir + "/" + img)
                    image_documents.append(img_description)
                    text_data = {
                            "description": defect_info["description"] if defect_info["description"] else "INVALID",
                            "visual-cues": defect_info["visual-cues"] if defect_info["visual-cues"] else "INVALID",
                            "recommendations": recommendations
                    }

                    text_meta = [text_data]
                    embed_text(text_id=img_id, text=img_description,
                               text_encoder=clip_model, repository_name='defects_texts',
                               chroma_db=chromadb_wrapper, documents=[img_description],
                               text_metadata=text_meta)

                embed_image_from_file(image_id=img_id, img_path=img_path, image_encoder=clip_model,
                                      repository_name='defects_images', image_metadata=image_metadata,
                                      chroma_db=chromadb_wrapper, documents=image_documents)
