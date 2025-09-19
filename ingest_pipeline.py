from sentence_transformers import SentenceTransformer
import uuid
from typing import Callable
from pathlib import Path
import os
import numpy as np
import json

from src.utils import read_json
from src.db_models import ImageModel, DefectTextModel
from src.chromadb_wrapper import ChromaDBHttpWrapper


def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def embed_image_from_file(img_data: dict, image_encoder, chroma_db: ChromaDBHttpWrapper, *,
                          to_rgb: bool = True, normalize: Callable = normalize,
                          repository_name: str = 'images') -> str:
    img_model = ImageModel.create_from(data=img_data,
                                       image_encoder=image_encoder,
                                       to_rgb=to_rgb, normalize=normalize)

    chroma_db.add(
        repository_name=repository_name,
        ids=[img_model.uid],
        embeddings=[img_model.img_embedding],
        metadatas={'defect_label': img_model.defect_label, 'severity': img_model.severity,
                   'hull_material': img_model.hull_material},
        documents=[img_model.description]
    )

    return img_model.uid


def embed_text(text_db_model: DefectTextModel, chroma_db: ChromaDBHttpWrapper, repository_name: str) -> None:
    stringify_recommendations = json.dumps(text_db_model.recommendations)

    text_metadata = {

        'recommendations': stringify_recommendations,
        'label': text_db_model.defect_label

    }
    documents = [text_db_model.text_description]
    chroma_db.add(repository_name=repository_name,
                  ids=[text_db_model.uid],
                  embeddings=[text_db_model.text_embedding],
                  metadatas=text_metadata,
                  documents=documents)


def create_faqs(defects: list[dict], faqs_repo: str, chroma_db: ChromaDBHttpWrapper, text_encoder) -> None:
    for defect in defects:
        label = defect['label']
        faqs = defect['faqs'][0]

        # description faq
        defect_description_q = faqs['description']
        documents = [defect_description_q]
        metadata = {
            'label': defect['label'],
            'answer': defect['description']
        }

        txt_emb = normalize(text_encoder.encode(defect_description_q))
        chroma_db.add(repository_name=faqs_repo,
                      ids=[uuid.uuid4().hex],
                      embeddings=[txt_emb],
                      metadatas=metadata,
                      documents=documents)

        # common-causes faq
        common_causes_q = faqs['common-causes']
        documents = [common_causes_q]

        common_causes = defect['common-causes']
        if not common_causes:
            answer = f"No information is available to answer this question"
        else:

            # loop over the variants
            answer = f"The causes of {label} depend on the hull material."
            for cause in common_causes:
                variant = cause['variant']
                causes = ".".join(cause['causes'])
                answer += f"For a hull made of {variant} some causes are: {causes}."

        metadata = {
            'label': defect['label'],
            'answer': answer
        }

        txt_emb = normalize(text_encoder.encode(common_causes_q))
        chroma_db.add(repository_name=faqs_repo,
                      ids=[uuid.uuid4().hex],
                      embeddings=[txt_emb],
                      metadatas=metadata,
                      documents=documents)

        # visual-cues
        visual_queues_q = faqs['visual-cues']
        documents = [visual_queues_q]

        visual_queues = defect['visual-cues']
        if not visual_queues:
            answer = f"No information is available to answer this question"
        else:
            # loop over the variants
            answer = f"The visual cues of {label} depend on the hull material."
            for cues in visual_queues:
                variant = cues['variant']
                cues = ".".join(cues['cues'])
                answer += f"For a hull made of {variant} some cues are: {cues}."

        metadata = {
            'label': defect['label'],
            'answer': answer
        }

        txt_emb = normalize(text_encoder.encode(visual_queues_q))
        chroma_db.add(repository_name=faqs_repo,
                      ids=[uuid.uuid4().hex],
                      embeddings=[txt_emb],
                      metadatas=metadata,
                      documents=documents)

        # inspection notes
        inspection_notes_q = faqs['inspection-notes']
        documents = [inspection_notes_q]

        inspection_notes = defect['inspection-notes']
        if not inspection_notes:
            answer = f"No information is available to answer this question"
        else:
            # loop over the variants
            answer = f"The visual cues of {label} depend on the hull material."
            for note in inspection_notes:
                variant = note['variant']
                notes = ".".join(note['notes'])
                answer += f"For a hull made of {variant} some inspection guidelines are: {notes}."

        metadata = {
            'label': defect['label'],
            'answer': answer
        }

        txt_emb = normalize(text_encoder.encode(inspection_notes_q))
        chroma_db.add(repository_name=faqs_repo,
                      ids=[uuid.uuid4().hex],
                      embeddings=[txt_emb],
                      metadatas=metadata,
                      documents=documents)

        # recommendations
        recommendations_q = faqs['recommendations']
        documents = [recommendations_q]

        recommendations = defect['recommendations']
        if not recommendations:
            answer = f"No information is available to answer this question"
        else:
            # loop over the variants
            answer = (f"The recommendations for addressing defect {label} depend on the hull material and the extent "
                      f"of the defect.")
            for recommendation in recommendations:
                variant = recommendation['variant']
                minor_actions: list[str] = recommendation['minor']['actions']
                moderate_actions: list[str] = recommendation['moderate']['actions']
                severe_actions: list[str] = recommendation['severe']['actions']

                answer += f"For a hull made of {variant} and minor defects some recommendations are: {",".join(minor_actions)}."
                answer += f"For a hull made of {variant} and moderate defects some recommendations are: {",".join(moderate_actions)}."
                answer += f"For a hull made of {variant} and severe defects some recommendations are: {",".join(severe_actions)}."

        metadata = {
            'label': defect['label'],
            'answer': answer
        }

        txt_emb = normalize(text_encoder.encode(inspection_notes_q))
        chroma_db.add(repository_name=faqs_repo,
                      ids=[uuid.uuid4().hex],
                      embeddings=[txt_emb],
                      metadatas=metadata,
                      documents=documents)

        q = f"what are the criteria for the {label} to be minor?"
        if not recommendations:
            answer = f"No information is available to answer this question"
        else:
            answer = (f"The criteria for classifying defect {label} as minor depend on the hull material"
                      f"of the defect.")

            for recommendation in recommendations:
                variant = recommendation['variant']
                minor_criteria: list[str] = recommendation['minor']['criteria']
                answer += f"For a hull made of {variant} some criteria are: {",".join(minor_criteria)}."

        txt_emb = normalize(text_encoder.encode(q))
        metadata = {
            'label': defect['label'],
            'answer': answer
        }
        chroma_db.add(repository_name=faqs_repo,
                      ids=[uuid.uuid4().hex],
                      embeddings=[txt_emb],
                      metadatas=metadata,
                      documents=documents)

        q = f"what are the criteria for the {label} to be moderate?"
        if not recommendations:
            answer = f"No information is available to answer this question"
        else:
            answer = (f"The criteria for classifying defect {label} as moderate depend on the hull material"
                      f"of the defect.")

            for recommendation in recommendations:
                variant = recommendation['variant']
                moderate_criteria: list[str] = recommendation['moderate']['criteria']
                answer += f"For a hull made of {variant} some criteria are: {",".join(moderate_criteria)}."

        txt_emb = normalize(text_encoder.encode(q))
        metadata = {
            'label': defect['label'],
            'answer': answer
        }
        chroma_db.add(repository_name=faqs_repo,
                      ids=[uuid.uuid4().hex],
                      embeddings=[txt_emb],
                      metadatas=metadata,
                      documents=documents)

        q = f"what are the criteria for the {label} to be severe?"
        if not recommendations:
            answer = f"No information is available to answer this question"
        else:
            answer = (f"The criteria for classifying defect {label} as moderate depend on the hull material"
                      f"of the defect.")

            for recommendation in recommendations:
                variant = recommendation['variant']
                moderate_criteria: list[str] = recommendation['severe']['criteria']
                answer += f"For a hull made of {variant} some criteria are: {",".join(moderate_criteria)}."

        txt_emb = normalize(text_encoder.encode(q))
        metadata = {
            'label': defect['label'],
            'answer': answer
        }
        chroma_db.add(repository_name=faqs_repo,
                      ids=[uuid.uuid4().hex],
                      embeddings=[txt_emb],
                      metadatas=metadata,
                      documents=documents)


if __name__ == '__main__':

    FAQS_REPO: str = 'faqs_repo'
    IMAGES_REPO: str = 'defects_images'
    DEFECTS_REPO_TEXT: str = 'defects_texts'

    DATA_PATH = Path('./data')
    IMAGES_PATH = DATA_PATH / "hull_defects_imgs"

    hull_defects = read_json(DATA_PATH / "hull_defects_short.json")
    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)

    chromadb_wrapper.delete_collection(FAQS_REPO)
    chromadb_wrapper.delete_collection(IMAGES_REPO)
    chromadb_wrapper.delete_collection(DEFECTS_REPO_TEXT)

    chromadb_wrapper.create_collection(FAQS_REPO)
    chromadb_wrapper.create_collection(IMAGES_REPO)
    chromadb_wrapper.create_collection(DEFECTS_REPO_TEXT)

    # CLIP model for both text & images
    clip_model = SentenceTransformer("clip-ViT-L-14")

    # access all the defects
    defects = hull_defects['defects']

    create_faqs(defects=defects, faqs_repo=FAQS_REPO,
                chroma_db=chromadb_wrapper, text_encoder=clip_model)

    # get the directories for images
    dirs = os.listdir(IMAGES_PATH)

    for defect in defects:

        # get the data first access the label
        # as the directories are named as such
        label = defect['label']

        print(f'Processing dir={label}')
        dir_path = IMAGES_PATH / label

        if not dir_path.is_dir():
            print(f'No defect_info found for {label} embedding just the text')

            # if we have no image we will just embed the text
            defect_model = DefectTextModel.create_from(data=defect, text_encoder=clip_model,
                                                       normalize=normalize)

            # just embed the text
            embed_text(defect_model, chroma_db=chromadb_wrapper, repository_name=DEFECTS_REPO_TEXT, )
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

                img_id = embed_image_from_file(img_data=img_data, image_encoder=clip_model, to_rgb=True,
                                               chroma_db=chromadb_wrapper, repository_name=IMAGES_REPO,
                                               normalize=normalize
                                               )

                image_ids.append(img_id)
            # we need to associate the image with a description of the defect
            defect['id'] = ";".join(image_ids)
            text_db_model = DefectTextModel.create_from(data=defect, text_encoder=clip_model, normalize=normalize)
            embed_text(text_db_model=text_db_model, chroma_db=chromadb_wrapper, repository_name=DEFECTS_REPO_TEXT)
