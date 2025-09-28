import uuid
from typing import Callable
from pydantic import BaseModel
from PIL import Image


class ImageModel(BaseModel):

    uid: str
    img_embedding: list[float]
    description: str
    hull_material: str
    defect_label: str
    severity: str = "INVALID"

    @staticmethod
    def create_from(data: dict, image_encoder) -> "ImageModel":
        img_id = uuid.uuid4().hex
        defect_label = data["label"]
        hull_material = data['variant']
        description = data['description']
        source = data['source']

        img_path = data['img_path']

        img_embedding = image_encoder.embed_image(img_path)

        return ImageModel(uid=img_id, description=description, source=source,
                          hull_material=hull_material, defect_label=defect_label,
                          img_embedding=img_embedding)


class DefectTextModel(BaseModel):

    uid: str
    text_embedding: list[float]
    text_description: str
    defect_label: str
    recommendations: list[dict]

    @staticmethod
    def create_from(data: dict, text_encoder, *, normalize: Callable = None) -> "DefectTextModel":

        if 'id' not in data:
            text_id = uuid.uuid4().hex
        else:
            text_id = data['id']

        text  = data['description']
        if normalize:
            txt_emb = normalize(text_encoder.encode(text))
        else:
            txt_emb = text_encoder.encode(text)

        return DefectTextModel(uid=text_id,
                               text_embedding=txt_emb,
                               text_description=text,
                               defect_label=data['label'],
                               recommendations=data['recommendations'])
