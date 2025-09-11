from pathlib import Path
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from src.chromadb_wrapper import ChromaDBHttpWrapper
from src.hybrid_weighted_fusion import HybridWeightedFusion

def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


if __name__ == '__main__':
    DATA_PATH = Path('./data')
    img_path = DATA_PATH / 'test/crack/1.jpg'

    # CLIP model for both text & images
    clip_model = SentenceTransformer("clip-ViT-L-14")

    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)
    retriever = HybridWeightedFusion()

    # read the image and create the embeddings
    # load the image
    image = Image.open(img_path)

    img_embedding = normalize(clip_model.encode(image))
    text_embedding = normalize(clip_model.encode("this image shows a crack on gel-coat"))

    #import pdb; pdb.set_trace()
    results = retriever.multistage_retrieval(
        image_embeddings=img_embedding,
        text_embeddings=text_embedding,
        images_repository="defects_images", text_repository='defects_texts',
        vector_db=chromadb_wrapper,
        top_img=15,
        top_final=5,
        w_img=0.7,
        w_txt=0.3
    )

    for r in results:
        print(f"ID: {r['id']} | Score: {r['score']:.4f} | Defect: {r['metadata']['type']} | Desc: {r['description']}")