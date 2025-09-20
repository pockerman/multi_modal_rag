from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from src.chromadb_wrapper import ChromaDBHttpWrapper
from src.hybrid_weighted_fusion import HybridWeightedFusion
from src.utils import load_document
from src.query_utils import call_ollama


def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


if __name__ == '__main__':
    OLLAMA_URL = "http://localhost:11434/"
    PROMPTS_PATH = Path('./prompts')
    SUMMARY_PROMPT_PATH = PROMPTS_PATH / "v6_summary.txt"
    DEFECT_PROMPT_PATH = PROMPTS_PATH / "v6_from_summary.txt"

    DATA_PATH = Path('./data')

    img_path = DATA_PATH / 'test/crack/4.jpg'

    # CLIP model for both text & hull_defects_imgs
    clip_model = SentenceTransformer("clip-ViT-L-14")

    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)
    retriever = HybridWeightedFusion(cross_encoder_model='cross-encoder/ms-marco-MiniLM-L6-v2')

    summary_prompt = load_document(SUMMARY_PROMPT_PATH)
    defect_prompt = load_document(DEFECT_PROMPT_PATH)

    # # read the image and create the embeddings
    # # load the image
    # image = Image.open(img_path)
    #
    # img_embedding = normalize(clip_model.encode(image))
    # text_embedding = normalize(clip_model.encode("this image shows a crack on gel-coat"))
    #
    # results = retriever.multistage_retrieval(
    #     query="What's in this image?",
    #     image_embeddings=img_embedding,
    #     text_embeddings=text_embedding,
    #     images_repository="defects_images", text_repository='defects_texts',
    #     vector_db=chromadb_wrapper,
    #     top_img=15,
    #     top_final=5,
    #     w_img=0.7,
    #     w_txt=0.3
    # )
    #
    # for r in results:
    #     print(f"ID: {r['id']} | Score: {r['score']:.4f} | Defect: {r['metadata']['label']} | Desc: {r['description']}")

    ollama_result = call_ollama(image_path=img_path,
                                summary_prompt=summary_prompt,
                                defect_prompt=defect_prompt,
                                chromadb_wrapper=chromadb_wrapper, retriever=retriever,
                                text_repository='defects_texts',
                                images_repository="defects_images",
                                clip_model=clip_model, ollama_path=OLLAMA_URL,
                                top_img=15, top_final=5, w_img=0.7, w_txt=0.3,
                                n_tries=3, timeout=60.0, image_caption=None,
                                model="llava", temperature=0.0)

    print(ollama_result)
