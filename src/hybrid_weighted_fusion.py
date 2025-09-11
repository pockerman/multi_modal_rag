from typing import Callable


import numpy as np
from .chromadb_wrapper import ChromaDBHttpWrapper


def normalize_scores(distances):
    """Convert distances to similarity scores in [0,1]."""
    sims = 1 / (1 + np.array(distances))  # simple inverse distance
    return (sims - sims.min()) / (sims.max() - sims.min() + 1e-10)


class HybridWeightedFusion:

    def __init__(self):
        pass


    def retrieve(self, image_embeddings: list[float],
                 text_embeddings: list[float], images_repository: str,
                 text_repository: str, vector_db: ChromaDBHttpWrapper,
                 n_results: int):

        image_results = vector_db.query(repository_name=images_repository,
                                        query_embeddings=image_embeddings, n_results=n_results)

        text_results = vector_db.query(repository_name=text_repository,
                                       query_embeddings=text_embeddings, n_results=n_results)

        return image_results, text_results

    def weighted_fusion(self, img_results: dict, txt_results: dict,
                        w_img: float = 0.6, w_txt: float = 0.4, top_k: int = 5,
                        score_normalizer: Callable = normalize_scores):
        fused = {}

        # Normalize scores
        img_scores = score_normalizer(img_results["distances"][0])
        txt_scores = score_normalizer(txt_results["distances"][0])

        # Merge image results
        for i, doc_id in enumerate(img_results["ids"][0]):
            fused[doc_id] = fused.get(doc_id, 0) + w_img * img_scores[i]

        # Merge text results
        for i, doc_id in enumerate(txt_results["ids"][0]):
            fused[doc_id] = fused.get(doc_id, 0) + w_txt * txt_scores[i]

        # Sort by fused score
        fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Retrieve metadata for top results
        final_results = []
        for doc_id, score in fused_sorted[:top_k]:
            # Look metadata up in one of the collections (same IDs)
            meta = image_collection.get(ids=[doc_id])["metadatas"][0]
            desc = text_collection.get(ids=[doc_id])["documents"][0]
            final_results.append({
                "id": doc_id,
                "score": float(score),
                "metadata": meta,
                "description": desc
            })

        return final_results

    def multistage_retrieval(self, image_embeddings: list[float],
                             text_embeddings: list[float], images_repository: str,
                             text_repository: str, vector_db: ChromaDBHttpWrapper,
                             top_img: int, w_img: float, w_txt: float, top_final: int) -> list[dict]:

        # ---- Stage 1: Image Recall & text retrieval----
        img_results, text_results = self.retrieve(image_embeddings=image_embeddings,
                                                  text_embeddings=text_embeddings, images_repository=images_repository,
                                                  text_repository=text_repository, vector_db=vector_db,
                                                  n_results=top_img)

        candidate_ids = set(img_results["ids"][0]) if img_results else set()

        # ---- Stage 3: Weighted Fusion ----
        fused = {}

        if len(img_results['distances'][0]) != 0:
            img_scores = normalize_scores(img_results["distances"][0])
            for i, doc_id in enumerate(img_results["ids"][0]):
                fused[doc_id] = fused.get(doc_id, 0) + w_img * img_scores[i]

        if len(text_results['distances'][0]) != 0:
            txt_scores = normalize_scores(text_results["distances"][0])
            for i, doc_id in enumerate(text_results["ids"][0]):
                fused[doc_id] = fused.get(doc_id, 0) + w_txt * txt_scores[i]

            # Rerank by fused score
        fused_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        image_collection = vector_db.get_collection(images_repository)
        text_collection = vector_db.get_collection(text_repository)

        # ---- Collect top-K results with metadata ----
        final_results = []
        for doc_id, score in fused_sorted[:top_final]:
            meta = image_collection.get(ids=[doc_id])["metadatas"][0]
            desc = text_collection.get(ids=[doc_id])["documents"][0]
            final_results.append({
                "id": doc_id,
                "score": float(score),
                "metadata": meta,
                "description": desc
            })

        return final_results
