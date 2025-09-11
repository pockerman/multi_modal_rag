from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from .chromadb_wrapper import ChromaDBHttpWrapper


def normalize_scores(distances):
    """Convert distances to similarity scores in [0,1]."""
    sims = 1 / (1 + np.array(distances))  # simple inverse distance
    return (sims - sims.min()) / (sims.max() - sims.min() + 1e-10)


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """RRF fusion as a fallback for heterogeneous retrievers."""
    scores = {}
    for r in rankings:
        for rank, doc_id in enumerate(r, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
    return scores


class HybridWeightedFusion:

    def __init__(self, device: str = "cpu", cross_encoder_model: Optional[str] = None):
        self.device = device
        self.cross_encoder_model = cross_encoder_model
        self.tokenizer = None

        if cross_encoder_model:
            self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
            self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(
                cross_encoder_model
            ).to(device)

    def retrieve(self, image_embeddings: list[float],
                 text_embeddings: list[float], images_repository: str,
                 text_repository: str, vector_db: ChromaDBHttpWrapper,
                 n_results: int):

        image_results = vector_db.query(repository_name=images_repository,
                                        query_embeddings=image_embeddings, n_results=n_results)

        text_results = vector_db.query(repository_name=text_repository,
                                       query_embeddings=text_embeddings, n_results=n_results)

        return image_results, text_results

    def fuse(self, img_results: dict, text_results: dict,
             w_img: float = 0.6, w_txt: float = 0.4,
             fusion_method: str = 'RRF') -> list:

        # ---- Stage 3: Weighted Fusion ----
        fused = {}

        if fusion_method == 'RRF':
            # ---- Stage 2 (alternative): Reciprocal Rank Fusion ----
            rankings = []
            if img_results and "ids" in img_results:
                rankings.append(img_results["ids"][0])
            if text_results and "ids" in text_results:
                rankings.append(text_results["ids"][0])
            fused = reciprocal_rank_fusion(rankings)
        else:
            # weighted fusion

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
        return fused_sorted

    def multistage_retrieval(self, query: str,
                             image_embeddings: list[float],
                             text_embeddings: list[float], images_repository: str,
                             text_repository: str, vector_db: ChromaDBHttpWrapper,
                             top_img: int, w_img: float, w_txt: float, top_final: int,
                             fusion_method: str = 'RRF', use_rerank: bool = True,
                             rerank_score_weight: float = 0.5) -> list[dict]:

        # ---- Stage 1: Image Recall & text retrieval----
        img_results, text_results = self.retrieve(image_embeddings=image_embeddings,
                                                  text_embeddings=text_embeddings, images_repository=images_repository,
                                                  text_repository=text_repository, vector_db=vector_db,
                                                  n_results=top_img)

        # --- stage 2: fuse the results from the retrieval
        fused_sorted = self.fuse(img_results=img_results, text_results=text_results,
                                 w_img=w_img, w_txt=w_txt, fusion_method=fusion_method)

        image_collection = vector_db.get_collection(images_repository)
        text_collection = vector_db.get_collection(text_repository)

        candidates = []
        for doc_id, score in fused_sorted[: top_final * 2]:  # keep buffer for reranking
            meta = image_collection.get(ids=[doc_id])["metadatas"][0]
            desc = text_collection.get(ids=[doc_id])["documents"][0]
            candidates.append({
                "id": doc_id,
                "score": float(score),
                "metadata": meta,
                "description": desc
            })

        # ---- Stage 4: Re-ranking with cross-encoder ----
        score_weight = 1.0 - rerank_score_weight
        if use_rerank and self.cross_encoder_model:
                for c in candidates:
                    rerank_score = self._cross_encoder_score(query, c["description"])
                    # combine fusion and rerank scores (can tune weights)
                    c["score"] = score_weight * c["score"] + rerank_score_weight * rerank_score
                candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        return candidates[:top_final]

    def _cross_encoder_score(self, query: str, doc: str) -> float:
        """Re-rank candidate pairs using a cross-encoder."""
        if not self.cross_encoder_model:
            return 0.0

        inputs = self.tokenizer(
            query, doc, return_tensors="pt", truncation=True, padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.cross_encoder_model(**inputs)
            score = torch.sigmoid(outputs.logits).squeeze().item()

        return score
