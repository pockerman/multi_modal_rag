from pathlib import Path
from statistics import mean
import math
from collections import defaultdict
from src.utils import read_json, write_json

from src.chromadb_wrapper import ChromaDBHttpWrapper
from src.openclip_embedding import OpenCipEmbeddings
from src.sentence_transformer_embedding import SentenceTransformerEmbeddings
from src.siglip_embedding import SigLipEmbeddings


def precision_at_k(results: dict[str, dict],
                   ground_truth_label: str, k: int) -> dict[str, float]:
    """
    retrieved: list of defect labels returned by the system
    ground_truth: correct defect label (string) or list of valid labels
    k: cutoff rank

    Returns precision@k
    """

    precision_dict = {}
    for img in results:
        retrieve_result = results[img]['retrieve_result']
        relevant = sum(1 for r in retrieve_result if r[0] == ground_truth_label)
        precision_dict[img] = relevant / k
        #print(f'Precision@{k} for image={img} is {relevant / k:0.2f}')

    print(f'Average Precision@{k} {sum(list(precision_dict.values()))/len(list(precision_dict.values()))}')
    return precision_dict


def recall_at_k(results: dict[str, dict], ground_truth_label: str,
                ground_truth_set_size: int, k: int) -> dict[str, float]:
    """
    retrieved: list of defect labels returned by the system
    ground_truth_set: set of all relevant labels for the query
                      (e.g., all corrosion images for corrosion query)
    k: cutoff rank

    Returns recall@k
    """

    recall_dict = {}
    for img in results:
        retrieve_result = results[img]['retrieve_result']
        relevant = sum(1 for r in retrieve_result if r[0] == ground_truth_label)
        recall_dict[img] = relevant / float(ground_truth_set_size)
        #print(f'Recall@{k} for image={img} is {relevant / float(ground_truth_set_size):0.2f}')

    print(f'Average Recall@{k} {sum(list(recall_dict.values())) / len(list(recall_dict.values()))}')
    return recall_dict


def reciprocal_rank(img_results: list[tuple[str, float]], ground_truth_label: str):
    """
    retrieved: ranked list of items (e.g. image IDs or labels)
    ground_truth_set: set of all relevant items for this query
    Returns Reciprocal Rank (RR) for one query
    """

    for i, item in enumerate(img_results, start=1):
        if item[0] == ground_truth_label:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(results: dict[str, dict], ground_truth_label: str) -> float:
    """
    all_retrieved: list of retrieval results per query (list of lists)
    all_ground_truth: list of ground truth sets per query (list of sets)
    Returns MRR across all queries
    """
    rr_scores = []

    for img in results:
        retrieve_result = results[img]['retrieve_result']
        rr_scores.append(reciprocal_rank(retrieve_result, ground_truth_label=ground_truth_label))
    mrr = sum(rr_scores) / len(rr_scores)
    print(f'MRR for {len(results)} image queries is {mrr}')
    return mrr


def dcg_at_k(relevances, k: int):
    """
    relevances: list of relevance scores in ranked order (highest rank first)
    k: cutoff
    """
    relevances = relevances[:k]
    return sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(results: dict[str, dict], ground_truth_label: str, k: int):
    """
    relevances: list of relevance scores in ranked order (highest rank first)
    k: cutoff
    """
    ndcg_dict = {}
    for img in results:
        retrieve_result = results[img]['retrieve_result']

        relevances = []

        for item in retrieve_result:
            if item[0] == ground_truth_label:
                relevances.append(1)
            else:
                relevances.append(0)

        dcg = dcg_at_k(relevances, k)
        # Ideal DCG: sort relevances in descending order
        idcg = dcg_at_k(sorted(relevances, reverse=True), k)
        img_ndcg = dcg / idcg if idcg > 0 else 0.0
        #print(f'nDCG@{k} for image={img} is {img_ndcg} ')
        ndcg_dict[img] = img_ndcg

    print(f'Average nDCG@{k} {sum(list(ndcg_dict.values())) / len(list(ndcg_dict.values()))}')
    return ndcg_dict


def reciprocal_rank_fusion(results_list, k=60, top_n=5):
    """
    Perform Reciprocal Rank Fusion (RRF) on ChromaDB query results.

    Args:
        results_dicts: list of ChromaDB query outputs (dicts).
        k: smoothing constant
        top_n: number of final results to return

    Returns:
        dict in ChromaDB-like format with fused ranking.
    """
    scores = defaultdict(float)
    metadata_map = {}
    doc_map = {}
    distance_map = defaultdict(list)
    for results in results_list:
        ids = results["ids"][0]  # list of IDs
        metas = results.get("metadatas", [[]])[0]
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0] if "distances" in results else []

        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] += 1.0 / (k + rank)
            idx = rank - 1
            # Store metadata/docs if not already
            if doc_id not in metadata_map and metas:
                metadata_map[doc_id] = metas[rank - 1]
            if doc_id not in doc_map and docs:
                doc_map[doc_id] = docs[rank - 1]

            if distances:
                try:
                    distance_map[doc_id].append(distances[idx])
                except IndexError:
                    # distance list might be shorter; just skip
                    pass

    # Sort by score
    fused_ids = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    fused_scores = [scores[doc_id] for doc_id in fused_ids]
    fused_metas = [metadata_map.get(doc_id) for doc_id in fused_ids]
    fused_docs = [doc_map.get(doc_id) for doc_id in fused_ids]
    # aggregated distances: mean of observed distances (or None if none available)
    fused_distances = []
    for doc_id in fused_ids:
        vals = distance_map.get(doc_id)
        if vals:
            fused_distances.append(mean(vals))
        else:
            fused_distances.append(None)  # or a sentinel like float('inf')

    return {
        "ids": [fused_ids],
        "rrf_scores": [fused_scores],  # note: RRF scores, not Chroma distances
        "metadatas": [fused_metas],
        "documents": [fused_docs],
        'distances': [fused_distances]
    }


if __name__ == '__main__':
    N_TOP_RESULTS = 5
    N_DOCS_FOR_RETRIEVE = 5
    SMOOTH_CONSTANT = 15
    RESULTS_FILE_INDEX = 11

    DATA_PATH = Path('./data')
    TEST_IMGS_PATH = DATA_PATH / "test/hull_defects_imgs"
    TEST_DIRS = ['corrosion', 'crack', 'dent', 'blister', 'delamination', 'buckling']
    TEST_IMGS_INFO = DATA_PATH / "test/test_image_retrieval.json"
    RESULTS_FILE = DATA_PATH / f"test/test_results/{RESULTS_FILE_INDEX}.json"

    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)

    embeders = [OpenCipEmbeddings(device='cpu'),
                SentenceTransformerEmbeddings(model_name="clip-ViT-L-14"),
                SigLipEmbeddings(model_name="google/siglip-base-patch16-224")]

    chroma_db_repos = ['defects_images_' + embeder.embedder_id
                       for embeder in embeders]

    # read the test images
    test_queries = read_json(TEST_IMGS_INFO)
    print("Loaded test images...")

    total_results_label = {
        'embedding_models': [embeder.embedder_id for embeder in embeders],
        'n_results': N_TOP_RESULTS,
        'normalized': True,
        'retrieval_strategy': {
            'name': 'RRF',
            'top_n': N_TOP_RESULTS,
            'n_fetched_docs': N_DOCS_FOR_RETRIEVE,
            'smooth_constant': SMOOTH_CONSTANT
        },
        'label_result': {}
    }

    for label in test_queries:
        print(f"For label {label} found {len(test_queries[label])} images")
        label_images = test_queries[label]
        results = {}
        for img in label_images:
            img_path = TEST_IMGS_PATH / f'{label}/{img['img']}'

            image_results = []
            for chroma_db_repo, embedder in zip(chroma_db_repos, embeders):
                # read the image and create the embeddings
                img_embedding = embedder.embed_image(img_path)
                image_search_results = chromadb_wrapper.query(repository_name=chroma_db_repo,
                                                              query_embeddings=img_embedding,
                                                              n_results=N_DOCS_FOR_RETRIEVE)

                image_results.append(image_search_results)

            image_results = reciprocal_rank_fusion(results_list=image_results,
                                                   k=SMOOTH_CONSTANT, top_n=N_TOP_RESULTS)

            # print(image_results)

            # from the results we got from the query which images have
            # have the same label?
            found_correct_label = 0
            metadatas = image_results['metadatas'][0]
            distances = image_results['distances'][0]

            results[img['img']] = {
                'label': label,
                'retrieve_result': [],
            }
            for rslt_label, rslt_dist in zip(metadatas, distances):
                results[img['img']]['retrieve_result'].append((rslt_label['defect_label'], rslt_dist))

        precision_dict = precision_at_k(results=results, ground_truth_label=label, k=N_TOP_RESULTS)
        recall_dict = recall_at_k(results=results, ground_truth_label=label,
                                  k=N_TOP_RESULTS, ground_truth_set_size=len(label_images))

        mrr = mean_reciprocal_rank(results=results, ground_truth_label=label)
        ndcg_dict = ndcg_at_k(results=results, ground_truth_label=label, k=N_TOP_RESULTS)

        total_results_label['label_result'][label] = {
            'precision': precision_dict,
            'recall': recall_dict,
            'mrr': mrr,
            'ndcg': ndcg_dict,
            'results': results
        }

    write_json(total_results_label, RESULTS_FILE)
