from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image
import math

from src.utils import read_json
from src.query_utils import normalize
from src.chromadb_wrapper import ChromaDBHttpWrapper


def precision_at_k(results: dict[str, dict], ground_truth_label: str, k: int):
    """
    retrieved: list of defect labels returned by the system
    ground_truth: correct defect label (string) or list of valid labels
    k: cutoff rank

    Returns precision@k
    """

    for img in results:
        retrieve_result = results[img]['retrieve_result']
        relevant = sum(1 for r in retrieve_result if r[0] == ground_truth_label)
        print(f'Precision@{k} for image={img} is {relevant / k:0.2f}')


def recall_at_k(results: dict[str, dict], ground_truth_label: str,
                ground_truth_set_size: int, k: int) -> dict[str,float]:
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
        recall_dict[img] = relevant
        print(f'Recall@{k} for image={img} is {relevant / float(ground_truth_set_size):0.2f}')
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



def dcg_at_k(relevances, k: int ):
    """
    relevances: list of relevance scores in ranked order (highest rank first)
    k: cutoff
    """
    relevances = relevances[:k]
    return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(relevances))


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
        print(f'nDCG@{k} for image={img} is {img_ndcg} ')
        ndcg_dict[img] = img_ndcg
    return ndcg_dict


if __name__ == '__main__':
    N_RESULTS = 5

    DATA_PATH = Path('./data')
    PROMPTS_PATH = Path('./prompts')

    TEST_IMGS_PATH = DATA_PATH / "test/hull_defects_imgs"
    TEST_DIRS = ['corrosion', 'crack']
    TEST_IMGS_INFO = DATA_PATH / "test/test_image_retrieval.json"

    # CLIP model for both text & hull_defects_imgs
    clip_model = SentenceTransformer("clip-ViT-L-14")

    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)

    # read the test images
    test_queries = read_json(TEST_IMGS_INFO)
    print("Loaded test images...")

    for label in test_queries:
        print(f"For label {label} found {len(test_queries[label])} images")
        label_images = test_queries[label]
        results = {}
        for img in label_images:
            img_path = TEST_IMGS_PATH / f'{label}/{img['img']}'

            # read the image and create the embeddings
            # load the image
            image = Image.open(img_path)
            img_embedding = normalize(clip_model.encode(image))

            image_results = chromadb_wrapper.query(repository_name="defects_images",
                                                   query_embeddings=img_embedding,
                                                   n_results=5)

            # from the results we got from the query which images have
            # have the same label?
            found_correct_label = 0
            metadatas = image_results['metadatas'][0]
            distances = image_results['distances'][0]

            results[img['img']] = {
                'label': label,
                'retrieve_result': [],
                'embedding_model': 'SentenceTransformer("clip-ViT-L-14")',
                'n_results': 5,
                'normalized': True
            }
            for rslt_label, rslt_dist in zip(metadatas, distances):
                results[img['img']]['retrieve_result'].append((rslt_label['defect_label'], rslt_dist))

        precision_at_k(results=results, ground_truth_label=label, k=N_RESULTS)
        recall_at_k(results=results, ground_truth_label=label,
                    k=N_RESULTS, ground_truth_set_size=len(label_images))

        mean_reciprocal_rank(results=results, ground_truth_label=label)
        ndcg_at_k(results=results, ground_truth_label=label, k=N_RESULTS)
