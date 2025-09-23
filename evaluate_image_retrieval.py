from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image

from src.hybrid_weighted_fusion import HybridWeightedFusion
from src.utils import read_json
from src.utils import load_document
from src.query_utils import call_ollama, normalize
from src.chromadb_wrapper import ChromaDBHttpWrapper


def precision_at_k(results: dict[str, dict], ground_truth_label: str, k):
    """
    retrieved: list of defect labels returned by the system
    ground_truth: correct defect label (string) or list of valid labels
    k: cutoff rank

    Returns precision@k
    """

    for img in results:
        retrieve_result = results[img]['retrieve_result']
        relevant = sum(1 for r in retrieve_result if r[0] == ground_truth_label)
        print(f'Precision at k={k} for image={img} is {relevant / k}')


if __name__ == '__main__':
    N_RESULTS = 5

    DATA_PATH = Path('./data')
    PROMPTS_PATH = Path('./prompts')

    TEST_IMGS_PATH = DATA_PATH / "test/hull_defects_imgs"
    TEST_DIRS = ['corrosion', 'crack']
    TEST_IMGS_INFO = DATA_PATH / "test/test_image_retrieval.json"

    SUMMARY_PROMPT_PATH = PROMPTS_PATH / "v6_summary.txt"
    DEFECT_PROMPT_PATH = PROMPTS_PATH / "v6_from_summary.txt"

    # CLIP model for both text & hull_defects_imgs
    clip_model = SentenceTransformer("clip-ViT-L-14")

    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)

    # read the test images
    test_queries = read_json(TEST_IMGS_INFO)

    summary_prompt = load_document(SUMMARY_PROMPT_PATH)
    defect_prompt = load_document(DEFECT_PROMPT_PATH)

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

            #print(image_results)

            # from the results we got from the query which images have
            # have the same label?
            found_correct_label = 0
            metadatas = image_results['metadatas'][0]
            distances = image_results['distances'][0]

            results[img['img']] = {
                'label': label,
                'retrieve_result':  [],
                'embedding_model': 'SentenceTransformer("clip-ViT-L-14")',
                'n_results': 5,
                'normalized': True
            }
            for rslt_label, rslt_dist in zip(metadatas, distances):
                results[img['img']]['retrieve_result'].append((rslt_label['defect_label'], rslt_dist))

        print(results)

        precision_at_k(results=results, ground_truth_label=label, k=N_RESULTS)