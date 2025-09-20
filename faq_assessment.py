from pathlib import Path
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from src.chromadb_wrapper import ChromaDBHttpWrapper
from src.utils import load_document
from src.relevance_classifier import classify_relevance


def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


if __name__ == '__main__':
    FAQS_REPO: str = 'faqs_repo'
    TOP_K_DOCUMENTS = 2
    MODEL = 'mistral'
    TEMPERATURE = 0.0
    OLLAMA_URL = "http://localhost:11434/"

    GENERATE_URL = OLLAMA_URL + "api/generate"
    TIMEOUT = 120.0
    PROMPTS_PATH = Path('./prompts')
    FAQ_PATH = PROMPTS_PATH / "faqs/v1.txt"
    CLASSIFIER_PATH = PROMPTS_PATH / "faqs/query_classifier.txt"
    clip_model = SentenceTransformer("clip-ViT-L-14")
    chromadb_wrapper = ChromaDBHttpWrapper(host='0.0.0.0', port=8003)

    faq_prompt = load_document(FAQ_PATH)

    ask = True
    while ask:

        user_query = input("Ask a question (use e! to exit): ")

        if user_query.lower() == 'e!':
            print("Exiting chat...")
            break

        if not classify_relevance(query=user_query, ollama_base_url=OLLAMA_URL,
                                  prompt_path=CLASSIFIER_PATH,
                                  model='mistral', temperature=0.0):
            print("The query is relevant to maritime vessels. So I cannot answer it....")
            continue

        # query the database
        text_embeddings = normalize(clip_model.encode(user_query))
        image_results = chromadb_wrapper.query(repository_name=FAQS_REPO,
                                               query_embeddings=text_embeddings,
                                               n_results=TOP_K_DOCUMENTS)

        print(f'Retrieved {len(image_results['ids'][0])} relevant documents')
        print(image_results)

        possible_answers = ""
        for doc in image_results['metadatas'][0]:
            possible_answers += doc['answer']
        query_prompt = faq_prompt.format(user_query=user_query, possible_answers=possible_answers)

        data = {
            "model": MODEL,
            "prompt": query_prompt,
            "options": {"temperature": TEMPERATURE},
            "stream": False,
            "raw": False
        }

        response = requests.post(GENERATE_URL, json=data, timeout=TIMEOUT)
        response.raise_for_status()
        summary_response = response.json()['response']
        print(f'Summary response: {summary_response}')
