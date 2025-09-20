import requests
from PIL import Image
import base64
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
from .utils import extract_json, check_json
from .hybrid_weighted_fusion import HybridWeightedFusion
from .chromadb_wrapper import ChromaDBHttpWrapper


def normalize(v):
    v = np.array(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-10)


def retry(data: dict, prompt: str, n_tries, url: str, timeout: float = 60.0) -> dict:
    retry_prompt = f"""
    Your previous response was invalid. It must be valid JSON.
    Do not include explanations or extra text. 

    Here is the original task:

    {prompt}
    """

    count = 0
    make_json = None
    while count < n_tries and make_json is None:
        print(f'Retry attempt {count}')

        data['prompt'] = retry_prompt

        response = requests.post(url, json=data, timeout=timeout)
        response.raise_for_status()

        model_response = response.json()['response']
        make_json = extract_json(model_response)

        if make_json is not None:
            retry_prompt = check_json(make_json, prompt=prompt)

            if isinstance(retry_prompt, str):
                make_json = None
        count += 1
    return make_json


def call_ollama(image_path: str | Path, image_caption: str | None,
                summary_prompt: str,
                defect_prompt: str,
                ollama_path: str, model: str, chromadb_wrapper: ChromaDBHttpWrapper,
                images_repository: str, text_repository: str,
                clip_model,
                retriever: HybridWeightedFusion,
                temperature: float,
                n_tries: int = 3, timeout: float = 60.0,
                w_img: float = 0.7, w_txt: float = 0.3,
                top_img: int = 15, top_final: int = 5) -> dict:
    url = ollama_path + "api/generate"

    # read the image and create the embeddings
    # load the image
    image = Image.open(image_path)

    img_embedding = normalize(clip_model.encode(image))
    text_embedding = normalize(clip_model.encode(image_caption)) if image_caption else None

    # Retrieve similar hull_defects_imgs
    results = retriever.multistage_retrieval(
        query="What's in this image?",
        image_embeddings=img_embedding,
        text_embeddings=text_embedding,
        images_repository=images_repository, text_repository=text_repository,
        vector_db=chromadb_wrapper,
        top_img=top_img,
        top_final=top_final,
        w_img=w_img,
        w_txt=w_txt
    )
    # get the model response and put these in the
    # prompt

    print(f'Retrieved {len(results)} similar hull_defects_imgs')

    retrieved_summaries = "\n".join([
        f"- {doc['description']}" for doc in results
    ])

    summary_prompt = summary_prompt.replace('{retrieved_summaries}', retrieved_summaries)

    print(f'Summary prompt {summary_prompt}')

    with open(image_path, "rb") as image_file:

        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        data = {
            "model": model,
            "prompt": summary_prompt,
            "hull_defects_imgs": [encoded_image],
            "options": {"temperature": temperature},
            "stream": False,
            "raw": False

        }

        try:
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()

            summary_response = response.json()['response']

            print(f'Summary response: {summary_response}')

            defect_prompt_from_summary = defect_prompt.replace('{summary}', summary_response)
            data = {
                "model": model,
                "prompt": defect_prompt_from_summary,
                "options": {"temperature": temperature},
                "stream": False,
                "raw": False
            }
            response = requests.post(url, json=data, timeout=timeout)
            response.raise_for_status()

            model_response = response.json()['response']

            print(f'Model response is: {model_response}')

            make_json = extract_json(model_response)

            if make_json is None:
                make_json = retry(data, prompt=defect_prompt_from_summary, n_tries=n_tries, url=url, timeout=timeout)

            return make_json
        except requests.RequestException as e:
            print(f"Error calling Ollama server: {e}")
            return {}
