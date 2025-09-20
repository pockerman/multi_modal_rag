from pathlib import Path
from langchain_ollama import ChatOllama
from . utils import load_document


def classify_relevance(query: str, prompt_path: Path, *,
                       ollama_base_url: str,
                       model: str = 'mistral', temperature: float = 0.0) -> bool:
    prompt = load_document(prompt_path)

    ollama_model = ChatOllama(model=model,
                              temperature=temperature,
                              base_url=ollama_base_url, )

    messages = [
        ("system", "You are a maritime query relevance classifier."),
        ("user", prompt.format(query=query))
    ]

    model_response = ollama_model.invoke(messages)

    if model_response.content.lower().strip() == 'relevant':
        return True

    return False
