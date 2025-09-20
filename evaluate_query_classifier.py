from pathlib import Path
from src.relevance_classifier import classify_relevance
from src.utils import read_json

if __name__ == '__main__':
    MODEL = 'mistral'
    TEMPERATURE = 0.0
    OLLAMA_URL = "http://localhost:11434/"

    PROMPTS_PATH = Path('./prompts')
    CLASSIFIER_PATH = PROMPTS_PATH / "query_classifier/query_classifier.txt"
    DATA_PATH = Path('./data')
    FAQ_PATH = DATA_PATH / "test/test_query_classifier.json"
    # read the test queries
    test_queries = read_json(FAQ_PATH)

    test_queries = test_queries['queries']

    total = len(test_queries)
    correct = 0
    incorrect = 0

    for query in test_queries:
        question = query['query']
        expected = query['expected']
        result = classify_relevance(query=question,
                                    ollama_base_url=OLLAMA_URL,
                                    prompt_path=CLASSIFIER_PATH,
                                    model='mistral', temperature=0.0)

        if result == expected:
            correct += 1
        else:
            incorrect += 1

    print(f'Classifier accuracy {float(correct) / float(total):0.2f}')
