from typing import Optional, Any
import chromadb


class ChromaDBHttpWrapper:
    def __init__(self, host: str = '0.0.0.0', port: int = 8003):
        self._chroma_client = chromadb.HttpClient(host=host, port=port)

    def delete_collection(self, collection_name: str) -> None:
        self._chroma_client.delete_collection(collection_name)

    def create_collection(self, collection_name: str) -> Any:
        return self._chroma_client.create_collection(collection_name)

    def get_collection(self, repository_name: str) -> Any:
        return self._chroma_client.get_collection(repository_name)

    def add(self, repository_name: str, *, ids: Optional[Any] = None,
            embeddings: Optional[Any] = None,
            metadatas: Optional[Any] = None,
            documents: Optional[Any] = None) -> Any:
        collection = self._chroma_client.get_collection(repository_name)
        collection.add(ids=ids,
                       embeddings=embeddings, metadatas=metadatas, documents=documents)
        return self

    def query(self, repository_name: str, n_results: int, query_embeddings: list[float]) -> Any:
        collection = self._chroma_client.get_collection(repository_name)
        return collection.query(query_embeddings=query_embeddings,
                                n_results=n_results)

