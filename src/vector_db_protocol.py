from typing import Protocol, Optional, Any


class VectorDBProtocol(Protocol):

    def add(self, repository_name: str, *, ids: Optional[Any] = None,
            embeddings: Optional[Any] = None,
            metadatas: Optional[Any] = None,
            documents: Optional[Any] = None) -> Any:
        ...
