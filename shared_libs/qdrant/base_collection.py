from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams


class QdrantCollection(ABC):
    """Abstract base class for Qdrant collections.

    Provides common methods for creating a collection, adding records, and
    searching by embedding. Subclasses must define collection-specific attributes.
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Name of the Qdrant collection."""

    @property
    @abstractmethod
    def vector_size(self) -> int:
        """Dimensionality of vectors stored in the collection."""

    @property
    def distance(self) -> Distance:
        """Distance metric for vector search. Defaults to COSINE."""
        return Distance.COSINE

    def __init__(self, host: str = "localhost", port: int = 6333) -> None:
        """Initialize Qdrant client connection.

        Args:
            host (str): Host address of Qdrant server.
            port (int): Port of Qdrant server.
        """
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self) -> None:
        """Create the Qdrant collection with the specified schema if it does not already exist.

        Raises:
            RuntimeError: If collection creation fails.
        """
        try:
            collections_names = [collection.name for collection in self.client.get_collections().collections]

            if self.collection_name in collections_names:
                return
            else:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_size, distance=self.distance),
                )
        except Exception as e:
            msg = f"Failed to create collection '{self.collection_name}': {e}"
            raise RuntimeError(msg)

    def add_records(self, records: list[dict[str, Any]]) -> None:
        """Add multiple records (vectors with payload) to the Qdrant collection.

        Each record dict should contain:
            - "id" (int or str): Unique identifier for the point.
            - "vector" (List[float]): The embedding vector.
            - "payload" (Dict[str, Any]): Metadata associated with the point.

        Args:
            records (List[Dict[str, Any]]): List of records to upsert.

        Raises:
            ValueError: If records list is empty.
            RuntimeError: If upsert operation fails.
        """
        if not records:
            raise ValueError("No records provided for upsert.")

        points = []
        for rec in records:
            point = PointStruct(
                id=int(rec["id"]),
                vector=rec["vector"],
                payload=rec.get("payload", {}),
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to upsert records into '{self.collection_name}': {e}")
        
    def search(
        self,
        query_vector: list[float],
        limit: int = 10,
        filter: Optional[Filter] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for the nearest vectors in the collection, applying an optional score threshold.

        Args:
            query_vector (List[float]): The query embedding vector.
            limit (int, optional): Number of nearest neighbors to retrieve. Defaults to 10.
            filter (Optional[Filter], optional): Payload filter to apply. Defaults to None.
            score_threshold (Optional[float], optional): Minimum score for results; only points with score >= threshold are returned. Defaults to None.
        Returns:
            List[Dict[str, Any]]: List of search results, each as a dict with keys "id", "payload", and "score".

        Raises:
            RuntimeError: If the search operation fails.
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                # filter=filter,
                score_threshold=score_threshold,
            )
        except Exception as e:
            msg = f"Search in '{self.collection_name}' failed: {e}"
            raise RuntimeError(msg)

        return [
            {"id": point.id, "payload": point.payload, "score": point.score}
            for point in results
        ]