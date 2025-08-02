from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict

from arango import ArangoClient
from arango.exceptions import CollectionCreateError

if TYPE_CHECKING:
    from arango.collection import EdgeCollection, StandardCollection
    from arango.database import Database


class ArangoCollectionBase(ABC):
    """Abstract base class for interacting with an ArangoDB collection.

    Provides common methods for creating the collection (if not exists), inserting,
    retrieving, updating, and deleting documents. Subclasses must specify the
    collection name and whether it is an edge collection.
    """

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """Name of the ArangoDB collection."""

    @property
    def is_edge_collection(self) -> bool:
        """Whether this collection should be created as an edge collection.

        Defaults to False (standard document collection). Override in subclasses
        if an edge collection is required.
        """
        return False

    def __init__(
        self,
        host: str,
        port: int = 8529,
        username: str = "root",
        password: str = "",
        db_name: str = "_system",
    ) -> None:
        """Initialize the ArangoDB client and ensure the collection exists.

        Args:
            host (str): Hostname or IP address of the ArangoDB server (including protocol).
            port (int): Port on which ArangoDB is listening. Defaults to 8529.
            username (str): Username for ArangoDB authentication. Defaults to "root".
            password (str): Password for ArangoDB authentication. Defaults to empty string.
            db_name (str): Name of the database to connect to. Defaults to "_system".

        Raises:
            RuntimeError: If unable to connect to the database or create/access the collection.
        """
        # Build the full URL for the ArangoDB server
        url = f"{host}:{port}"
        try:
            client = ArangoClient(hosts=url)
            self.db: Database = client.db(db_name, username=username, password=password)
        except Exception as e:
            msg = f"Failed to connect to ArangoDB at {url}/{db_name}: {e}"
            raise RuntimeError(msg)

        # Ensure the collection exists (create if needed)
        try:
            if self.is_edge_collection:
                if not self.db.has_collection(self.collection_name):
                    self.db.create_collection(self.collection_name, edge=True)
                self.collection: EdgeCollection = self.db.collection(self.collection_name)
            else:
                if not self.db.has_collection(self.collection_name):
                    self.db.create_collection(self.collection_name)
                self.collection: StandardCollection = self.db.collection(self.collection_name)
        except CollectionCreateError as e:
            msg = f"Failed to create or access collection '{self.collection_name}': {e}"
            raise RuntimeError(msg)
        except Exception as e:
            msg = f"Unexpected error accessing collection '{self.collection_name}': {e}"
            raise RuntimeError(msg)

    def insert_item(self, item: dict[str, Any], return_new: bool = False) -> dict[str, Any]:
        """Insert a document into the collection.

        Args:
            item (Dict[str, Any]): The document body to insert. If a key (_key) field
                is included, it will be used; otherwise ArangoDB will generate one.
            return_new (bool): If True, return the newly created document object
                (including generated fields). Defaults to False.

        Returns:
            Dict[str, Any]: The insert result. If return_new is True, includes 'new'.
                            Otherwise returns metadata about the operation.

        Raises:
            ValueError: If the insert operation fails.
        """
        try:
            result = self.collection.insert(item, return_new=return_new)
            return result
        except Exception as e:
            msg = f"Failed to insert item into '{self.collection_name}': {e}"
            raise ValueError(msg)

    def get_item(self, key: str) -> dict[str, Any] | None:
        """Retrieve a document by its key.

        Args:
            key (str): The _key of the document to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The document if found, otherwise None.

        Raises:
            RuntimeError: If retrieval fails due to server or permission error.
        """
        try:
            return self.collection.get(key)
        except Exception as e:
            msg = f"Failed to get item '{key}' from '{self.collection_name}': {e}"
            raise RuntimeError(msg)

    def update_item(self, key: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing document by its key.

        Args:
            key (str): The _key of the document to update.
            updates (Dict[str, Any]): A dict of attributes to update. Only provided fields
                will be changed; other fields remain intact.
            return_new (bool): If True, return the updated document. Defaults to False.

        Returns:
            Dict[str, Any]: The update result. If return_new is True, includes 'new'.
                            Otherwise returns metadata about the operation.

        Raises:
            ValueError: If the update operation fails.
        """
        try:
            result = self.collection.update_match({"_key": key}, updates)
            return result
        except Exception as e:
            msg = f"Failed to update item '{key}' in '{self.collection_name}': {e}"
            raise ValueError(msg)

    def delete_item(self, key: str) -> bool:
        """Delete a document by its key.

        Args:
            key (str): The _key of the document to delete.

        Returns:
            bool: True if deletion succeeded, False otherwise.

        Raises:
            RuntimeError: If deletion fails due to server or permission error.
        """
        try:
            return self.collection.delete(key)
        except Exception as e:
            msg = f"Failed to delete item '{key}' from '{self.collection_name}': {e}"
            raise RuntimeError(msg)

    def find_all(self, limit: int = 100, skip: int = 0) -> list[dict[str, Any]]:
        """Retrieve all documents from the collection (with pagination).

        Args:
            limit (int): Maximum number of documents to return. Defaults to 100.
            skip (int): Number of documents to skip (for paging). Defaults to 0.

        Returns:
            List[Dict[str, Any]]: A list of documents (each as a dict).

        Raises:
            RuntimeError: If query fails.
        """
        try:
            cursor = self.collection.all(limit=limit, skip=skip)
            return list(cursor)
        except Exception as e:
            msg = f"Failed to retrieve documents from '{self.collection_name}': {e}"
            raise RuntimeError(msg)

    def query(
        self,
        query_str: str,
        bind_vars: dict[str, Any] | None = None,
        count: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute an AQL query against the database.

        Args:
            query_str (str): The AQL query string.
            bind_vars (Optional[Dict[str, Any]]): Dictionary of bind parameters. Defaults to None.
            count (bool): If True, return the count of results as well. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of result documents or records.

        Raises:
            RuntimeError: If query execution fails.
        """
        try:
            cursor = self.db.aql.execute(query_str, bind_vars=bind_vars or {}, count=count)
            return list(cursor)
        except Exception as e:
            msg = f"Failed to execute query on '{self.collection_name}': {e}"
            raise RuntimeError(msg)
