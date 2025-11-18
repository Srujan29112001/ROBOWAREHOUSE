"""ChromaDB vector store for semantic search."""

import logging
from typing import Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class ChromaDBVectorStore:
    """
    ChromaDB vector store for robot knowledge retrieval.

    Stores manipulation strategies, best practices, and examples
    as embeddings for semantic search.
    """

    def __init__(
        self,
        persist_directory: str = "./data/cache/chromadb",
        collection_name: str = "robot_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist data
            collection_name: Name of the collection
            embedding_model: Sentence transformer model
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name

        # Load embedding model
        self.embedder = SentenceTransformer(embedding_model)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(f"ChromaDB initialized with collection: {collection_name}")

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to vector store.

        Args:
            documents: List of text documents
            metadatas: Optional metadata for each document
            ids: Optional IDs for documents
        """
        # Generate embeddings
        embeddings = self.embedder.encode(documents).tolist()

        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]

        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

        logger.info(f"Added {len(documents)} documents to ChromaDB")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict[str, List]:
        """
        Query vector store.

        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Query results with documents and distances
        """
        # Generate query embedding
        query_embedding = self.embedder.encode([query_text]).tolist()

        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where,
        )

        return results

    def get_relevant_strategies(
        self,
        task_description: str,
        n_results: int = 3,
    ) -> List[Dict]:
        """
        Get relevant manipulation strategies for a task.

        Args:
            task_description: Description of the task
            n_results: Number of strategies to return

        Returns:
            List of relevant strategies
        """
        results = self.query(
            query_text=task_description,
            n_results=n_results,
            where={"type": "strategy"},
        )

        strategies = []
        for i in range(len(results["documents"][0])):
            strategies.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return strategies

    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs."""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Get number of documents in collection."""
        return self.collection.count()

    def reset(self) -> None:
        """Reset collection (delete all documents)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset")
