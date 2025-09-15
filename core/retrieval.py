"""
Retrieval component for the self-RAG system.
"""

import logging
from typing import List, Dict, Any, Optional

import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VECTORSTORE_DIR = "vectorstore/chroma"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LangChainEmbeddingModel:
    """Wrapper for LangChain's HuggingFaceEmbeddings with additional functionality."""

    def __init__(self, model_name: str = EMBEDDING_MODEL, **kwargs):
        """
        Initialize the LangChain embedding model.

        Args:
            model_name: Name of the HuggingFace model to use
            **kwargs: Additional arguments for HuggingFaceEmbeddings
        """
        self.model_name = model_name
        try:
            # Initialize LangChain's HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=kwargs.get("model_kwargs", {"device": "cpu"}),
                encode_kwargs=kwargs.get(
                    "encode_kwargs", {"normalize_embeddings": False}
                ),
                multi_process=kwargs.get("multi_process", False),
                show_progress=kwargs.get("show_progress", False),
            )
            logger.info(f"Loaded LangChain embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using LangChain's HuggingFaceEmbeddings.

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])

        try:
            # Use LangChain's embed_documents method
            embeddings_list = self.embeddings.embed_documents(texts)
            return np.array(embeddings_list)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return np.array([])

    def embed_query(self, text: str) -> np.ndarray:
        """
        Embed a single query text.

        Args:
            text: Query text to embed

        Returns:
            Numpy array of the embedding
        """
        if not text.strip():
            return np.array([])

        try:
            # Use LangChain's embed_query method
            embedding = self.embeddings.embed_query(text)
            return np.array(embedding)
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            return np.array([])


class VectorRetriever:
    """LangChain-based retriever using Chroma vector store."""

    def __init__(
            self,
            persist_directory: str = VECTORSTORE_DIR,
            collection_name: str = "creditexplain",
            embedding_model: Optional[HuggingFaceEmbeddings] = None,
    ):
        """
        Initialize the Chroma retriever using LangChain.

        Args:
            persist_directory: Directory for Chroma persistence
            collection_name: Name of the collection to use
            embedding_model: Pre-initialized embedding model
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize embedding model if not provided
        if embedding_model is None:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            self.embedding_model = embedding_model

        # Initialize Chroma vector store
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedding_function=self.embedding_model,
        )

        logger.info(
            f"Initialized LangChain Chroma retriever for collection: {collection_name}"
        )

    def retrieve(
            self,
            query_embedding: List[float],
            k: int = 50,
            filter_dict: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using LangChain's similarity_search_by_vector.

        Args:
            query_embedding: Embedding vector of the query
            k: Number of results to return
            filter_dict: Metadata filters to apply

        Returns:
            List of retrieved documents with metadata
        """
        try:
            # Convert filter_dict to LangChain's expected format
            langchain_filter = None
            if filter_dict:
                langchain_filter = {
                    "$and": [{k: {"$eq": v}} for k, v in filter_dict.items()]
                }

            # Use LangChain's built-in similarity search
            results = self.vector_store.similarity_search_by_vector(
                embedding=query_embedding,
                k=min(k, 100),
                filter=langchain_filter,
            )

            # Convert to expected format
            items = []
            for doc in results:
                items.append(
                    {
                        "id": doc.metadata.get("id", ""),
                        "doc_text": doc.page_content,
                        "metadata": doc.metadata,
                        "distance": 0.0,
                    }
                )

            logger.info(f"Retrieved {len(items)} documents using LangChain")
            return items

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def retrieve_by_text(
            self, query: str, k: int = 50, filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using query text (convenience method).
        """
        try:
            results = self.vector_store.similarity_search(
                query=query, k=min(k, 100), filter=filter_dict
            )

            items = []
            for doc in results:
                items.append(
                    {
                        "id": doc.metadata.get("id", ""),
                        "doc_text": doc.page_content,
                        "metadata": doc.metadata,
                        "distance": 0.0,
                    }
                )

            return items

        except Exception as e:
            logger.error(f"Text retrieval failed: {e}")
            return []
