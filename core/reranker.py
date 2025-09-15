"""
LangChain-compatible reranker component for Self-RAG system.
Uses cross-encoder model to re-rank retrieved passages by query relevance.
"""

import logging
from typing import List, Dict, Tuple, Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.document_compressors.base import DocumentCompressorPipeline
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LangChainReranker:
    """LangChain-compatible cross-encoder reranker."""

    def __init__(
            self, model_name: str = RERANKER_MODEL, top_n: int = 6, device: str = "cpu"
    ):
        """
        Initialize the LangChain cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            top_n: Number of top results to return
        """
        super().__init__()
        self.model_name = model_name
        self.top_n = top_n

        try:
            # Create HuggingFaceCrossEncoder instance first
            cross_encoder_model = HuggingFaceCrossEncoder(
                model_name=model_name, model_kwargs={"device": device}
            )

            # Create LangChain's cross-encoder reranker
            self.compressor = CrossEncoderReranker(
                model=cross_encoder_model, top_n=top_n
            )

            logger.info(f"Loaded LangChain reranker model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load LangChain reranker {model_name}: {e}")
            self.compressor = None

    def invoke(self, input_dict: Dict) -> Dict:
        """
        LangChain compatible invoke method for reranking.

        Args:
            input_dict: Dictionary with 'query' and 'documents' keys

        Returns:
            Dictionary with reranked documents and scores
        """
        query = input_dict.get("query", "")
        documents = input_dict.get("documents", [])

        if self.compressor is None:
            logger.warning("Reranker not initialized, returning original documents")
            return {
                "documents": documents[: self.top_n],
                "scores": [1.0] * min(len(documents), self.top_n),
            }

        try:
            # Compress (rerank) the documents
            compressed_docs = self.compressor.compress_documents(
                documents=documents, query=query
            )

            # Extract scores from metadata if available
            scores = []
            for doc in compressed_docs:
                score = doc.metadata.get("relevance_score", 1.0)
                scores.append(score)

            return {"documents": compressed_docs, "scores": scores}

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return {
                "documents": documents[: self.top_n],
                "scores": [1.0] * min(len(documents), self.top_n),
            }

    def rerank(
            self, query: str, candidates: List[Dict], top_n: Optional[int] = None
    ) -> Tuple[List[Dict], List[float]]:
        """
        Original rerank method maintained for backward compatibility.

        Args:
            query: User query text
            candidates: List of candidate passages with 'doc_text'
            top_n: Number of top results to return (overrides default if provided)

        Returns:
            Tuple of (top candidates, their scores)
        """
        if top_n is None:
            top_n = self.top_n

        # Convert to LangChain Documents
        documents = []
        for candidate in candidates:
            doc = Document(
                page_content=candidate.get("doc_text", ""),
                metadata={
                    "id": candidate.get("id", ""),
                    "original_metadata": candidate.get("metadata", {}),
                    "original_distance": candidate.get("distance", 0.0),
                },
            )
            documents.append(doc)

        # Use LangChain invoke
        result = self.invoke({"query": query, "documents": documents})

        # Convert back to original format
        reranked_candidates = []
        scores = []

        for doc, score in zip(result["documents"], result["scores"]):
            reranked_candidates.append(
                {
                    "id": doc.metadata.get("id", ""),
                    "doc_text": doc.page_content,
                    "metadata": doc.metadata.get("original_metadata", {}),
                    "distance": doc.metadata.get("original_distance", 0.0),
                    "rerank_score": score,
                }
            )
            scores.append(score)

        return reranked_candidates, scores


class ContextualCompressionRetrieverWrapper:
    """Wrapper for LangChain's ContextualCompressionRetriever for easy integration."""

    def __init__(
            self,
            base_retriever,
            reranker_model: str = RERANKER_MODEL,
            device: str = "cpu",
    ):
        """
        Initialize contextual compression retriever with reranking.

        Args:
            base_retriever: Base retriever (vector, BM25, or ensemble)
            reranker_model: Cross-encoder model name for reranking
        """
        # Create the cross-encoder model instance first
        cross_encoder = HuggingFaceCrossEncoder(
            model_name=reranker_model,
            model_kwargs={"device": device}
        )
        # Pass the instance to CrossEncoderReranker
        self.reranker = CrossEncoderReranker(model=cross_encoder)
        self.compression_pipeline = DocumentCompressorPipeline(
            transformers=[self.reranker]
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compression_pipeline, base_retriever=base_retriever
        )

    def get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        """Get relevant documents with reranking."""
        return self.compression_retriever.get_relevant_documents(query, **kwargs)

    def invoke(self, query: str, **kwargs) -> List[Document]:
        """LangChain invoke method."""
        return self.get_relevant_documents(query, **kwargs)


# Factory function for creating different types of retrievers
def create_advanced_retriever(retriever_type: str = "vector", **kwargs):
    """
    Factory function to create advanced retrievers with reranking.

    Args:
        retriever_type: Type of retriever ('vector', 'hybrid', 'compression')
        **kwargs: Additional arguments for specific retriever types

    Returns:
        Configured retriever instance
    """
    if retriever_type == "compression":
        # Create a compression retriever with built-in reranking
        base_retriever = kwargs.get("base_retriever")
        if not base_retriever:
            raise ValueError("base_retriever required for compression retriever")

        return ContextualCompressionRetrieverWrapper(
            base_retriever=base_retriever,
            reranker_model=kwargs.get(
                "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
        )

    elif retriever_type == "hybrid":
        # Create hybrid retriever with reranking
        vector_retriever = kwargs.get("vector_retriever")
        bm25_retriever = kwargs.get("bm25_retriever")

        if not vector_retriever or not bm25_retriever:
            raise ValueError(
                "Both vector_retriever and bm25_retriever required for hybrid"
            )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
        )

        return ContextualCompressionRetrieverWrapper(
            base_retriever=ensemble_retriever,
            reranker_model=kwargs.get(
                "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
        )

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


# Example usage
if __name__ == "__main__":
    # Initialize the reranker
    reranker = LangChainReranker(
        model_name=RERANKER_MODEL, top_n=6
    )

    # Example query with candidates
    query = "What are the capital requirements for banks?"
    candidates = [
        {
            "id": "doc1_chunk1",
            "doc_text": "Banks must maintain minimum capital ratios as per Basel III guidelines.",
            "metadata": {"source": "basel_iii.pdf", "page": 12},
            "distance": 0.85,
        },
        {
            "id": "doc2_chunk3",
            "doc_text": "Capital requirements vary by jurisdiction and bank size.",
            "metadata": {"source": "regulation_guide.pdf", "page": 5},
            "distance": 0.92,
        },
    ]

    # Rerank using LangChain compatible method
    reranked, scores = reranker.rerank(query, candidates)

    print(f"Reranked {len(candidates)} candidates:")
    for i, (candidate, score) in enumerate(zip(reranked, scores)):
        print(f"{i + 1}. Score: {score:.3f} - {candidate['doc_text'][:50]}...")
