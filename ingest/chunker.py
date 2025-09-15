"""
Advanced chunking for regulatory documents with LangChain integration.
Preserves semantic structure and compliance metadata for CreditExplain RAG.
"""

import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RegulatoryChunker:
    """Advanced chunker for regulatory documents with semantic awareness."""

    def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            model_name: str = SENTENCE_TRANSFORMER_MODEL,
    ):
        """
        Initialize the regulatory document chunker.

        Args:
            chunk_size: Target size for chunks (in characters)
            chunk_overlap: Overlap between chunks for context preservation
            model_name: Sentence transformer model for semantic chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Primary splitter for most documents
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n## ", "\n\n# ", "\n\n", "\n", ". ", "! ", "? ", " ", ""],
            keep_separator=True,
        )

        # Token-based splitter for model context limits
        self.token_splitter = SentenceTransformersTokenTextSplitter(
            model_name=model_name,
            chunk_size=512,  # tokens, not characters
            chunk_overlap=50,
        )

        logger.info(f"Initialized RegulatoryChunker with chunk_size={chunk_size}")

    def chunk_documents(
            self, documents: List[Document], strategy: str = "recursive"
    ) -> List[Document]:
        """
        Chunk LangChain Documents with regulatory-aware strategies.

        Args:
            documents: List of LangChain Documents to chunk
            strategy: Chunking strategy ("recursive", "token", "semantic")

        Returns:
            List of chunked LangChain Documents with preserved metadata
        """
        if not documents:
            return []

        all_chunks = []

        for doc in documents:
            try:
                if strategy == "token":
                    chunks = self.token_splitter.split_documents([doc])
                elif strategy == "semantic":
                    chunks = self._semantic_chunking(doc)
                else:  # recursive default
                    chunks = self.recursive_splitter.split_documents([doc])

                # Enhance chunks with regulatory metadata
                enhanced_chunks = self._enhance_regulatory_chunks(chunks, doc.metadata)
                all_chunks.extend(enhanced_chunks)

            except Exception as e:
                logger.error(
                    f"Failed to chunk document {doc.metadata.get('source', 'unknown')}: {e}"
                )
                # Keep original document as fallback
                all_chunks.append(doc)

        logger.info(f"Chunked {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks

    def _semantic_chunking(self, document: Document) -> List[Document]:
        """
        Experimental semantic chunking for regulatory documents.
        Groups related regulatory concepts together.
        """
        # For now, fall back to recursive with regulatory-aware separators
        regulatory_separators = [
            "\n\nARTICLE ",
            "\n\nSECTION ",
            "\n\nSUBSECTION ",
            "\n\nCLAUSE ",
            "\n\nParagraph ",
            "\n\n• ",
            "\n\n- ",
            "\n\n* ",
            "\n\n",
            "\n",
            ". ",
        ]

        custom_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=regulatory_separators,
            keep_separator=True,
        )

        return custom_splitter.split_documents([document])

    def _enhance_regulatory_chunks(
            self, chunks: List[Document], original_metadata: Dict[str, Any]
    ) -> List[Document]:
        """
        Enhance chunks with regulatory-specific metadata and provenance.

        Args:
            chunks: List of chunk documents
            original_metadata: Metadata from the original document

        Returns:
            Enhanced chunks with regulatory metadata
        """
        enhanced_chunks = []

        for i, chunk in enumerate(chunks):
            # Create enhanced metadata
            enhanced_metadata = chunk.metadata.copy()

            # Preserve all original metadata
            enhanced_metadata.update(original_metadata)

            # Add chunk-specific metadata
            enhanced_metadata.update(
                {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size_chars": len(chunk.page_content),
                    "chunk_type": self._determine_chunk_type(chunk.page_content),
                }
            )

            # Extract and add regulatory section information if possible
            section_info = self._extract_regulatory_sections(chunk.page_content)
            enhanced_metadata.update(section_info)

            # Create enhanced document
            enhanced_chunk = Document(
                page_content=chunk.page_content, metadata=enhanced_metadata
            )

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    @staticmethod
    def _determine_chunk_type(text: str) -> str:
        """Determine the type of regulatory content in the chunk."""
        text_lower = text.lower()

        if any(
                keyword in text_lower for keyword in ["definition", "means", "shall mean"]
        ):
            return "definition"
        elif any(
                keyword in text_lower for keyword in ["prohibit", "must not", "shall not"]
        ):
            return "prohibition"
        elif any(keyword in text_lower for keyword in ["require", "must", "shall"]):
            return "requirement"
        elif any(keyword in text_lower for keyword in ["penalty", "fine", "sanction"]):
            return "enforcement"
        elif any(
                keyword in text_lower
                for keyword in ["exception", "provided that", "unless"]
        ):
            return "exception"
        else:
            return "general"

    @staticmethod
    def _extract_regulatory_sections(text: str) -> Dict[str, Any]:
        """Extract regulatory section information from chunk text."""
        # Simple pattern matching for regulatory document structure
        lines = text.split("\n")
        section_info = {}

        for line in lines:
            line = line.strip()
            if line.startswith("Article ") or line.startswith("ARTICLE "):
                section_info["article"] = line
            elif line.startswith("Section ") or line.startswith("SECTION "):
                section_info["section"] = line
            elif line.startswith("Clause ") or line.startswith("CLAUSE "):
                section_info["clause"] = line
            elif line.startswith("Subsection ") or line.startswith("SUBSECTION "):
                section_info["subsection"] = line

        return section_info

    def chunk_rules(self, rules: List[Dict], strategy: str = "recursive") -> List[Dict]:
        """
        Backward-compatible function to chunk rules (original interface).

        Args:
            rules: List of rule dictionaries
            strategy: Chunking strategy to use

        Returns:
            List of chunked rules with metadata
        """
        # Convert to LangChain Documents
        documents = []
        for rule in rules:
            doc = Document(
                page_content=rule.get("text", ""),
                metadata={k: v for k, v in rule.items() if k != "text"},
            )
            documents.append(doc)

        # Chunk using advanced method
        chunked_docs = self.chunk_documents(documents, strategy)

        # Convert back to original format
        chunks = []
        for doc in chunked_docs:
            chunk_data = {
                "doc_id": doc.metadata.get("doc_id", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "text": doc.page_content,
                "metadata": doc.metadata,
            }
            chunks.append(chunk_data)

        return chunks


def chunk_rules(rules, chunk_size=1000, chunk_overlap=200):
    """Backward compatible chunk_rules function."""
    chunker = RegulatoryChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker.chunk_rules(rules)


# Example usage
if __name__ == "__main__":
    # Example regulatory document
    sample_rules = [
        {
            "doc_id": "CBN_REG_2023_001",
            "text": "ARTICLE 1: Definitions. For purposes of this regulation, 'Bank' shall mean any financial institution licensed under the Banks and Other Financial Institutions Act. 'Capital' means the sum of tier 1 and tier 2 capital as defined in Basel III guidelines.\n\nARTICLE 2: Capital Requirements. All banks must maintain a minimum capital adequacy ratio of 10%. This requirement shall be reviewed annually by the Central Bank.",
            "jurisdiction": "Nigeria",
            "effective_date": "2023-01-01",
            "authority": "Central Bank of Nigeria",
        }
    ]

    # Test the chunker
    chunker = RegulatoryChunker(chunk_size=500, chunk_overlap=50)

    # Method 1: LangChain way
    documents = [
        Document(page_content=rule["text"], metadata=rule) for rule in sample_rules
    ]
    chunks = chunker.chunk_documents(documents)

    print(f"Chunked into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}: {chunk.metadata.get('chunk_type', 'unknown')}")
        print(f"Content: {chunk.page_content[:100]}...")
        print("---")

    # Method 2: Backward compatible
    chunks_old = chunk_rules(sample_rules)
    print(f"Backward compatible: {len(chunks_old)} chunks")
