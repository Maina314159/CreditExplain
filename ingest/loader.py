"""
Document loader for CreditExplain RAG system.
Uses LangChain for unified document loading and processing.
Handles PDFs, JSON, CSV, and other regulatory documents.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredPDFLoader,
    JSONLoader,
    CSVLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pandas import DataFrame

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditDocumentLoader:
    """Unified document loader for credit/compliance documents with LangChain integration."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document loader.

        Args:
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            f"Initialized CreditDocumentLoader with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def load_pdf_documents(
            self, file_path: str, extract_metadata: bool = True
    ) -> List[Document]:
        """
        Load and chunk PDF documents using LangChain's UnstructuredPDFLoader.

        Args:
            file_path: Path to PDF file or directory
            extract_metadata: Whether to extract metadata from PDF

        Returns:
            List of LangChain Documents with chunks and metadata
        """
        try:
            path = Path(file_path)
            if path.is_dir():
                loader = DirectoryLoader(
                    str(path),
                    glob="**/*.pdf",
                    loader_cls=UnstructuredPDFLoader,
                    use_multithreading=True,
                )
            else:
                loader = UnstructuredPDFLoader(str(path))

            documents = loader.load()

            # Add document-type metadata
            for doc in documents:
                doc.metadata["doc_type"] = "regulation"
                doc.metadata["ingestion_date"] = datetime.now().isoformat()

            # Chunk the documents
            chunks = self.text_splitter.split_documents(documents)
            logger.info(
                f"Loaded and chunked {len(documents)} PDF documents into {len(chunks)} chunks"
            )
            return chunks

        except Exception as e:
            logger.error(f"Failed to load PDF documents from {file_path}: {e}")
            raise

    def load_json_rules(
            self, file_path: str, jq_schema: str = ".",
    ) -> List[Document]:
        """
        Load JSON rules with structured metadata using LangChain's JSONLoader.

        Args:
            file_path: Path to JSON file
            jq_schema: JQ schema for extracting content
            content_key: Key containing the main content

        Returns:
            List of LangChain Documents with structured metadata
        """
        try:
            loader = JSONLoader(
                file_path=file_path,
                jq_schema=jq_schema,
                text_content=False,
                metadata_func=self._extract_json_metadata,
            )

            documents = loader.load()

            # Ensure all documents have basic metadata
            for doc in documents:
                if "doc_type" not in doc.metadata:
                    doc.metadata["doc_type"] = "rule"
                doc.metadata["ingestion_date"] = datetime.now().isoformat()

            logger.info(f"Loaded {len(documents)} JSON rules from {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load JSON rules from {file_path}: {e}")
            raise

    def _extract_json_metadata(
            self, record: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract metadata from JSON records for compliance documents.

        Args:
            record: JSON record being processed
            metadata: Existing metadata

        Returns:
            Enhanced metadata dictionary
        """
        metadata_extracted = metadata.copy()

        # Extract common compliance document metadata
        metadata_fields = [
            "jurisdiction",
            "effective_date",
            "section",
            "clause",
            "page",
            "regulation_id",
            "version",
            "authority",
        ]

        for field in metadata_fields:
            if field in record:
                metadata_extracted[field] = record[field]

        # Add document ID if not present
        if "doc_id" not in metadata_extracted and "id" in record:
            metadata_extracted["doc_id"] = record["id"]

        return metadata_extracted

    def load_csv_data(
            self, file_path: str, source_column: Optional[str] = None
    ) -> List[Document]:
        """
        Load CSV data for borrower information or reference data.

        Args:
            file_path: Path to CSV file
            source_column: Optional column to use as content source

        Returns:
            List of LangChain Documents with CSV data
        """
        try:
            loader = CSVLoader(
                file_path=file_path,
                source_column=source_column,
                metadata_columns=["all"],
            )

            documents = loader.load()

            # Add CSV-specific metadata
            for doc in documents:
                doc.metadata["doc_type"] = "reference_data"
                doc.metadata["ingestion_date"] = datetime.now().isoformat()
                doc.metadata["data_source"] = "csv"

            logger.info(f"Loaded {len(documents)} records from CSV {file_path}")
            return documents

        except Exception as e:
            logger.error(f"Failed to load CSV data from {file_path}: {e}")
            raise

    def _extract_csv_metadata(
            self, record: Dict[str, Any], metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract metadata from CSV records.

        Args:
            record: CSV record as dictionary
            metadata: Existing metadata

        Returns:
            Enhanced metadata dictionary
        """
        metadata_extracted = metadata.copy()

        # Add all CSV columns as metadata for better filtering
        for key, value in record.items():
            if key not in [
                "content",
                "text",
                "document",
            ]:  # Avoid overriding content fields
                metadata_extracted[f"csv_{key}"] = str(value)

        return metadata_extracted

    def load_documents(
            self, file_path: str, doc_type: Optional[str] = None
    ) -> List[Document]:
        """
        Universal document loader that auto-detects file type.

        Args:
            file_path: Path to file or directory
            doc_type: Optional document type override

        Returns:
            List of processed LangChain Documents
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {file_path}")

        if path.is_dir():
            # Load all supported documents from directory
            all_documents = []

            # Load PDFs
            try:
                pdf_docs = self.load_pdf_documents(file_path)
                all_documents.extend(pdf_docs)
            except Exception as e:
                logger.warning(f"Failed to load PDFs from {file_path}: {e}")

            # Load JSONs
            try:
                json_docs = self.load_json_rules(file_path)
                all_documents.extend(json_docs)
            except Exception as e:
                logger.warning(f"Failed to load JSONs from {file_path}: {e}")

            return all_documents

        else:
            # Load single file based on extension
            extension = path.suffix.lower()

            if extension == ".pdf":
                return self.load_pdf_documents(file_path)
            elif extension == ".json":
                return self.load_json_rules(file_path)
            elif extension == ".csv":
                return self.load_csv_data(file_path)
            else:
                # Fallback to unstructured loader
                try:
                    loader = UnstructuredFileLoader(file_path)
                    documents = loader.load()
                    for doc in documents:
                        doc.metadata["doc_type"] = doc_type or "unknown"
                        doc.metadata["ingestion_date"] = datetime.now().isoformat()
                    return documents
                except Exception as e:
                    logger.error(
                        f"Unsupported file format: {extension} for {file_path}"
                    )
                    raise


# Maintain backward compatibility functions
def load_rules(file_path: str) -> List[Dict[str, Any]]:
    """Backward compatible rules loading."""
    loader = CreditDocumentLoader()
    documents = loader.load_json_rules(file_path)
    return [
        {
            "doc_id": doc.metadata.get("doc_id", f"doc_{i}"),
            **doc.metadata,
            "text": doc.page_content,
        }
        for i, doc in enumerate(documents)
    ]


def load_borrowers(file_path: str, as_dicts: bool = True) -> Union[List[Dict[str, Any]], DataFrame]:
    """Backward compatible borrower loading."""
    loader = CreditDocumentLoader()
    documents = loader.load_csv_data(file_path)

    if as_dicts:
        return [
            {
                "doc_id": doc.metadata.get("source", f"row_{i}"),
                **doc.metadata,
                "text": doc.page_content,
            }
            for i, doc in enumerate(documents)
        ]
    else:
        records = []
        for doc in documents:
            record = doc.metadata.copy()
            record["text"] = doc.page_content
            records.append(record)
        return pd.DataFrame(records)


if __name__ == "__main__":
    # Example usage
    loader = CreditDocumentLoader()

    # Load regulatory PDFs
    try:
        pdf_docs = loader.load_pdf_documents("data/raw/")
        print(f"Loaded {len(pdf_docs)} PDF document chunks")
    except Exception as e:
        print(f"PDF loading failed: {e}")

    # Load JSON rules
    try:
        json_docs = loader.load_json_rules("data/interim/rules.json")
        print(f"Loaded {len(json_docs)} JSON rules")
    except Exception as e:
        print(f"JSON loading failed: {e}")

    # Load CSV data
    try:
        csv_docs = loader.load_csv_data("data/interim/borrowers.csv")
        print(f"Loaded {len(csv_docs)} CSV records")
    except Exception as e:
        print(f"CSV loading failed: {e}")
