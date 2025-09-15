"""
Enhanced normalization for CreditExplain RAG system.
Comprehensive PII redaction and regulatory document normalization.
"""

import logging
import re
from typing import List, Dict, Any

from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Comprehensive PII patterns for regulatory compliance
PII_PATTERNS = {
    # Email addresses
    "EMAIL": r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
    # Phone numbers (international formats)
    "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    # Social Security Numbers / National IDs
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "NATIONAL_ID": r"\b(?:\d{6,12}|[A-Z]{2,3}\d{6,9})\b",
    # Credit card numbers (multiple formats)
    "CREDIT_CARD": r"\b(?:\d{4}[- ]?){3}\d{4}\b",
    # Bank account numbers (generic patterns)
    "BANK_ACCOUNT": r"\b(?:account|acct|acc\.?)\s*(?:no|number|#)?\s*[:#]?\s*(\d{8,20})\b",
    # Passport numbers
    "PASSPORT": r"\b(?:passport|ppt)\s*(?:no|number|#)?\s*[:#]?\s*([A-Z]{1,2}\d{6,9}[A-Z]?)\b",
    # Driver's license numbers
    "DRIVER_LICENSE": r"\b(?:driver\'?s?\s*license|dl|lic\.?)\s*(?:no|number|#)?\s*[:#]?\s*([A-Z0-9]{7,15})\b",
    # Person names (common in regulatory documents)
    "PERSON_NAME": r"\b(?:mr|mrs|ms|dr)\.?\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b|\b(customer|client|applicant|borrower)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b",
}

# Regulatory document normalization patterns
REGULATORY_PATTERNS = {
    # Standardize clause references
    "CLAUSE": [(r"(?i)clause\s+(\d+)", r"Clause \1")],
    # Standardize section references
    "SECTION": [(r"(?i)section\s+(\d+)", r"Section \1")],
    # Standardize article references
    "ARTICLE": [(r"(?i)article\s+(\d+)", r"Article \1")],
    # Normalize regulation names
    "BASEL": [(r"(?i)basel\s+iii", "Basel 3"), (r"(?i)basel\s+ii", "Basel 2")],
    # Standardize financial terms
    "FINANCIAL_TERMS": [
        (r"(?i)capital adequacy ratio", "CAR"),
        (r"(?i)loan to value", "LTV"),
        (r"(?i)know your customer", "KYC"),
    ],
    # Normalize currency formats
    "CURRENCY": [
        (r"\$\s*(\d+)", r"$\1"),
        (r"(\d+)\s*USD", r"$\1"),
    ],
    # Standardize percentage formats
    "PERCENTAGE": [
        (r"(\d+)\s*percent", r"\1%"),
        (r"(\d+)\s*\%", r"\1%"),
    ],
}

# Terms that should never be redacted (regulatory content)
REGULATORY_WHITELIST = [
    r"basel\s+[iI]+",
    r"section\s+\d+",
    r"article\s+\d+",
    r"clause\s+\d+",
    r"regulation\s+\d+",
    r"capital\s+requirement",
    r"adequacy\s+ratio",
    r"financial\s+institution",
    r"central\s+bank",
    r"consumer\s+protection",
    r"credit\s+risk",
    r"compliance",
    r"jurisdiction",
]


def normalize_text(
        text: str, redact_pii: bool = True, normalize_regulatory: bool = True
) -> str:
    """
    Enhanced text normalization with comprehensive PII redaction and regulatory formatting.

    Args:
        text: Input text to normalize
        redact_pii: Whether to redact personally identifiable information
        normalize_regulatory: Whether to apply regulatory-specific normalization

    Returns:
        Normalized text with PII redacted and regulatory formatting applied
    """
    if not isinstance(text, str):
        text = str(text)

    # Basic text cleaning
    text = text.strip()
    text = re.sub(r"\s+", " ", text)  # Collapse multiple spaces
    text = re.sub(r"\n+", " ", text)  # Replace newlines with spaces

    # Comprehensive PII redaction
    if redact_pii:
        for pii_type, pattern in PII_PATTERNS.items():
            try:
                text = re.sub(pattern, f"[{pii_type}]", text, flags=re.IGNORECASE)
            except Exception as e:
                logger.warning(f"Failed to redact {pii_type}: {e}")
                continue

    # Regulatory document normalization
    if normalize_regulatory:
        for category, patterns in REGULATORY_PATTERNS.items():
            for pattern, replacement in patterns:
                try:
                    text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
                except Exception as e:
                    logger.warning(f"Failed to apply {category} normalization: {e}")
                    continue

    return text


def normalize_rules(rules: List[Dict], in_place: bool = True, **kwargs) -> List[Dict]:
    """
    Normalize all rules' text with enhanced PII redaction.

    Args:
        rules: List of rule dictionaries
        in_place: Whether to modify the rules in place
        **kwargs: Additional arguments for normalize_text

    Returns:
        Normalized rules
    """
    if not rules:
        return rules

    logger.info(f"Normalizing {len(rules)} rules with PII redaction")

    if in_place:
        for rule in rules:
            rule["text"] = normalize_text(rule.get("text", ""), **kwargs)
        return rules
    else:
        return [
            {**rule, "text": normalize_text(rule.get("text", ""), **kwargs)}
            for rule in rules
        ]


def normalize_documents(documents: List[Document], **kwargs) -> List[Document]:
    """
    Normalize LangChain documents with PII redaction and regulatory formatting.

    Args:
        documents: List of LangChain Documents
        **kwargs: Additional arguments for normalize_text

    Returns:
        Normalized LangChain Documents
    """
    if not documents:
        return documents

    logger.info(f"Normalizing {len(documents)} LangChain documents")

    normalized_docs = []
    for i, doc in enumerate(documents):
        try:
            normalized_content = normalize_text(doc.page_content, **kwargs)
            normalized_doc = Document(
                page_content=normalized_content,
                metadata=doc.metadata.copy(),  # Preserve all metadata
            )
            normalized_docs.append(normalized_doc)
        except Exception as e:
            logger.error(f"Failed to normalize document {i}: {e}")
            # Keep original document as fallback
            normalized_docs.append(doc)

    return normalized_docs


def detect_pii(text: str) -> Dict[str, List[str]]:
    """
    Detect PII in text without redacting it.
    Useful for auditing and compliance reporting.

    Args:
        text: Text to scan for PII

    Returns:
        Dictionary of PII types and detected values
    """
    detected_pii = {}

    for pii_type, pattern in PII_PATTERNS.items():
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                detected_pii[pii_type] = list(set(matches))  # Remove duplicates
        except Exception as e:
            logger.warning(f"Failed to detect {pii_type}: {e}")

    return detected_pii


def create_normalization_report(text: str) -> Dict[str, Any]:
    """
    Create a comprehensive report of normalization changes.

    Args:
        text: Original text

    Returns:
        Report with detected PII, normalization changes, and statistics
    """
    original_text = text
    normalized_text = normalize_text(text)

    return {
        "original_length": len(original_text),
        "normalized_length": len(normalized_text),
        "detected_pii": detect_pii(original_text),
        "changes_applied": original_text != normalized_text,
        "original_sample": (
            original_text[:200] + "..." if len(original_text) > 200 else original_text
        ),
        "normalized_sample": (
            normalized_text[:200] + "..."
            if len(normalized_text) > 200
            else normalized_text
        ),
    }


# Example usage and testing
if __name__ == "__main__":
    # Test with sample regulatory text containing PII
    sample_text = """
    Customer John Smith with SSN 123-45-6789 applied for a loan of $500,000. 
    Contact email: john.smith@email.com, phone: +1-555-0123. 
    According to Basel III guidelines, the capital adequacy ratio must exceed 10%. 
    See Clause 4, Section 2.1 for details on credit risk assessment.
    """

    print("Original text:")
    print(sample_text)
    print("\n" + "=" * 50 + "\n")

    # Normalize with PII redaction
    normalized = normalize_text(sample_text)
    print("Normalized text:")
    print(normalized)
    print("\n" + "=" * 50 + "\n")

    # Generate normalization report
    report = create_normalization_report(sample_text)
    print("Normalization report:")
    for key, value in report.items():
        if key == "detected_pii" and value:
            print(f"Detected PII: {value}")
        elif key not in ["original_sample", "normalized_sample"]:
            print(f"{key}: {value}")
