"""
Provenance and audit logging for Self-RAG system.
Captures full agentic loop for compliance and evaluation.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import aiofiles
from langchain_core.callbacks import BaseCallbackHandler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AUDIT_DIR = "./audit/"


@dataclass
class AuditConfig:
    """Configuration for audit logging."""

    audit_dir: str = AUDIT_DIR
    max_file_size_mb: int = 10  # Rotate files after this size
    use_async: bool = True  # Use async file operations
    log_to_console: bool = False  # Log to console


@dataclass
class CandidateProvenance:
    """Structured data for each retrieved candidate."""

    candidate_id: str
    doc_text_preview: str  # First 200 chars
    metadata: Dict[str, Any]
    retrieval_score: float  # Original similarity score
    rerank_score: Optional[float] = None
    isrel_score: Optional[float] = None
    issup_score: Optional[float] = None
    isuse_score: Optional[float] = None


@dataclass
class AuditRecord:
    """Structured audit record for full SELF-RAG provenance."""

    run_id: str
    timestamp: str
    case_id: Optional[str]
    query: str

    # Agentic decision-making
    retrieval_decision: Dict[str, Any]  # Critic's decision
    retrieval_performed: bool

    # Retrieval & ranking
    retrieved_count: int
    top_candidates: List[CandidateProvenance]
    rerank_scores: List[float]

    # Generation & selection
    selected_candidate_index: Optional[int]
    selected_candidate_scores: Optional[Dict[str, float]]
    confidence: str  # HIGH/MEDIUM/LOW

    # Final result
    result: Dict[str, Any]
    follow_up_questions: List[str]

    # Performance
    latency_s: float
    model_versions: Dict[str, str]  # critic, generator, embedding models

    # System info
    error: Optional[str] = None
    status: str = "success"  # success, insufficient_support, error


class ProvenanceLogger:
    """LangChain-compatible provenance logger with structured auditing."""

    def __init__(self, config: Optional[AuditConfig] = None):
        self.config = config or AuditConfig()
        os.makedirs(self.config.audit_dir, exist_ok=True)
        self.current_file = self._get_current_audit_file()

    def _get_current_audit_file(self) -> str:
        """Get the current audit file, rotating if necessary."""
        timestamp = datetime.now().strftime("%Y%m%d")
        return os.path.join(self.config.audit_dir, f"audit_{timestamp}.jsonl")

    async def _write_async(self, record: Dict[str, Any]):
        """Async write to audit file."""
        try:
            async with aiofiles.open(self.current_file, "a", encoding="utf-8") as f:
                await f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Async audit write failed: {e}")
            self._write_sync(record)  # Fallback to sync

    def _write_sync(self, record: Dict[str, Any]):
        """Sync write to audit file."""
        try:
            with open(self.current_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Sync audit write failed: {e}")

    @staticmethod
    def create_audit_record(
            run_id: str,
            query: str,
            top_candidates: List[Dict],
            result: Dict[str, Any],
            provenance_meta: Dict[str, Any],
            latency_s: float,
            case_id: Optional[str] = None,
    ) -> AuditRecord:
        """Create a structured audit record from SELF-RAG execution."""

        # Build candidate provenance
        candidate_provenance = []
        for candidate in top_candidates:
            candidate_provenance.append(
                CandidateProvenance(
                    candidate_id=candidate.get("id", "unknown"),
                    doc_text_preview=candidate.get("doc_text", "")[:200],
                    metadata=candidate.get("metadata", {}),
                    retrieval_score=candidate.get("distance", 0.0),
                    rerank_score=candidate.get("rerank_score"),
                    isrel_score=candidate.get("isrel_score"),
                    issup_score=candidate.get("issup_score"),
                    isuse_score=candidate.get("isuse_score"),
                )
            )

        return AuditRecord(
            run_id=run_id,
            timestamp=datetime.now().isoformat() + "Z",
            case_id=case_id,
            query=query,
            retrieval_decision=provenance_meta.get("retrieval_decision", {}),
            retrieval_performed=provenance_meta.get("retrieval_performed", True),
            retrieved_count=provenance_meta.get("retrieval_count", 0),
            top_candidates=candidate_provenance,
            rerank_scores=provenance_meta.get("rerank_scores", []),
            selected_candidate_index=provenance_meta.get("selected_candidate_index"),
            selected_candidate_scores=provenance_meta.get(
                "selected_candidate_scores", {}
            ),
            confidence=result.get("confidence", "UNKNOWN"),
            result=result,
            follow_up_questions=result.get("follow_up_questions", []),
            latency_s=latency_s,
            model_versions=provenance_meta.get("model_versions", {}),
            error=provenance_meta.get("error"),
            status=provenance_meta.get("status", "success"),
        )

    async def write_audit_async(
            self,
            run_id: str,
            query: str,
            top_candidates: List[Dict],
            result: Dict[str, Any],
            provenance_meta: Dict[str, Any],
            latency_s: float,
            case_id: Optional[str] = None,
    ) -> str:
        """Async version of audit writing."""
        audit_record = self.create_audit_record(
            run_id, query, top_candidates, result, provenance_meta, latency_s, case_id
        )

        record_dict = asdict(audit_record)
        await self._write_async(record_dict)

        if self.config.log_to_console:
            logger.info(f"Audit logged: {run_id}, Latency: {latency_s:.2f}s")

        return self.current_file

    def write_audit(
            self,
            run_id: str,
            query: str,
            top_candidates: List[Dict],
            result: Dict[str, Any],
            provenance_meta: Dict[str, Any],
            latency_s: float,
            case_id: Optional[str] = None,
    ) -> str:
        """Sync version for backward compatibility."""
        audit_record = self.create_audit_record(
            run_id, query, top_candidates, result, provenance_meta, latency_s, case_id
        )

        record_dict = asdict(audit_record)
        self._write_sync(record_dict)

        if self.config.log_to_console:
            logger.info(f"Audit logged: {run_id}, Latency: {latency_s:.2f}s")

        return self.current_file


class ProvenanceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for provenance logging."""

    def __init__(self, provenance_logger: ProvenanceLogger):
        self.provenance_logger = provenance_logger
        self.start_time = None

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ):
        """Record start time for latency calculation."""
        self.start_time = time.time()

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Log provenance when chain completes."""
        if self.start_time:
            latency_s = time.time() - self.start_time
            # Extract audit data from outputs and log
            # This would be customized based on the chain structure
