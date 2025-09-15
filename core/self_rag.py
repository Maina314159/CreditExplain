"""
Agentic Self-RAG System for Credit Compliance Explanation.
Uses LangChain for orchestration and follows SELF-RAG principles.
"""

import logging
import time
import uuid
from typing import List, Dict, Optional

from dotenv import load_dotenv

from core.critic import GroqCritic as Critic
from core.generator import GroqGenerator as Generator
from core.provenance import ProvenanceLogger, AuditConfig
from core.reranker import LangChainReranker as Reranker
from core.retrieval import LangChainEmbeddingModel as EmbeddingModel, VectorRetriever

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vectorstore directory
VECTORSTORE_DIR = "vectorstore/chroma"


class SelfRAG:
    """Agentic RAG system with self-reflection and adaptive retrieval."""

    def __init__(
            self,
            embed_model: Optional[EmbeddingModel] = None,
            retriever: Optional[VectorRetriever] = None,
            reranker: Optional[Reranker] = None,
            generator: Optional[Generator] = None,
            critic: Optional[Critic] = None,
            top_k: int = 50,
            top_n: int = 6,
            fully_supported_threshold: float = 0.7,
    ):
        """
        Initialize the Self-RAG system.

        Args:
            embed_model: Embedding model for query encoding
            retriever: Vector retriever for document search
            reranker: Cross-encoder for passage re-ranking
            generator: LLM for answer generation
            critic: Model for retrieval decision and answer scoring
            top_k: Number of initial passages to retrieve
            top_n: Number of passages to re-rank and process
            fully_supported_threshold: Minimum support score for valid answer
        """
        self.embed_model = embed_model or EmbeddingModel()
        self.retriever = retriever or VectorRetriever()
        self.reranker = reranker or Reranker()
        self.generator = generator or Generator()
        self.critic = critic or Critic()
        self.top_k = top_k
        self.top_n = top_n
        self.fully_supported_threshold = fully_supported_threshold
        self.provenance_logger = ProvenanceLogger(AuditConfig(log_to_console=True))

        # Weighting for candidate selection
        self.selection_weights = {
            "isrel": 0.45,  # Relevance weight
            "issup": 0.40,  # Support weight
            "isuse": 0.15,  # Usefulness weight
        }

        logger.info(
            "SelfRAG system initialized with adaptive retrieval and self-reflection"
        )

    def _calculate_combined_score(self, score_components: Dict[str, float]) -> float:
        """Calculate weighted combined score from critic components."""
        return sum(
            self.selection_weights.get(key, 0) * score_components.get(key, 0)
            for key in ["isrel", "issup", "isuse"]
        )

    def _process_candidate(
            self, query: str, candidate_passage: Dict, candidate_index: int
    ) -> Optional[Dict]:
        """Process a single candidate passage through generation and scoring."""
        try:
            # Generate answer for this specific passage
            generated_answer = self.generator.generate_from_passages(
                query, [candidate_passage]
            )

            # Critic scoring for the generated answer
            score_components = self.critic.score_candidate(
                query=query,
                answer=generated_answer.get("explanation", ""),
                passage_text=candidate_passage.get("doc_text", ""),
            )

            combined_score = self._calculate_combined_score(score_components)

            return {
                "candidate_answer": generated_answer,
                "passage": candidate_passage,
                "score_components": score_components,
                "combined_score": combined_score,
                "candidate_index": candidate_index,
            }

        except Exception as e:
            logger.error(f"Error processing candidate {candidate_index}: {e}")
            return None

    def run(self, query: str, case_id: Optional[str] = None) -> Dict:
        """
        Execute the full Self-RAG pipeline for a given query.

        Args:
            query: User's natural language query
            case_id: Optional case identifier for auditing

        Returns:
            Dictionary containing answer, provenance, and metadata
        """
        run_id = str(uuid.uuid4())
        start_time = time.time()
        global processing_time
        processing_time = 0
        logger.info(f"Starting Self-RAG processing for query: {query[:100]}...")

        try:
            # 1) Adaptive Retrieval Decision
            retrieval_decision = self.critic.decide_retrieve(query)
            should_retrieve = retrieval_decision.get("retrieve", True)

            if not should_retrieve:
                logger.info("Critic decided retrieval not needed")
                # Generate response without retrieval
                generated_response = self.generator.generate_from_passages(query, [])
                processing_time = time.time() - start_time

                # Build provenance metadata for no-retrieval case
                provenance_meta = {
                    "retrieval_decision": retrieval_decision,
                    "retrieval_performed": False,
                    "model_versions": {
                        "critic": self.critic.config.model_name,
                        "generator": self.generator.config.model_name,
                    },
                    "status": "success",
                }

                # Log audit
                audit_path = self.provenance_logger.write_audit(
                    run_id,
                    query,
                    [],
                    generated_response,
                    provenance_meta,
                    processing_time,
                    case_id,
                )

                return {
                    "run_id": run_id,
                    "answer": generated_response,
                    "audit_id": audit_path,
                    "retrieval_performed": False,
                }

            # 2) Embed query and retrieve initial candidates
            logger.info("Retrieving relevant passages...")
            query_embedding = self.embed_model.embed([query])[0]
            initial_candidates = self.retriever.retrieve(query_embedding.tolist(), k=self.top_k)

            if not initial_candidates:
                logger.warning("No passages retrieved from vector store")
                return self._handle_empty_retrieval(run_id, query, start_time, case_id)

            # 3) Re-rank passages for precision
            logger.info(f"Re-ranking {len(initial_candidates)} passages...")
            top_passages, rerank_scores = self.reranker.rerank(
                query, initial_candidates, top_n=self.top_n
            )

            # 4) Generate and score candidate answers for top passages
            logger.info(
                f"Generating and scoring answers for {len(top_passages)} top passages..."
            )
            candidate_results = []

            for idx, passage in enumerate(top_passages):
                result = self._process_candidate(query, passage, idx)
                if result:
                    candidate_results.append(result)

            if not candidate_results:
                logger.error("All candidate processing failed")
                return self._handle_processing_failure(
                    run_id, query, top_passages, start_time, case_id
                )

            # 5) Select best candidate based on combined score
            candidate_results.sort(key=lambda x: x["combined_score"], reverse=True)
            best_candidate = candidate_results[0]
            best_score_components = best_candidate["score_components"]

            # 6) Check if answer is sufficiently supported
            if best_score_components.get("issup", 0.0) < self.fully_supported_threshold:
                logger.warning("Best answer has insufficient support")
                return self._handle_insufficient_support(
                    run_id,
                    query,
                    top_passages,
                    candidate_results,
                    best_score_components,
                    start_time,
                    case_id,
                )

            # 7) Prepare successful response with PII redaction
            final_answer = best_candidate["candidate_answer"]

            # TODO: Implement more granular PII redaction if needed

            # Generate follow-up questions
            follow_up_questions = self.generator.generate_follow_ups(
                query, final_answer, top_passages
            )
            final_answer["follow_up_questions"] = follow_up_questions

            # Calculate processing time for successful path
            processing_time = time.time() - start_time

            # Build comprehensive provenance_meta for the new ProvenanceLogger
            provenance_meta = {
                "retrieval_decision": retrieval_decision,
                "retrieval_performed": True,
                "retrieval_count": len(initial_candidates),
                "rerank_scores": rerank_scores,
                "selected_candidate_index": 0,
                "selected_candidate_scores": best_score_components,
                "model_versions": {
                    "critic": self.critic.config.model_name,
                    "generator": self.generator.config.model_name,
                    "embedding": self.embed_model.model_name,
                },
                "status": "success",
            }

            # Log audit using the new ProvenanceLogger
            audit_path = self.provenance_logger.write_audit(
                run_id,
                query,
                top_passages,
                final_answer,
                provenance_meta,
                processing_time,
                case_id,
            )

            logger.info(f"Successfully processed query in {processing_time:.2f}s")

            return {
                "run_id": run_id,
                "answer": final_answer,
                "provenance_meta": provenance_meta,
                "audit_path": audit_path,
                "retrieval_performed": True,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Unexpected error in Self-RAG pipeline: {e}")
            return self._handle_pipeline_error(run_id, query, e, start_time, case_id)

    def _handle_empty_retrieval(
            self, run_id: str, query: str, start_time: float, case_id: str
    ) -> Dict:
        """Handle case where no passages are retrieved."""
        processing_time = time.time() - start_time
        error_response = {
            "explanation": "I couldn't find any relevant documents to answer your question. Please try rephrasing or ensure relevant documents are uploaded.",
            "citations": [],
            "confidence": "LOW",
        }

        provenance_meta = {
            "error": "empty_retrieval",
            "retrieval_performed": True,
            "status": "error",
        }

        audit_path = self.provenance_logger.write_audit(
            run_id, query, [], error_response, provenance_meta, processing_time, case_id
        )

        return {
            "run_id": run_id,
            "answer": error_response,
            "provenance_meta": provenance_meta,
            "audit_id": audit_path,
            "retrieval_performed": True,
            "error": "empty_retrieval",
        }

    def _handle_insufficient_support(
            self,
            run_id: str,
            query: str,
            passages: List[Dict],
            candidates: List[Dict],
            scores: Dict[str, float],
            start_time: float,
            case_id: str,
    ) -> Dict:
        """Handle case where no answer meets the support threshold."""
        processing_time = time.time() - start_time

        response = {
            "status": "insufficient_support",
            "message": "The available documents don't provide sufficient support for a confident answer. You may need to provide additional documentation.",
            "best_attempt": candidates[0]["candidate_answer"] if candidates else None,
            "support_score": scores.get("issup", 0.0),
        }

        provenance_meta = {
            "error": "insufficient_support",
            "retrieval_performed": True,
            "status": "error",
        }

        audit_path = self.provenance_logger.write_audit(
            run_id, query, [], response, provenance_meta, processing_time, case_id
        )

        return {
            "run_id": run_id,
            "answer": response,
            "provenance_meta": provenance_meta,
            "audit_id": audit_path,
            "retrieval_performed": True,
            "error": "insufficient_support",
        }

    def _handle_processing_failure(
            self,
            run_id: str,
            query: str,
            passages: List[Dict],
            start_time: float,
            case_id: str,
    ) -> Dict:
        """Handle case where all candidate processing fails."""
        processing_time = time.time() - start_time

        error_response = {
            "explanation": "I encountered an error while processing the documents. Please try again or contact support if the issue persists.",
            "citations": [],
            "confidence": "LOW",
        }

        provenance_meta = {
            "error": "processing_failure",
            "retrieval_performed": True,
            "status": "error",
        }

        audit_path = self.provenance_logger.write_audit(
            run_id, query, [], error_response, provenance_meta, processing_time, case_id
        )
        return {
            "run_id": run_id,
            "answer": error_response,
            "provenance_meta": provenance_meta,
            "audit_id": audit_path,
            "retrieval_performed": True,
            "error": "processing_failure",
        }

    def _handle_pipeline_error(
            self, run_id: str, query: str, error: Exception, start_time: float, case_id: str
    ) -> Dict:
        """Handle unexpected errors in the pipeline."""
        processing_time = time.time() - start_time

        error_response = {
            "explanation": "A system error occurred while processing your request. Our team has been notified.",
            "citations": [],
            "confidence": "LOW",
        }

        provenance_meta = {
            "error": "pipeline_error",
            "retrieval_performed": True,
            "status": "error",
        }

        audit_path = self.provenance_logger.write_audit(
            run_id, query, [], error_response, provenance_meta, processing_time, case_id
        )

        return {
            "run_id": run_id,
            "answer": error_response,
            "provenance_meta": provenance_meta,
            "audit_id": audit_path,
            "retrieval_performed": False,
            "error": "pipeline_error",
        }


if __name__ == "__main__":
    """Interactive CLI for testing the Self-RAG system."""

    # Initialize the Self-RAG system
    rag_system = SelfRAG()

    print("🧠 Welcome to CreditExplain RAG System!")
    print("=" * 50)
    print(
        "I'm a specialized AI assistant for financial compliance and credit regulations."
    )
    print(
        "I can answer questions based on regulatory documents, internal policies, and model cards."
    )
    print("\n💡 Try asking me about:")
    print("  • Banking regulations in Nigeria or Kenya")
    print("  • KYC/AML requirements (FATF Recommendations)")
    print("  • Consumer protection rules")
    print("  • Capital requirements for financial institutions")
    print("  • How our credit approval model works")
    print("\n⏎ Press Enter without typing (or type 'quit') to exit.")
    print("=" * 50)

    while True:
        try:
            # Get user input
            user_query = input("\n🤔 Your question: ").strip()

            # Exit if user presses Enter without input
            if not user_query:
                print("\n👋 Thank you for using CreditExplain. Goodbye!")
                break

            # Check for exit commands
            if user_query.lower() in ["exit", "quit", "bye"]:
                print("\n👋 Thank you for using CreditExplain. Goodbye!")
                break

            print(f"\n🔍 Processing your query...")

            # Run the RAG pipeline
            result = rag_system.run(user_query)

            # Display results
            print("\n✅ Answer:")
            print("-" * 40)

            if "answer" in result:
                answer = result["answer"]

                # Handle different answer formats
                if "explanation" in answer:
                    print(f"{answer['explanation']}")
                elif "message" in answer:
                    print(f"{answer['message']}")
                    if "best_attempt" in answer and answer["best_attempt"]:
                        print(
                            f"\n📋 Best attempt: {answer['best_attempt'].get('explanation', '')[:200]}..."
                        )

                # Display confidence
                if "confidence" in answer:
                    confidence = answer.get("confidence", "UNKNOWN")
                    confidence_icon = (
                        "🟢"
                        if confidence == "HIGH"
                        else "🟡" if confidence == "MEDIUM" else "🔴"
                    )
                    print(f"\n{confidence_icon} Confidence: {confidence}")

                # Display citations if available
                if "citations" in answer and answer["citations"]:
                    print(f"\n📚 Citations:")
                    for i, citation in enumerate(answer["citations"], 1):
                        doc_id = citation.get("doc_id", "Unknown Document")
                        print(
                            f"   {i}. [{doc_id}] {citation.get('text_excerpt', '')[:100]}..."
                        )

                # Display follow-up questions
                if "follow_up_questions" in answer and answer["follow_up_questions"]:
                    print(f"\n💭 Suggested follow-up questions:")
                    for i, question in enumerate(
                            answer["follow_up_questions"][:3], 1
                    ):  # Show top 3
                        print(f"   {i}. {question}")

            # Display errors if any
            if "error" in result:
                error_type = result.get("error", "unknown_error")
                print(f"\n❌ Error Type: {error_type.replace('_', ' ').title()}")

            # Display performance metrics
            processing_time = result.get("processing_time", 0)
            print(f"\n⏱️  Processing time: {processing_time:.2f}s")
            print(
                f"📊 Retrieval performed: {'Yes' if result.get('retrieval_performed', False) else 'No'}"
            )

            print("-" * 40)

        except KeyboardInterrupt:
            print("\n\n👋 Thank you for using CreditExplain. Goodbye!")
            break
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            print("Please try again or contact support.")
