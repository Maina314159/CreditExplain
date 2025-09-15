"""
Critic component for Self-RAG system.
Uses open-source models via Groq for retrieval decisions and answer scoring.
Implements LangChain runnables for better integration and reliability.
"""

import logging
import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields
from typing import Dict, Optional, List, Set

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_groq import ChatGroq

from core.prompts import CRITIC_RETRIEVE_PROMPT, CRITIC_SCORE_PROMPT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CRITIC_MODEL = "llama-3.3-70b-versatile"


@dataclass
class CriticConfig:
    """Configuration for the critic component."""

    model_name: str = CRITIC_MODEL
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 30
    max_tokens: int = 1024
    fallback_retrieve: bool = True
    fallback_scores: Dict[str, float] = None

    def __post_init__(self):
        if self.fallback_scores is None:
            self.fallback_scores = {"isrel": 0.5, "issup": 0.5, "isuse": 0.5}

    @classmethod
    def get_field_names(cls) -> Set[str]:
        return {f.name for f in fields(cls)}


class BaseCritic(ABC):
    """Abstract base class for critic components."""

    def __init__(self, config: Optional[CriticConfig] = None):
        """
        Initialize the critic component.

        Args:
            config: Critic config parameters.
        """
        self.config = config or CriticConfig()
        self.llm = None
        self.retrieve_chain = None
        self.score_chain = None

    @abstractmethod
    def _initialize_llm(self):
        """Initialize the language model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _create_retrieve_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for retrieval decision."""
        pass

    @abstractmethod
    def _create_score_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for answer scoring."""
        pass

    def decide_retrieve(self, query: str) -> Dict:
        """
        Decide whether retrieval is needed for the given query.

        Args:
            query: User's input query

        Returns:
            Dictionary with retrieval decision and notes
        """
        try:
            result = self.retrieve_chain.invoke({"query": query})

            # Validate result structure
            if not isinstance(result, dict):
                raise ValueError("Invalid response format")

            if "retrieve" not in result:
                result["retrieve"] = self.config.fallback_retrieve

            return result

        except (OutputParserException, ValueError, Exception) as e:
            logger.warning(f"Retrieval decision failed: {e}. Using fallback.")
            return {
                "retrieve": self.config.fallback_retrieve,
                "notes": f"Fallback due to error: {str(e)}",
            }

    def score_candidate(self, query: str, answer: str, passage_text: str) -> Dict:
        """
        Score a candidate answer against a source passage.

        Args:
            query: Original user query
            answer: Generated answer to evaluate
            passage_text: Source passage text for validation

        Returns:
            Dictionary with relevance, support, and usefulness scores
        """
        try:
            # Truncate long texts to avoid token limits
            truncated_answer = answer[:2000] if len(answer) > 2000 else answer
            truncated_passage = (
                passage_text[:2000] if len(passage_text) > 2000 else passage_text
            )

            result = self.score_chain.invoke(
                {
                    "query": query,
                    "answer": truncated_answer,
                    "passage": truncated_passage,
                }
            )

            # Validate and normalize scores
            validated_scores = self._validate_scores(result)
            return validated_scores

        except (OutputParserException, ValueError, Exception) as e:
            logger.warning(f"Scoring failed: {e}. Using fallback scores.")
            return {
                **self.config.fallback_scores,
                "notes": f"Fallback due to error: {str(e)}",
            }

    def _validate_scores(self, scores: Dict) -> Dict:
        """
        Validate and normalize critic scores.

        Args:
            scores: Raw scores from LLM

        Returns:
            Validated and normalized scores
        """
        validated = scores.copy()

        # Ensure all required keys are present
        for key in ["isrel", "issup", "isuse"]:
            if key not in validated:
                validated[key] = self.config.fallback_scores[key]
                logger.warning(
                    f"Missing score key '{key}', using fallback: {validated[key]}"
                )

        # Normalize scores to 0.0-1.0 range
        for key in ["isrel", "issup", "isuse"]:
            try:
                score = float(validated[key])
                validated[key] = max(0.0, min(1.0, score))  # Clamp to [0, 1]
            except (ValueError, TypeError):
                validated[key] = self.config.fallback_scores[key]
                logger.warning(
                    f"Invalid score for '{key}', using fallback: {validated[key]}"
                )

        # Ensure notes field exists
        if "notes" not in validated:
            validated["notes"] = ""

        return validated

    def batch_score_candidates(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Score multiple candidate answers in a batch.

        Args:
            query: Original user query
            candidates: List of candidate dictionaries with 'answer' and 'passage_text'

        Returns:
            List of scored candidates
        """
        scored_candidates = []

        for i, candidate in enumerate(candidates):
            try:
                scores = self.score_candidate(
                    query=query,
                    answer=candidate.get("answer", ""),
                    passage_text=candidate.get("passage_text", ""),
                )

                scored_candidates.append(
                    {**candidate, "scores": scores, "candidate_index": i}
                )

            except Exception as e:
                logger.error(f"Failed to score candidate {i}: {e}")
                # Add fallback scores for failed candidates
                scored_candidates.append(
                    {
                        **candidate,
                        "scores": {
                            **self.config.fallback_scores,
                            "notes": f"Scoring failed: {e}",
                        },
                        "candidate_index": i,
                    }
                )

        return scored_candidates


class GroqCritic(BaseCritic):
    """Critic component using open-source models via Groq API."""

    def __init__(self, config: Optional[CriticConfig] = None):
        """
        Initialize the Groq-based critic.

        Args:
            config: Critic configuration parameters
        """
        super().__init__(config)
        self._initialize_llm()
        self.retrieve_chain = self._create_retrieve_chain()
        self.score_chain = self._create_score_chain()
        self.config = config or CriticConfig()

        logger.info(f"GroqCritic initialized with model: {self.config.model_name}")

    def _initialize_llm(self):
        """Initialize the Groq chat model."""
        self.llm = ChatGroq(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def _create_retrieve_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for retrieval decision."""
        prompt = ChatPromptTemplate.from_template(CRITIC_RETRIEVE_PROMPT)
        parser = JsonOutputParser()

        return prompt | self.llm | parser

    def _create_score_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for answer scoring."""
        prompt = ChatPromptTemplate.from_template(CRITIC_SCORE_PROMPT)
        parser = JsonOutputParser()

        return prompt | self.llm | parser


class OllamaCritic(BaseCritic):
    """Critic component using local models via Ollama."""

    def __init__(self, config: Optional[CriticConfig] = None, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama-based critic for offline use.

        Args:
            config: Ollama model name
            base_url: Ollama server URL
        """
        # Override config with Ollama-specific defaults if not provided
        ollama_config = config or CriticConfig(
            model_name="llama3.1:8b",
            max_tokens=1024
        )
        super().__init__(ollama_config)
        self.base_url = base_url
        self._initialize_llm()
        self.retrieve_chain = self._create_retrieve_chain()
        self.score_chain = self._create_score_chain()

        logger.info(f"OllamaCritic initialized with model: {self.config.model_name}")

    def _initialize_llm(self):
        """Initialize the Ollama language model."""
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            logger.error(
                "langchain-ollama not installed. Please install with: pip install langchain-ollama"
            )
            raise

        self.llm = ChatOllama(
            model=self.config.model_name,
            temperature=self.config.temperature,
            base_url=self.base_url
        )

    def _create_retrieve_chain(self):
        """Create retrieval decision chain for Ollama."""
        prompt = ChatPromptTemplate.from_template(CRITIC_RETRIEVE_PROMPT)
        parser = JsonOutputParser()

        return prompt | self.llm | parser

    def _create_score_chain(self):
        """Create scoring chain for Ollama."""
        prompt = ChatPromptTemplate.from_template(CRITIC_SCORE_PROMPT)
        parser = JsonOutputParser()

        return prompt | self.llm | parser


class CriticFactory:
    """Factory class for creating critic instances."""

    _registry = {
        "groq": GroqCritic,
        "ollama": OllamaCritic,
    }

    @classmethod
    def create_critic(cls, critic_type: str = "groq", **kwargs) -> BaseCritic:
        """
        Factory method to create critic instances.

        Args:
            critic_type: Type of critic to create ('groq' or 'ollama')
            **kwargs: Additional arguments for critic configuration

        Returns:
            Configured critic instance
        """
        critic_type = critic_type.lower()

        if critic_type not in cls._registry:
            raise ValueError(
                f"Unknown critic type: {critic_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        # Get the field names from the dataclass
        config_field_names = CriticConfig.get_field_names()

        # Extract config parameters if provided
        config_params = {k: v for k, v in kwargs.items()
                         if k in config_field_names}

        # Create config if any parameters provided
        config = CriticConfig(**config_params) if config_params else None

        # Extract non-config parameters
        other_params = {k: v for k, v in kwargs.items()
                        if k not in config_field_names}

        critic_class = cls._registry[critic_type]

        if config and other_params:
            return critic_class(config=config, **other_params)
        elif config:
            return critic_class(config=config)
        elif other_params:
            return critic_class(**other_params)
        else:
            return critic_class()


# Convenience function for backward compatibility
def create_critic(critic_type: str = "groq", **kwargs) -> BaseCritic:
    """
    Factory function to create critic instances.

    Args:
        critic_type: Type of critic to create ('groq' or 'ollama')
        **kwargs: Additional arguments for critic configuration

    Returns:
        Configured critic instance
    """
    return CriticFactory.create_critic(critic_type, **kwargs)
