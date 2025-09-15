"""
Generator component for Self-RAG system.
Uses open-source models via Groq for answer generation and follow-up question creation.
Implements LangChain runnables for better integration and reliability.
"""

import logging
import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, fields
from typing import List, Dict, Optional, Set

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_groq import ChatGroq

from core.prompts import GENERATOR_PROMPT, FOLLOW_UP_PROMPT

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GENERATOR_MODEL = "llama-3.3-70b-versatile"


@dataclass
class GeneratorConfig:
    """Configuration for the generator component."""

    model_name: str = GENERATOR_MODEL
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 3
    timeout: int = 60
    max_passage_length: int = 1000
    max_follow_up_questions: int = 5

    @classmethod
    def get_field_names(cls) -> Set[str]:
        return {f.name for f in fields(cls)}


class BaseGenerator(ABC):
    """Abstract base class for generator component implementations."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the generator component.

        Args:
            config: Generator config parameters.
        """
        self.config = config or GeneratorConfig()
        self.llm = None
        self.generator_chain = None
        self.follow_up_chain = None

    @abstractmethod
    def _initialize_llm(self):
        """Initialize the language model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _create_generator_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for answer generation."""
        pass

    @abstractmethod
    def _create_follow_up_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for generation of follow-up questions."""
        pass

    def _format_passages_block(self, passages: List[Dict]) -> str:
        """
        Format passages into a structured block for the prompt.

        Args:
            passages: List of passage dictionaries.

        Returns:
            Formatted passages block string.
        """
        if not passages:
            return "No relevant passages avaiilable."

        passages_block = ""
        for passage in passages:
            passage_id = passage.get("id", "unknown_id")
            doc_type = passage.get("metadata", {}).get("doc_type", "document")

            text = passage.get("doc_text", "")[: self.config.max_passage_length]
            passages_block += f"[ID: {passage_id} | Type: {doc_type}]\n{text}\n\n"

        return passages_block.strip()

    def generate_from_passages(self, query: str, passages: List[Dict]) -> Dict:
        """
        Generate an evidence-based answer from provided passages.

        Args:
            query: User's input query
            passages: List of relevant passages with metadata

        Returns:
            Dictionary containing explanation, citations, and confidence
        """
        try:
            # Format passages for the prompt
            passages_block = self._format_passages_block(passages)

            # Invoke the generation chain
            result = self.generator_chain.invoke(
                {"query": query, "passages_block": passages_block}
            )

            # Validate and enhance the result
            validated_result = self._validate_generation_result(result, query)
            validated_result["model_version"] = self.config.model_name
            validated_result["passages_used"] = len(passages)

            return validated_result

        except (OutputParserException, ValueError, Exception) as e:
            logger.error(f"Answer generation failed: {e}")
            return self._create_fallback_response(query, passages, str(e))

    def generate_follow_ups(
            self, query: str, answer: Dict, passages: List[Dict]
    ) -> List[str]:
        """
        Generate relevant follow-up questions based on the context.

        Args:
            query: Original user query
            answer: Generated answer dictionary
            passages: List of passages used for generation

        Returns:
            List of follow-up questions
        """
        try:
            # Prepare context for follow-up generation
            context = {
                "original_query": query,
                "answer_explanation": answer.get("explanation", ""),
                "passages_count": len(passages),
                "confidence": answer.get("confidence", "UNKNOWN"),
            }

            # Invoke the follow-up chain
            result = self.follow_up_chain.invoke(context)

            # Extract questions from result
            questions = result.get("questions", [])

            # Validate and truncate questions
            if isinstance(questions, list) and questions:
                return questions[: self.config.max_follow_up_questions]
            else:
                return self._generate_default_follow_ups()

        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
            return self._generate_default_follow_ups()

    @staticmethod
    def _validate_generation_result(result: Dict, query: str) -> Dict:
        """
        Validate and enhance the generation result.

        Args:
            result: Raw result from LLM
            query: Original user query

        Returns:
            Validated and enhanced result
        """
        validated = {"explanation": "", "citations": [], "confidence": "MEDIUM"}

        # Ensure explanation exists
        if "explanation" in result and result["explanation"]:
            validated["explanation"] = result["explanation"]
        else:
            validated["explanation"] = (
                f"I couldn't generate a specific answer for '{query}' based on the provided documents."
            )

        # Ensure citations is a list
        if "citations" in result and isinstance(result["citations"], list):
            validated["citations"] = result["citations"]

        # Validate confidence level
        if "confidence" in result and result["confidence"] in ["HIGH", "MEDIUM", "LOW"]:
            validated["confidence"] = result["confidence"]

        return validated

    def _create_fallback_response(
            self, query: str, passages: List[Dict], error: str
    ) -> Dict:
        """
        Create a fallback response when generation fails.

        Args:
            query: Original user query
            passages: Passages that were attempted to be used
            error: Error message

        Returns:
            Fallback response dictionary
        """
        logger.warning(f"Using fallback response due to error: {error}")

        if passages:
            explanation = (
                f"I found relevant documents but encountered an error processing them for your query: '{query}'."
                f" Please try again or rephrase your question."
            )
        else:
            explanation = (
                f"I couldn't find any relevant documents to answer your question: '{query}'. "
                f"Please try rephrasing or ensure relevant documents are uploaded."
            )

        return {
            "explanation": explanation,
            "citations": [],
            "confidence": "LOW",
            "model_version": self.config.model_name,
            "error": error,
            "passages_used": len(passages),
        }

    @staticmethod
    def _generate_default_follow_ups() -> List[str]:
        """
        Generate default follow-up questions when specialized generation fails.

        Returns:
            List of generic follow-up questions
        """
        return [
            "What are the key terms or concepts mentioned in this regulation?",
            "How does this apply to different types of financial institutions?",
            "Are there any exceptions or special cases for this rule?",
            "What are the consequences of non-compliance with this regulation?",
            "Where can I find more detailed information about this topic?",
        ]

    def batch_generate(
            self, queries: List[str], passages_list: List[List[Dict]]
    ) -> List[Dict]:
        """
        Generate answers for multiple queries in a batch.

        Args:
            queries: List of user queries
            passages_list: List of passage lists for each query

        Returns:
            List of generated answers
        """
        if len(queries) != len(passages_list):
            raise ValueError("Number of queries must match number of passage lists")

        results = []
        for i, (query, passages) in enumerate(zip(queries, passages_list)):
            try:
                result = self.generate_from_passages(query, passages)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch generation failed for query {i}: {e}")
                results.append(self._create_fallback_response(query, passages, str(e)))

        return results


class GroqGenerator(BaseGenerator):
    """Generator component using open-source models via Groq API."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the Groq-based generator.

        Args:
            config: Generator configuration parameters
        """
        super().__init__(config)
        self._initialize_llm()
        self.generator_chain = self._create_generator_chain()
        self.follow_up_chain = self._create_follow_up_chain()
        self.config = config or GeneratorConfig()

        logger.info(f"GroqGenerator initialized with model: {self.config.model_name}")

    def _initialize_llm(self):
        """Initialize the Groq language model."""
        self.llm = ChatGroq(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            max_tokens=self.config.max_tokens,
            api_key=os.getenv("GROQ_API_KEY"),
        )

    def _create_generator_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for answer generation."""
        prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT)
        parser = JsonOutputParser()

        return prompt | self.llm | parser

    def _create_follow_up_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for follow-up question generation."""
        prompt = ChatPromptTemplate.from_template(FOLLOW_UP_PROMPT)
        parser = JsonOutputParser()

        return prompt | self.llm | parser


class OllamaGenerator(BaseGenerator):
    """Generator component using local models via Ollama."""

    def __init__(self, config: Optional[GeneratorConfig] = None, base_url: str = "http://localhost:11434"):
        """
        Initialize the Ollama-based generator for offline use.

        Args:
            config: Generator configuration parameters
            base_url: Ollama server URL
        """
        # Override config with Ollama-specific defaults if not provided
        ollama_config = config or GeneratorConfig(
            model_name="llama3.1:8b",
            max_tokens=1024
        )

        super().__init__(ollama_config)
        self.base_url = base_url
        self._initialize_llm()
        self.generator_chain = self._create_generator_chain()
        self.follow_up_chain = self._create_follow_up_chain()

        logger.info(f"OllamaGenerator initialized with model: {self.config.model_name}")

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
            base_url=self.base_url,
            num_predict=self.config.max_tokens,
        )

    def _create_generator_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for answer generation."""
        prompt = ChatPromptTemplate.from_template(GENERATOR_PROMPT)
        parser = JsonOutputParser()

        return prompt | self.llm | parser

    def _create_follow_up_chain(self) -> RunnableSerializable:
        """Create LangChain runnable for follow-up question generation."""
        prompt = ChatPromptTemplate.from_template(FOLLOW_UP_PROMPT)
        parser = JsonOutputParser()

        return prompt | self.llm | parser


class GeneratorFactory:
    """Factory class for creating generator instances."""

    _registry = {
        "groq": GroqGenerator,
        "ollama": OllamaGenerator,
    }

    @classmethod
    def create_generator(cls, generator_type: str = "groq", **kwargs) -> BaseGenerator:
        """
        Factory method to create generator instances.

        Args:
            generator_type: Type of generator to create ('groq' or 'ollama')
            **kwargs: Additional arguments for generator configuration

        Returns:
            Configured generator instance
        """
        generator_type = generator_type.lower()

        if generator_type not in cls._registry:
            raise ValueError(
                f"Unknown generator type: {generator_type}. "
                f"Available types: {list(cls._registry.keys())}"
            )

        # Get the field names from the dataclass
        config_field_names = GeneratorConfig.get_field_names()

        # Extract config parameters if provided
        config_params = {k: v for k, v in kwargs.items()
                         if k in config_field_names}

        # Create config if any parameters provided
        config = GeneratorConfig(**config_params) if config_params else None

        # Extract non-config parameters
        other_params = {k: v for k, v in kwargs.items()
                        if k not in config_field_names}

        generator_class = cls._registry[generator_type]

        if config and other_params:
            return generator_class(config=config, **other_params)
        elif config:
            return generator_class(config=config)
        elif other_params:
            return generator_class(**other_params)
        else:
            return generator_class()
