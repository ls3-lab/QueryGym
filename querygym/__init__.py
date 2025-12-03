"""
querygym: LLM-based Query Reformulation Toolkit

Simple usage:
    import querygym as qg
    
    # Create a reformulator
    reformulator = qg.create_reformulator("genqr_ensemble")
    
    # Reformulate queries
    result = reformulator.reformulate(qg.QueryItem("q1", "what causes diabetes"))
"""

__version__ = "0.1.0"

# Core data structures
from .core.base import QueryItem, ReformulationResult, MethodConfig, BaseReformulator

# LLM and prompts
from .core.llm import OpenAICompatibleClient
from .core.prompts import PromptBank

# Searcher interface
from .core.searcher import BaseSearcher, SearchHit, SearcherRegistry, create_searcher

# Searcher wrappers for user convenience
from .core.searcher_wrappers import wrap_pyserini_searcher, wrap_pyterrier_retriever, wrap_custom_searcher

# Data loaders
from .data.dataloader import DataLoader, UnifiedQuerySource

# Format-specific loaders
from . import loaders

# All reformulation methods
from .methods import (
    GENQR,
    GenQREnsemble,
    Query2Doc,
    QAExpand,
    MuGI,
    LameR,
    Query2E,
    CSQE,
)

# High-level runner
from .core.runner import run_method, build_llm
from .core.registry import METHODS, register_method


# Import adapters to register them
from . import adapters

# Convenience factory function
def create_reformulator(
    method_name: str,
    model: str = "gpt-4",
    params: dict = None,
    llm_config: dict = None,
    prompt_bank_path: str = None,
    **kwargs
):
    """
    Create a reformulator instance with sensible defaults.
    
    Args:
        method_name: Name of the method (e.g., "genqr", "genqr_ensemble", "query2doc")
        model: LLM model name (default: "gpt-4")
        params: Method-specific parameters (default: {})
        llm_config: Additional LLM configuration (temperature, max_tokens, etc.)
        prompt_bank_path: Path to prompt bank YAML (default: bundled prompt_bank.yaml)
        **kwargs: Additional MethodConfig parameters (seed, retries)
    
    Returns:
        BaseReformulator instance
    
    Example:
        >>> import querygym as qg
        >>> reformulator = qg.create_reformulator("genqr_ensemble", model="gpt-4")
        >>> result = reformulator.reformulate(qg.QueryItem("q1", "what causes diabetes"))
    """
    from pathlib import Path
    
    if params is None:
        params = {}
    
    if llm_config is None:
        llm_config = {}
    
    # Set up LLM config
    llm_cfg = {
        "model": model,
        "temperature": llm_config.get("temperature", 0.8),
        "max_tokens": llm_config.get("max_tokens", 256),
        **{k: v for k, v in llm_config.items() if k not in ["temperature", "max_tokens"]}
    }
    
    # Create config
    config = MethodConfig(
        name=method_name,
        params=params,
        llm=llm_cfg,
        seed=kwargs.get("seed", 42),
        retries=kwargs.get("retries", 2)
    )
    
    # Build LLM client
    llm = build_llm(config)
    
    # Load prompt bank
    if prompt_bank_path is None:
        prompt_bank_path = Path(__file__).parent / "prompt_bank.yaml"
    pb = PromptBank(prompt_bank_path)
    
    # Get method class and instantiate
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(METHODS.keys())}")
    
    MethodClass = METHODS[method_name]
    return MethodClass(config, llm, pb)


def load_queries(path: str, format: str = "tsv", **kwargs):
    """
    Load queries from a local file.
    
    Args:
        path: Path to queries file
        format: File format - "tsv" or "jsonl" (default: "tsv")
        **kwargs: Additional parameters for DataLoader.load_queries()
    
    Returns:
        List of QueryItem objects
    
    Example:
        >>> queries = qg.load_queries("queries.tsv", format="tsv")
        >>> queries = qg.load_queries("queries.jsonl", format="jsonl")
    """
    return DataLoader.load_queries(path, format=format, **kwargs)


def load_qrels(path: str, format: str = "trec", **kwargs):
    """
    Load qrels from a local file.
    
    Args:
        path: Path to qrels file
        format: File format - "trec" (default: "trec")
        **kwargs: Additional parameters for DataLoader.load_qrels()
    
    Returns:
        Dict mapping qid -> {docid -> relevance}
    
    Example:
        >>> qrels = qg.load_qrels("qrels.txt")
    """
    return DataLoader.load_qrels(path, format=format, **kwargs)


def load_contexts(path: str, **kwargs):
    """
    Load contexts from a JSONL file.
    
    Args:
        path: Path to contexts JSONL file
        **kwargs: Additional parameters for DataLoader.load_contexts()
    
    Returns:
        Dict mapping qid -> list of context strings
    
    Example:
        >>> contexts = qg.load_contexts("contexts.jsonl")
    """
    return DataLoader.load_contexts(path, **kwargs)


def load_examples(path: str, **kwargs):
    """
    Load few-shot examples from a JSONL file.
    
    Used for methods like Query2E that need (query, passage) pairs as demonstrations.
    
    Args:
        path: Path to examples JSONL file
        **kwargs: Additional parameters for DataLoader.load_examples()
    
    Returns:
        List of dicts with 'query' and 'passage' keys
    
    Example:
        >>> examples = qg.load_examples("examples.jsonl")
        >>> reformulator = qg.create_reformulator("query2e", params={"mode": "fs", "examples": examples})
        
    JSONL format:
        {"query": "how long is flea life cycle?", "passage": "The life cycle of a flea..."}
        {"query": "cost of flooring?", "passage": "The cost of interior concrete..."}
    """
    return DataLoader.load_examples(path, **kwargs)


__all__ = [
    # Version
    "__version__",
    
    # Core classes
    "QueryItem",
    "ReformulationResult",
    "MethodConfig",
    "BaseReformulator",
    
    # LLM & Prompts
    "OpenAICompatibleClient",
    "PromptBank",
    
    # Searcher interface
    "BaseSearcher",
    "SearchHit",
    "SearcherRegistry",
    "create_searcher",
    
    # Searcher wrappers
    "wrap_pyserini_searcher",
    "wrap_pyterrier_retriever", 
    "wrap_custom_searcher",
    
    # Data
    "DataLoader",
    "UnifiedQuerySource",  # Deprecated
    "loaders",
    
    # Methods
    "GENQR",
    "GenQREnsemble",
    "Query2Doc",
    "QAExpand",
    "MuGI",
    "LameR",
    "Query2E",
    "CSQE",
    
    # High-level API
    "run_method",
    "build_llm",
    "create_reformulator",
    "load_queries",
    "load_qrels",
    "load_contexts",
    "load_examples",
    
    # Retrieval
    "Retriever",
    
    # Registry
    "METHODS",
    "register_method",
]
