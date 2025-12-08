#!/usr/bin/env python3
"""
Shared utilities for QueryGym + Pyserini pipeline.

This module provides common functions for:
- Loading dataset registry
- Loading Pyserini topics and qrels
- Setting up logging
- Creating output directories
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

try:
    from pyserini.search import get_topics, get_qrels
except ImportError:
    get_topics = None
    get_qrels = None


def load_dataset_registry(registry_path: str = "dataset_registry.yaml") -> Dict[str, Any]:
    """
    Load the dataset registry YAML file.
    
    Args:
        registry_path: Path to dataset_registry.yaml (default: project root)
        
    Returns:
        Dictionary containing the registry configuration
        
    Example:
        >>> registry = load_dataset_registry()
        >>> datasets = registry['datasets']
    """
    registry_file = Path(registry_path)
    
    if not registry_file.exists():
        raise FileNotFoundError(
            f"Dataset registry not found: {registry_path}\n"
            f"Expected location: {registry_file.absolute()}"
        )
    
    with open(registry_file, 'r') as f:
        registry = yaml.safe_load(f)
    
    return registry


def get_dataset_config(dataset_name: str, registry_path: str = "dataset_registry.yaml") -> Dict[str, Any]:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., "msmarco-v1-passage.trecdl2019")
        registry_path: Path to dataset_registry.yaml
        
    Returns:
        Dictionary containing dataset configuration
        
    Raises:
        ValueError: If dataset not found in registry
        
    Example:
        >>> config = get_dataset_config("msmarco-v1-passage.trecdl2019")
        >>> print(config['index']['name'])  # "msmarco-v1-passage"
    """
    registry = load_dataset_registry(registry_path)
    datasets = registry.get('datasets', {})
    
    if dataset_name not in datasets:
        available = list(datasets.keys())
        raise ValueError(
            f"Dataset '{dataset_name}' not found in registry.\n"
            f"Available datasets: {available[:5]}... ({len(available)} total)"
        )
    
    return datasets[dataset_name]


def load_pyserini_topics(topic_name: str) -> Dict[str, Dict[str, str]]:
    """
    Load topics using Pyserini's get_topics function.
    
    Args:
        topic_name: Pyserini topic name (e.g., "dl19-passage", "beir-v1.0.0-nfcorpus-test")
        
    Returns:
        Dictionary mapping topic IDs to topic dicts with 'title' key
        
    Example:
        >>> topics = load_pyserini_topics("dl19-passage")
        >>> print(topics[264014]['title'])  # "how long is life cycle of flea"
    """
    if get_topics is None:
        raise ImportError(
            "Pyserini is required for loading topics. Install with: pip install pyserini"
        )
    
    logging.info(f"Loading Pyserini topics: {topic_name}")
    topics = get_topics(topic_name)
    logging.info(f"Loaded {len(topics)} topics")
    
    return topics


def load_pyserini_qrels(qrels_name: str) -> Dict[str, Dict[str, int]]:
    """
    Load qrels using Pyserini's get_qrels function.
    
    Args:
        qrels_name: Pyserini qrels name (e.g., "dl19-passage")
        
    Returns:
        Dictionary mapping qid -> {docid -> relevance}
        
    Example:
        >>> qrels = load_pyserini_qrels("dl19-passage")
        >>> print(qrels['264014']['7067032'])  # relevance score
    """
    if get_qrels is None:
        raise ImportError(
            "Pyserini is required for loading qrels. Install with: pip install pyserini"
        )
    
    logging.info(f"Loading Pyserini qrels: {qrels_name}")
    qrels = get_qrels(qrels_name)
    
    # Count total relevance judgments
    total_judgments = sum(len(docs) for docs in qrels.values())
    logging.info(f"Loaded qrels for {len(qrels)} queries ({total_judgments} judgments)")
    
    return qrels


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO",
    log_to_file: bool = True
) -> None:
    """
    Configure logging for the pipeline.
    
    Args:
        log_dir: Directory for log files (if None, only logs to console)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to write logs to file
        
    Example:
        >>> setup_logging(log_dir=Path("outputs/logs"))
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_to_file and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {log_file}")


def create_output_dirs(output_base: Path) -> Dict[str, Path]:
    """
    Create standard output directory structure.
    
    Args:
        output_base: Base output directory
        
    Returns:
        Dictionary mapping directory names to Path objects
        
    Example:
        >>> dirs = create_output_dirs(Path("outputs/dl19_genqr"))
        >>> print(dirs['logs'])  # Path("outputs/dl19_genqr/logs")
    """
    output_base = Path(output_base)
    
    dirs = {
        'base': output_base,
        'logs': output_base / 'logs',
        'queries': output_base / 'queries',
        'runs': output_base / 'runs',
        'eval': output_base / 'eval',
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Created output directories in: {output_base}")
    
    return dirs


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save JSON file
        
    Example:
        >>> save_config({"method": "genqr", "model": "qwen2.5:7b"}, Path("config.json"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    logging.info(f"Saved configuration to: {output_path}")


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
        
    Example:
        >>> format_time(125.5)
        "2m 5.5s"
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def list_available_datasets(registry_path: str = "dataset_registry.yaml") -> List[str]:
    """
    List all available datasets in the registry.
    
    Args:
        registry_path: Path to dataset_registry.yaml
        
    Returns:
        List of dataset names
    """
    registry = load_dataset_registry(registry_path)
    datasets = registry.get('datasets', {})
    return sorted(datasets.keys())


def load_reformulation_config(config_path: str) -> Dict[str, Any]:
    """
    Load reformulation configuration from YAML file.
    
    Args:
        config_path: Path to reformulation config YAML file (can be relative or absolute)
        
    Returns:
        Dictionary containing the configuration
        
    Example:
        >>> config = load_reformulation_config("reformulation_config.yaml")
        >>> method_config = config['methods']['genqr']
    """
    config_file = Path(config_path)
    
    # If path is relative and doesn't exist, try relative to script directory
    if not config_file.is_absolute() and not config_file.exists():
        # Try relative to the utils.py location (examples/querygym_pyserini/)
        script_dir = Path(__file__).parent
        alt_path = script_dir / config_path
        if alt_path.exists():
            config_file = alt_path
        # Also try relative to project root
        elif (Path.cwd() / config_path).exists():
            config_file = Path.cwd() / config_path
    
    if not config_file.exists():
        # Try one more time with common locations
        script_dir = Path(__file__).parent
        common_paths = [
            script_dir / config_path,  # Same directory as script
            Path.cwd() / config_path,  # Current working directory
            Path.cwd() / "examples" / "querygym_pyserini" / config_path,  # Relative to project root
        ]
        for path in common_paths:
            if path.exists():
                config_file = path
                break
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Reformulation config not found: {config_path}\n"
            f"Tried locations:\n"
            f"  - {Path(config_path).absolute()}\n"
            f"  - {Path(__file__).parent / config_path}\n"
            f"  - {Path.cwd() / config_path}\n"
            f"  - {Path.cwd() / 'examples' / 'querygym_pyserini' / config_path}"
        )
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_method_config_from_yaml(
    config_path: str,
    method: str,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get configuration for a specific method from YAML config file.
    Merges global defaults, method-specific config, and CLI overrides.
    
    Args:
        config_path: Path to reformulation config YAML file
        method: Method name (e.g., "genqr", "query2doc")
        cli_overrides: Dictionary of CLI arguments that override config values
        
    Returns:
        Dictionary with keys: 'model', 'llm_config', 'method_params'
        
    Example:
        >>> config = get_method_config_from_yaml(
        ...     "reformulation_config.yaml",
        ...     "genqr",
        ...     cli_overrides={"model": "custom-model"}
        ... )
    """
    if cli_overrides is None:
        cli_overrides = {}
    
    # Load config file
    full_config = load_reformulation_config(config_path)
    
    # Start with global defaults
    global_config = full_config.get('global', {})
    global_llm = global_config.get('llm', {})
    global_retrieval = global_config.get('retrieval', {})
    
    # Get method-specific config
    methods_config = full_config.get('methods', {})
    method_config = methods_config.get(method, {})
    
    # Build final config, merging in order: global -> method -> CLI
    final_config = {
        'model': cli_overrides.get('model') or method_config.get('model') or global_llm.get('model'),
        'llm_config': {},
        'method_params': {}
    }
    
    # Merge LLM config
    llm_config = global_llm.copy()
    llm_config.update(method_config.get('llm', {}))
    
    # Apply CLI overrides to LLM config
    if 'base_url' in cli_overrides:
        llm_config['base_url'] = cli_overrides['base_url']
    if 'api_key' in cli_overrides:
        llm_config['api_key'] = cli_overrides['api_key']
    if 'temperature' in cli_overrides:
        llm_config['temperature'] = cli_overrides['temperature']
    if 'max_tokens' in cli_overrides:
        llm_config['max_tokens'] = cli_overrides['max_tokens']
    
    final_config['llm_config'] = llm_config
    
    # Merge method params
    method_params = method_config.get('params', {}).copy()
    
    # Add retrieval params from global config if not in method params
    if 'retrieval_k' not in method_params and 'retrieval_k' in global_retrieval:
        method_params['retrieval_k'] = global_retrieval['retrieval_k']
    
    # Apply CLI overrides to method params
    if 'retrieval_k' in cli_overrides:
        method_params['retrieval_k'] = cli_overrides['retrieval_k']
    
    final_config['method_params'] = method_params
    
    return final_config


def print_dataset_info(dataset_name: str, registry_path: str = "dataset_registry.yaml") -> None:
    """
    Print information about a dataset.
    
    Args:
        dataset_name: Name of the dataset
        registry_path: Path to dataset registry
    """
    config = get_dataset_config(dataset_name, registry_path)
    
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Name: {config.get('name', 'N/A')}")
    print(f"Index: {config['index']['name']}")
    print(f"Topics: {config['topics']['name']}")
    print(f"Qrels: {config['qrels']['name']}")
    print(f"BM25 Parameters:")
    print(f"  k1: {config['bm25_weights']['k1']}")
    print(f"  b: {config['bm25_weights']['b']}")
    print(f"Eval Metrics: {', '.join(config['output']['eval_metrics'])}")
    print(f"{'='*60}\n")

