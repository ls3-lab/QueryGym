#!/usr/bin/env python3
"""
Retrieval Script for QueryGym + Pyserini Pipeline

This script performs document retrieval using Pyserini with BM25,
using reformulated queries and dataset configurations from the registry.

Usage:
    python examples/querygym_pyserini/retrieve.py \
        --dataset msmarco-v1-passage.trecdl2019 \
        --queries outputs/dl19_genqr/queries/reformulated_queries.tsv \
        --output-dir outputs/dl19_genqr
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

import querygym as qg
try:
    from pyserini.search.lucene import LuceneSearcher
except ImportError:
    LuceneSearcher = None
from examples.querygym_pyserini.utils import (
    get_dataset_config,
    setup_logging,
    create_output_dirs,
    save_config,
    format_time,
    print_dataset_info
)


def retrieve_documents(
    dataset_name: str = None,
    queries_file: Path = None,
    output_dir: Path = None,
    k: int = 1000,
    threads: int = 16,
    registry_path: str = "dataset_registry.yaml",
    index_name: str = None
) -> Dict[str, Any]:
    """
    Perform Pyserini retrieval.
    
    Args:
        dataset_name: Name of dataset from registry
        queries_file: Path to TSV file with queries
        output_dir: Output directory
        k: Number of documents to retrieve per query
        threads: Number of threads for parallel retrieval
        registry_path: Path to dataset registry
        
    Returns:
        Dictionary containing retrieval metadata
    """
    logging.info("="*60)
    logging.info("Starting Document Retrieval")
    logging.info("="*60)
    
    start_time = time.time()
    
    # Get index configuration - either from registry or use provided
    if dataset_name:
        # Get dataset configuration from registry
        logging.info(f"Loading dataset: {dataset_name}")
        dataset_config = get_dataset_config(dataset_name, registry_path)
        index_name = dataset_config['index']['name']
        bm25_k1 = dataset_config['bm25_weights']['k1']
        bm25_b = dataset_config['bm25_weights']['b']
    else:
        # Use provided index_name and default BM25 weights
        if not index_name:
            raise ValueError("Either dataset_name or index_name must be provided")
        bm25_k1 = 0.9  # Default BM25 k1
        bm25_b = 0.4   # Default BM25 b
        logging.info(f"Using provided index: {index_name}")
        logging.info(f"Using default BM25 parameters: k1={bm25_k1}, b={bm25_b}")
    
    logging.info(f"Index: {index_name}")
    logging.info(f"BM25 parameters: k1={bm25_k1}, b={bm25_b}")
    
    # Load queries
    logging.info(f"Loading queries from: {queries_file}")
    queries = qg.load_queries(str(queries_file))
    logging.info(f"Loaded {len(queries)} queries")
    
    # Create Pyserini searcher
    logging.info("Initializing Pyserini searcher...")
    
    if LuceneSearcher is None:
        raise ImportError(
            "Pyserini is required for retrieval. Install with: pip install pyserini"
        )
    
    # Initialize searcher with prebuilt index
    logging.info(f"Loading Pyserini index: {index_name}")
    searcher = LuceneSearcher.from_prebuilt_index(index_name)
    searcher.set_bm25(k1=bm25_k1, b=bm25_b)
    logging.info("✓ Pyserini searcher initialized")
    
    # Create output directories
    dirs = create_output_dirs(output_dir)
    
    # Perform batch retrieval
    logging.info(f"Retrieving top {k} documents per query...")
    logging.info(f"Using {threads} threads")
    
    retrieval_start = time.time()
    
    # Prepare queries for batch search
    query_texts = [q.text for q in queries]
    query_ids = [q.qid for q in queries]
    
    # Batch search
    results = searcher.batch_search(
        queries=query_texts,
        qids=query_ids,
        k=k,
        threads=threads
    )
    
    retrieval_time = time.time() - retrieval_start
    logging.info(f"Retrieval complete in {format_time(retrieval_time)}")
    logging.info(f"Average time per query: {retrieval_time/len(queries):.3f}s")
    
    # Save run file in TREC format
    run_file = dirs['runs'] / 'run.txt'
    logging.info(f"Writing run file: {run_file}")
    
    with open(run_file, 'w') as f:
        for qid in query_ids:
            if qid in results:
                hits = results[qid]
                for rank, hit in enumerate(hits, start=1):
                    # TREC format: qid Q0 docid rank score run_name
                    f.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} QueryGym\n")
    
    logging.info(f"✓ Run file saved: {run_file}")
    
    # Calculate statistics
    total_retrieved = sum(len(results.get(qid, [])) for qid in query_ids)
    avg_retrieved = total_retrieved / len(queries) if queries else 0
    
    # Save metadata
    metadata = {
        'dataset': {
            'name': dataset_name,
            'index': index_name,
            'bm25_parameters': {
                'k1': bm25_k1,
                'b': bm25_b
            }
        },
        'retrieval': {
            'num_queries': len(queries),
            'k': k,
            'threads': threads,
            'total_retrieved': total_retrieved,
            'avg_retrieved_per_query': avg_retrieved
        },
        'timing': {
            'total_time_seconds': retrieval_time,
            'avg_time_per_query_seconds': retrieval_time / len(queries),
            'formatted_time': format_time(retrieval_time)
        },
        'outputs': {
            'run_file': str(run_file)
        }
    }
    
    metadata_file = dirs['base'] / 'retrieval_metadata.json'
    save_config(metadata, metadata_file)
    
    # Save retrieval log
    log_file = dirs['runs'] / 'retrieval_log.txt'
    with open(log_file, 'w') as f:
        f.write("Retrieval Log\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Index: {index_name}\n")
        f.write(f"BM25 Parameters: k1={bm25_k1}, b={bm25_b}\n")
        f.write(f"Queries: {len(queries)}\n")
        f.write(f"Top-k: {k}\n")
        f.write(f"Threads: {threads}\n\n")
        f.write(f"Total retrieval time: {format_time(retrieval_time)}\n")
        f.write(f"Average per query: {retrieval_time/len(queries):.3f}s\n")
        f.write(f"Total documents retrieved: {total_retrieved}\n")
        f.write(f"Average per query: {avg_retrieved:.1f}\n")
    
    logging.info(f"Saved retrieval log: {log_file}")
    
    total_time = time.time() - start_time
    logging.info("="*60)
    logging.info(f"Retrieval completed in {format_time(total_time)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("="*60)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve documents using Pyserini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python examples/querygym_pyserini/retrieve.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --queries outputs/dl19_genqr/queries/reformulated_queries.tsv \\
      --output-dir outputs/dl19_genqr

  # Custom top-k and threads
  python examples/querygym_pyserini/retrieve.py \\
      --dataset beir-v1.0.0-nfcorpus \\
      --queries outputs/nfcorpus_genqr/queries/reformulated_queries.tsv \\
      --output-dir outputs/nfcorpus_genqr \\
      --k 100 \\
      --threads 8
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name from dataset_registry.yaml'
    )
    parser.add_argument(
        '--queries',
        type=Path,
        required=True,
        help='Path to queries TSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=1000,
        help='Number of documents to retrieve per query (default: 1000)'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=16,
        help='Number of threads for parallel retrieval (default: 16)'
    )
    parser.add_argument(
        '--registry-path',
        type=str,
        default='dataset_registry.yaml',
        help='Path to dataset registry (default: dataset_registry.yaml)'
    )
    parser.add_argument(
        '--dataset-info',
        type=str,
        help='Show info about a specific dataset and exit'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Handle --dataset-info
    if args.dataset_info:
        print_dataset_info(args.dataset_info, args.registry_path)
        return
    
    # Validate queries file exists
    if not args.queries.exists():
        logging.error(f"Queries file not found: {args.queries}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(
        log_dir=args.output_dir / 'logs',
        log_level=args.log_level,
        log_to_file=True
    )
    
    # Log configuration
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Queries: {args.queries}")
    logging.info(f"Output: {args.output_dir}")
    logging.info(f"Top-k: {args.k}")
    logging.info(f"Threads: {args.threads}")
    
    try:
        # Run retrieval
        metadata = retrieve_documents(
            dataset_name=args.dataset,
            queries_file=args.queries,
            output_dir=args.output_dir,
            k=args.k,
            threads=args.threads,
            registry_path=args.registry_path
        )
        
        logging.info("✓ Retrieval completed successfully!")
        
    except Exception as e:
        logging.error(f"✗ Retrieval failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

