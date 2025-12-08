#!/usr/bin/env python3
"""
Evaluation Script for QueryGym + Pyserini Pipeline

This script evaluates retrieval results using Pyserini's evaluation tools
with qrels and metrics automatically loaded from the dataset registry.

Usage:
    python examples/querygym_pyserini/evaluate.py \
        --dataset msmarco-v1-passage.trecdl2019 \
        --run outputs/dl19_genqr/runs/run.txt \
        --output-dir outputs/dl19_genqr
"""

import argparse
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
import sys
import re

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.querygym_pyserini.utils import (
    get_dataset_config,
    setup_logging,
    create_output_dirs,
    save_config,
    format_time,
    print_dataset_info
)


def parse_trec_eval_output(output: str) -> Dict[str, float]:
    """
    Parse trec_eval output into a dictionary.
    
    Args:
        output: Raw trec_eval output string
        
    Returns:
        Dictionary mapping metric names to values
    """
    metrics = {}
    
    # Parse lines like: "map                   	all	0.4567"
    for line in output.split('\n'):
        if line.strip():
            parts = line.split()
            if len(parts) >= 3:
                metric_name = parts[0]
                # Skip "all" and get the value
                try:
                    value = float(parts[-1])
                    metrics[metric_name] = value
                except ValueError:
                    continue
    
    return metrics


def run_pyserini_eval(
    qrels_name: str = None,
    run_file: Path = None,
    metrics: List[str] = None,
    qrels_file: Path = None
) -> Dict[str, Any]:
    """
    Run evaluation using Pyserini's trec_eval wrapper.
    
    Args:
        qrels_name: Pyserini qrels name (e.g., "dl19-passage") - used if qrels_file not provided
        run_file: Path to run file
        metrics: List of metrics to evaluate
        qrels_file: Path to qrels file (alternative to qrels_name)
        
    Returns:
        Dictionary containing evaluation results
    """
    # Determine qrels source
    if qrels_file:
        qrels_source = str(qrels_file)
        logging.info(f"Using qrels file: {qrels_file}")
    elif qrels_name:
        qrels_source = qrels_name
        logging.info(f"Using Pyserini qrels: {qrels_name}")
    else:
        raise ValueError("Either qrels_name or qrels_file must be provided")
    
    logging.info("Running Pyserini evaluation...")
    logging.info(f"Qrels: {qrels_source}")
    logging.info(f"Run: {run_file}")
    logging.info(f"Metrics: {', '.join(metrics)}")
    
    all_outputs = []
    all_metrics = {}
    
    # Run evaluation for each metric
    for metric in metrics:
        # Build pyserini eval command
        cmd = [
            'python', '-m', 'pyserini.eval.trec_eval',
            '-c',  # Per-query output
        ]
        
        # Add relevance level for specific metrics
        if 'map' in metric.lower() or 'recall' in metric.lower():
            cmd.extend(['-l', '2'])  # Relevance level >= 2
        
        # Add metric flag
        cmd.extend(['-m', metric])
        
        # Add qrels source (name or file path) and run file
        cmd.extend([qrels_source, str(run_file)])
        
        logging.debug(f"Command: {' '.join(cmd)}")
        
        try:
            # Run pyserini eval
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            output = result.stdout
            all_outputs.append(f"=== {metric} ===\n{output}\n")
            
            # Parse metric value from output
            parsed = parse_trec_eval_output(output)
            all_metrics.update(parsed)
            
            logging.debug(f"✓ {metric} computed")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Pyserini eval failed for {metric} with exit code {e.returncode}")
            logging.error(f"stderr: {e.stderr}")
            raise
        except FileNotFoundError:
            raise FileNotFoundError(
                "Pyserini not found. Please install it:\n"
                "  pip install pyserini"
            )
    
    logging.info("✓ Pyserini evaluation completed successfully")
    
    return {
        'raw_output': '\n'.join(all_outputs),
        'metrics': all_metrics
    }


def evaluate_run(
    dataset_name: str = None,
    run_file: Path = None,
    output_dir: Path = None,
    registry_path: str = "dataset_registry.yaml",
    qrels_file: Path = None,
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate retrieval results.
    
    Args:
        dataset_name: Name of dataset from registry
        run_file: Path to TREC run file
        output_dir: Output directory
        registry_path: Path to dataset registry
        
    Returns:
        Dictionary containing evaluation metadata and results
    """
    logging.info("="*60)
    logging.info("Starting Evaluation")
    logging.info("="*60)
    
    start_time = time.time()
    
    # Get configuration - either from registry or use provided files
    if dataset_name:
        # Get dataset configuration from registry
        logging.info(f"Loading dataset: {dataset_name}")
        dataset_config = get_dataset_config(dataset_name, registry_path)
        qrels_name = dataset_config['qrels']['name']
        eval_metrics = dataset_config['output']['eval_metrics']
        qrels_file = None
    else:
        # Use provided qrels file
        if not qrels_file:
            raise ValueError("Either dataset_name or qrels_file must be provided")
        if not qrels_file.exists():
            raise FileNotFoundError(f"Qrels file not found: {qrels_file}")
        qrels_name = None
        eval_metrics = metrics if metrics else ['map', 'ndcg_cut.10', 'recall.1000']  # Default metrics
        logging.info(f"Using qrels file: {qrels_file}")
        logging.info(f"Using default metrics: {', '.join(eval_metrics)}")
    
    logging.info(f"Metrics: {', '.join(eval_metrics)}")
    
    # Create output directories
    dirs = create_output_dirs(output_dir)
    
    # Run Pyserini evaluation
    eval_start = time.time()
    eval_results = run_pyserini_eval(
        qrels_name=qrels_name,
        qrels_file=qrels_file,
        run_file=run_file,
        metrics=eval_metrics
    )
    eval_time = time.time() - eval_start
    
    logging.info(f"Evaluation complete in {format_time(eval_time)}")
    
    # Save full trec_eval output
    output_file = dirs['eval'] / 'eval_results.txt'
    with open(output_file, 'w') as f:
        f.write(eval_results['raw_output'])
    
    logging.info(f"✓ Full eval output saved: {output_file}")
    
    # Save parsed metrics as JSON
    metrics_file = dirs['eval'] / 'eval_results.json'
    save_config(eval_results['metrics'], metrics_file)
    
    # Create human-readable summary
    summary_file = dirs['eval'] / 'eval_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Evaluation Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Qrels: {qrels_name}\n")
        f.write(f"Run: {run_file.name}\n\n")
        
        # Print all computed metrics
        f.write("\n")
        f.write("All Computed Metrics:\n")
        f.write("-"*80 + "\n")
        for metric_name, value in sorted(eval_results['metrics'].items()):
            f.write(f"  {metric_name:25s}: {value:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    logging.info(f"✓ Summary saved: {summary_file}")
    
    # Print all metrics to console
    logging.info("\nAll Computed Metrics:")
    logging.info("-"*40)
    for metric_name, value in sorted(eval_results['metrics'].items()):
        logging.info(f"  {metric_name:20s}: {value:.4f}")
    logging.info("-"*40)
    
    # Prepare metadata
    metadata = {
        'dataset': {
            'name': dataset_name,
            'qrels': qrels_name
        },
        'evaluation': {
            'method': 'pyserini.eval.trec_eval',
            'metrics_requested': metrics,
            'metrics_computed': list(eval_results['metrics'].keys())
        },
        'results': eval_results['metrics'],
        'timing': {
            'eval_time_seconds': eval_time,
            'formatted_time': format_time(eval_time)
        },
        'outputs': {
            'full_output': str(output_file),
            'metrics_json': str(metrics_file),
            'summary': str(summary_file)
        }
    }
    
    metadata_file = dirs['base'] / 'evaluation_metadata.json'
    save_config(metadata, metadata_file)
    
    total_time = time.time() - start_time
    logging.info("="*60)
    logging.info(f"Evaluation completed in {format_time(total_time)}")
    logging.info(f"Output directory: {output_dir}")
    logging.info("="*60)
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results using Pyserini's evaluation tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python examples/querygym_pyserini/evaluate.py \\
      --dataset msmarco-v1-passage.trecdl2019 \\
      --run outputs/dl19_genqr/runs/run.txt \\
      --output-dir outputs/dl19_genqr

  # BEIR dataset
  python examples/querygym_pyserini/evaluate.py \\
      --dataset beir-v1.0.0-nfcorpus \\
      --run outputs/nfcorpus_genqr/runs/run.txt \\
      --output-dir outputs/nfcorpus_genqr
        """
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name from dataset_registry.yaml'
    )
    parser.add_argument(
        '--run',
        type=Path,
        required=True,
        help='Path to TREC run file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Output directory'
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
    
    # Validate run file exists
    if not args.run.exists():
        logging.error(f"Run file not found: {args.run}")
        sys.exit(1)
    
    # Setup logging
    setup_logging(
        log_dir=args.output_dir / 'logs',
        log_level=args.log_level,
        log_to_file=True
    )
    
    # Log configuration
    logging.info(f"Dataset: {args.dataset}")
    logging.info(f"Run: {args.run}")
    logging.info(f"Output: {args.output_dir}")
    
    try:
        # Run evaluation
        metadata = evaluate_run(
            dataset_name=args.dataset,
            run_file=args.run,
            output_dir=args.output_dir,
            registry_path=args.registry_path
        )
        
        logging.info("✓ Evaluation completed successfully!")
        
    except Exception as e:
        logging.error(f"✗ Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()



