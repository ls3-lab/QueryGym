from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List, Tuple
import random
import os
import csv

@register_method("query2doc")
class Query2Doc(BaseReformulator):
    """
    Query2Doc: Generate pseudo-documents from LLM knowledge.
    
    Modes:
        - "zs" (zero-shot): Generate passage directly [default]
        - "cot" (chain-of-thought): Zero-shot with reasoning
        - "fs", "fewshot", or "few-shot" (few-shot): Uses training examples from any dataset (dynamic)
    
    Few-Shot Config (via params or env vars):
        - dataset_type: "msmarco", "beir", or "generic" (uses appropriate loader)
        - num_examples: Number of few-shot examples (default: 4)
        
    For MS MARCO datasets (dataset_type="msmarco"):
        - collection_path / COLLECTION_PATH: Path to collection.tsv
        - train_queries_path / TRAIN_QUERIES_PATH: Path to queries.tsv
        - train_qrels_path / TRAIN_QRELS_PATH: Path to qrels file
    
    For BEIR datasets (dataset_type="beir"):
        - beir_data_dir / BEIR_DATA_DIR: Path to BEIR dataset directory
        - train_split: "train" or "dev" (default: "train")
    
    For generic datasets (dataset_type="generic" or omitted):
        - collection_path: TSV file (docid \t text)
        - train_queries_path: TSV file (qid \t query)
        - train_qrels_path: TREC format (qid 0 docid relevance)
    
    Note: MS MARCO env vars (MSMARCO_COLLECTION, etc.) are supported for backward compatibility.
    """
    VERSION = "1.0"
    CONCATENATION_STRATEGY = "query_repeat_plus_generated"
    DEFAULT_QUERY_REPEATS = 5
    
    def __init__(self, cfg, llm_client, prompt_resolver):
        super().__init__(cfg, llm_client, prompt_resolver)
        self._fewshot_data = None 
    
    def _load_fewshot_data(self):
        """Lazy load training data for few-shot mode (supports MS MARCO, BEIR, or generic datasets)."""
        if self._fewshot_data is not None:
            return self._fewshot_data
        
        try:
            # Get dataset type (msmarco, beir, or generic)
            dataset_type = self.cfg.params.get("dataset_type", "").lower()
            
            # Get paths from config params or environment variables
            collection_path = (
                self.cfg.params.get("collection_path") or 
                self.cfg.params.get("msmarco_collection") or 
                os.getenv("COLLECTION_PATH") or 
                os.getenv("MSMARCO_COLLECTION")
            )
            train_queries_path = (
                self.cfg.params.get("train_queries_path") or 
                self.cfg.params.get("msmarco_train_queries") or 
                os.getenv("TRAIN_QUERIES_PATH") or 
                os.getenv("MSMARCO_TRAIN_QUERIES")
            )
            train_qrels_path = (
                self.cfg.params.get("train_qrels_path") or 
                self.cfg.params.get("msmarco_train_qrels") or 
                os.getenv("TRAIN_QRELS_PATH") or 
                os.getenv("MSMARCO_TRAIN_QRELS")
            )
            
            # For BEIR, collection_path is actually the BEIR data directory
            if dataset_type == "beir":
                beir_data_dir = (
                    self.cfg.params.get("beir_data_dir") or 
                    collection_path or 
                    os.getenv("BEIR_DATA_DIR")
                )
                train_split = self.cfg.params.get("train_split", "train")
                
                if not beir_data_dir:
                    raise RuntimeError(
                        "Few-shot mode with BEIR requires beir_data_dir (via config or env var):\n"
                        "  - beir_data_dir / BEIR_DATA_DIR: Path to BEIR dataset directory"
                    )
            
            elif not all([collection_path, train_queries_path, train_qrels_path]):
                raise RuntimeError(
                    "Few-shot mode requires training data paths (via config params or env vars):\n"
                    "  - dataset_type: 'msmarco', 'beir', or 'generic' (optional)\n"
                    "  - collection_path / COLLECTION_PATH\n"
                    "  - train_queries_path / TRAIN_QUERIES_PATH\n"
                    "  - train_qrels_path / TRAIN_QRELS_PATH\n"
                    "\nFor BEIR datasets:\n"
                    "  - dataset_type: 'beir'\n"
                    "  - beir_data_dir / BEIR_DATA_DIR: Path to BEIR dataset directory\n"
                    "  - train_split: 'train' or 'dev' (default: 'train')\n"
                    "\nFor backward compatibility, MS MARCO env vars are also supported:\n"
                    "  - MSMARCO_COLLECTION, MSMARCO_TRAIN_QUERIES, MSMARCO_TRAIN_QRELS"
                )
            
            # Load data using appropriate loader based on dataset type
            if dataset_type == "beir":
                from ..loaders import beir
                corpus = beir.load_corpus(beir_data_dir)
                # Convert BEIR corpus format (dict with title/text) to simple text
                collection = {}
                for docid, doc_dict in corpus.items():
                    # Combine title and text
                    title = doc_dict.get("title", "").strip()
                    text = doc_dict.get("text", "").strip()
                    collection[docid] = f"{title} {text}".strip() if title else text
                
                train_queries_list = beir.load_queries(beir_data_dir)
                train_queries_dict = {q.qid: q.text for q in train_queries_list}
                train_qrels = beir.load_qrels(beir_data_dir, split=train_split)
                
            elif dataset_type == "msmarco":
                from ..loaders import msmarco
                collection = msmarco.load_collection(collection_path)
                train_queries_list = msmarco.load_queries(train_queries_path)
                train_queries_dict = {q.qid: q.text for q in train_queries_list}
                train_qrels = msmarco.load_qrels(train_qrels_path)
                
            else:
                # Generic: Load using DataLoader for maximum flexibility
                from ..data.dataloader import DataLoader
                
                # Load collection (TSV format: docid \t text)
                collection = {}
                with open(collection_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t')
                    for row in reader:
                        if len(row) >= 2:
                            docid, text = row[0], row[1]
                            collection[docid] = text
                
                # Load queries and qrels
                train_queries_list = DataLoader.load_queries(train_queries_path, format="tsv")
                train_queries_dict = {q.qid: q.text for q in train_queries_list}
                train_qrels = DataLoader.load_qrels(train_qrels_path, format="trec")
            
            self._fewshot_data = {
                "collection": collection,
                "train_queries": train_queries_dict,
                "train_qrels": train_qrels
            }
            
            return self._fewshot_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load few-shot training data: {e}")
    
    def _select_few_shot_examples(self, num_examples: int = 4) -> List[Tuple[str, str]]:
        """Randomly sample relevant query-passage pairs from training data."""
        try:
            data = self._load_fewshot_data()
            collection = data["collection"]
            train_queries = data["train_queries"]
            train_qrels = data["train_qrels"]
            
            # Collect all relevant query-doc pairs
            relevant_pairs = []
            for qid, doc_rels in train_qrels.items():
                for docid, relevance in doc_rels.items():
                    if relevance > 0:  # Only relevant
                        relevant_pairs.append((qid, docid))
            
            if not relevant_pairs:
                raise RuntimeError("No relevant query-document pairs found")
            
            # Sample pairs and fetch texts
            sample_size = min(num_examples * 10, len(relevant_pairs))
            sampled_pairs = random.sample(relevant_pairs, sample_size)
            
            examples = []
            for qid, docid in sampled_pairs:
                if len(examples) >= num_examples:
                    break
                
                query_text = train_queries.get(qid)
                doc_text = collection.get(docid)
                
                if query_text and doc_text:
                    examples.append((query_text, doc_text))
            
            if not examples:
                raise RuntimeError("Could not find valid examples with matching IDs")
            
            if len(examples) < num_examples:
                print(f"Warning: Only found {len(examples)}/{num_examples} examples")
            
            return examples
            
        except Exception as e:
            raise RuntimeError(f"Failed to select few-shot examples: {e}")
    
    def _format_examples(self, examples: List[Tuple[str, str]]) -> str:
        """Format examples for prompt template."""
        examples_text = ""
        for query, passage in examples:
            examples_text += f"Query: {query}\nPassage: {passage}\n\n"
        return examples_text

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        """Generate pseudo-document for query using LLM."""
        # Get parameters
        mode = str(self.cfg.params.get("mode", "zs")).lower()
        # Normalize few-shot mode variants
        if mode in ["fs", "fewshot", "few-shot"]:
            mode = "fs"
        temperature = float(self.cfg.llm.get("temperature", 0.7))
        max_tokens = int(self.cfg.llm.get("max_tokens", 256))
        
        metadata = {"mode": mode}
        
        try:
            # Select prompt based on mode
            if mode == "fs":
                # Few-shot: dynamic MS MARCO examples
                num_examples = int(self.cfg.params.get("num_examples", 4))
                examples = self._select_few_shot_examples(num_examples)
                examples_text = self._format_examples(examples)
                
                msgs = self.prompts.render("query2doc.fewshot.v1", query=q.text, examples=examples_text)
                metadata["prompt_id"] = "query2doc.fewshot.v1"
                metadata["num_examples"] = len(examples)
            else:
                # Zero-shot or CoT
                prompt_id = "query2doc.cot.v1" if mode == "cot" else "query2doc.zeroshot.v1"
                msgs = self.prompts.render(prompt_id, query=q.text)
                metadata["prompt_id"] = prompt_id
            
            # Generate pseudo-document using LLM
            pseudo_doc = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            
            if not pseudo_doc or not pseudo_doc.strip():
                raise RuntimeError("LLM returned empty response")
            
            # Use base class concatenation (query_repeat_plus_generated)
            reformulated = self.concatenate_result(q.text, pseudo_doc)
            
            metadata.update({
                "pseudo_doc": pseudo_doc,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            return ReformulationResult(q.qid, q.text, reformulated, metadata=metadata)
            
        except Exception as e:
            # Graceful error handling - fallback to original query
            error_msg = f"Query2Doc failed: {e}"
            print(f"Error for qid={q.qid}: {error_msg}")
            
            return ReformulationResult(
                q.qid, q.text, q.text,
                metadata={"mode": mode, "error": error_msg, "fallback": True}
            )
