from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from typing import List, Tuple
import random
import os
import csv

@register_method("query2e")
class Query2E(BaseReformulator):
    """
    Query2E: Query to keyword expansion.
    
    Modes:
        - zs (zero-shot): Simple keyword generation
        - fs (few-shot): Uses training examples for keyword generation
    
    Formula: (query × 5) + keywords
    
    Few-Shot Examples (3 ways to provide):
        1. Via --ctx-jsonl CLI flag (JSONL file with {"query": "...", "passage": "..."} per line)
        2. Via params["examples"] (list of {"query": "...", "passage": "..."} dicts)
        3. Auto-generate from training data (if no examples provided)
    
    Few-Shot Auto-Generation Config (via params or env vars):
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
        # User-provided examples (set via set_examples() or params["examples"])
        self._provided_examples = cfg.params.get("examples", None)
    
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
        """Randomly sample relevant query-document pairs from training data."""
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
        """Format examples for prompt template (query -> keywords extraction)."""
        examples_text = ""
        for query, passage in examples:
            # Extract keywords from passage (simple heuristic: split and take meaningful words)
            # In practice, you might want more sophisticated keyword extraction
            words = passage.lower().split()
            # Take first 5-7 meaningful words as keywords (simple heuristic)
            keywords = [w.strip('.,!?;:') for w in words[:7] if len(w) > 3]
            keywords_str = ", ".join(keywords[:5]) if keywords else "keywords, terms, phrases"
            
            examples_text += f"Query: {query}\nKeywords: {keywords_str}\n"
        
        return examples_text

    def _parse_keywords(self, raw_output: str) -> List[str]:
        """
        Parse keywords from LLM output, handling various formats:
        - Comma-separated: "keyword1, keyword2, keyword3"
        - Bullet points: "- keyword1\n- keyword2"
        - Numbered lists: "1. keyword1\n2. keyword2"
        - Mixed formats
        
        Returns:
            List of cleaned keyword strings
        """
        import re
        
        if not raw_output or not raw_output.strip():
            return []
        
        # Normalize the text
        text = raw_output.strip()
        
        # Remove common prefixes like "Keywords:", "Here are the keywords:", etc.
        text = re.sub(r'^(keywords|here are|the keywords|list of keywords)[:\s]*', '', text, flags=re.IGNORECASE)
        
        keywords = []
        
        # Check if it's a bullet/numbered list format
        if '\n' in text or text.strip().startswith('-') or re.match(r'^\d+\.', text.strip()):
            # Split by newlines
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Remove bullet points: -, *, •, ▪, etc.
                line = re.sub(r'^[\-\*•▪→►]\s*', '', line)
                
                # Remove numbered prefixes: 1., 1), (1), etc.
                line = re.sub(r'^[\(\[]?\d+[\)\]\.:\-]?\s*', '', line)
                
                # If line contains commas, it might be multiple keywords
                if ',' in line:
                    for part in line.split(','):
                        part = part.strip()
                        if part:
                            keywords.append(part)
                elif line:
                    keywords.append(line)
        else:
            # Assume comma-separated format
            for part in text.split(','):
                part = part.strip()
                if part:
                    keywords.append(part)
        
        # Clean each keyword
        cleaned_keywords = []
        for kw in keywords:
            # Remove quotes
            kw = kw.strip('"\'`')
            # Remove trailing punctuation
            kw = kw.rstrip('.,;:!?')
            # Remove leading/trailing whitespace
            kw = kw.strip()
            # Skip empty or very short keywords
            if kw and len(kw) > 1:
                cleaned_keywords.append(kw)
        
        return cleaned_keywords

    def set_examples(self, examples: List[Tuple[str, str]]) -> None:
        """
        Set user-provided examples for few-shot mode.
        
        Args:
            examples: List of (query, passage) tuples or list of dicts with 'query' and 'passage' keys
            
        Example:
            >>> reformulator.set_examples([
            ...     ("how long is flea life cycle?", "The life cycle of a flea..."),
            ...     ("cost of flooring?", "The cost of interior concrete..."),
            ... ])
        """
        # Convert dict format to tuple format if needed
        if examples and isinstance(examples[0], dict):
            self._provided_examples = [(ex["query"], ex["passage"]) for ex in examples]
        else:
            self._provided_examples = examples

    def reformulate(self, q: QueryItem, contexts=None) -> ReformulationResult:
        # Get mode parameter
        mode = str(self.cfg.params.get("mode", "zs"))  # Default to zero-shot
        temperature = float(self.cfg.llm.get("temperature", 0.3))
        max_tokens = int(self.cfg.llm.get("max_tokens", 256))
        
        metadata = {"mode": mode}
        
        try:
            # Select prompt based on mode
            if mode in ["fs", "fewshot"]:
                # Few-shot: use provided examples or auto-generate from training data
                num_examples = int(self.cfg.params.get("num_examples", 4))
                
                if self._provided_examples:
                    # Use user-provided examples
                    # Convert dict format to tuple format if needed
                    if isinstance(self._provided_examples[0], dict):
                        examples = [(ex["query"], ex["passage"]) for ex in self._provided_examples]
                    else:
                        examples = self._provided_examples
                    # Limit to num_examples if more provided
                    examples = examples[:num_examples]
                    metadata["examples_source"] = "provided"
                else:
                    # Auto-generate from training data (original behavior)
                    examples = self._select_few_shot_examples(num_examples)
                    metadata["examples_source"] = "auto_generated"
                
                examples_text = self._format_examples(examples)
                
                msgs = self.prompts.render("q2e.fs.v1", query=q.text, examples=examples_text)
                metadata["prompt_id"] = "q2e.fs.v1"
                metadata["num_examples"] = len(examples)
            elif mode in ["zs", "zeroshot"]:
                prompt_id = "q2e.zs.v1"
                msgs = self.prompts.render(prompt_id, query=q.text)
                metadata["prompt_id"] = prompt_id
            else:
                raise ValueError(f"Invalid mode '{mode}' for Query2E. Must be 'zs' (zero-shot) or 'fs' (few-shot).")
            
            # Generate keywords
            out = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)
            
            # Parse keywords using robust parser
            terms = self._parse_keywords(out)
            
            # Limit to max 20 keywords
            max_keywords = int(self.cfg.params.get("max_keywords", 20))
            if len(terms) > max_keywords:
                terms = terms[:max_keywords]
            
            generated_content = " ".join(terms)
            
            # Concatenate: query + keywords
            reformulated = self.concatenate_result(q.text, generated_content)
            
            metadata.update({
                "keywords": terms,
                "temperature": temperature,
                "max_tokens": max_tokens
            })
            
            return ReformulationResult(
                q.qid, 
                q.text, 
                reformulated,
                metadata=metadata
            )
            
        except Exception as e:
            # Graceful error handling - fallback to original query
            error_msg = f"Query2E failed: {e}"
            print(f"Error for qid={q.qid}: {error_msg}")
            
            return ReformulationResult(
                q.qid, q.text, q.text,
                metadata={"mode": mode, "error": error_msg, "fallback": True}
            )
