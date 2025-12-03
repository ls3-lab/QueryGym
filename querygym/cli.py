from __future__ import annotations
import typer, csv, json
from pathlib import Path
from typing import Optional

from .core.base import MethodConfig
from .core.runner import run_method
from .core.prompts import PromptBank
from .data.dataloader import UnifiedQuerySource
app = typer.Typer(help="querygym Toolkit CLI")

def build_script_lines(index_path:str, topics:str, run:str, qrels:Optional[str]=None,
                       bm25:bool=True, extra:str="") -> list[str]:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Pyserini search",
        "python -m pyserini.search.lucene \\",
        f"  --index {index_path} \\",
        f"  --topics {topics} \\",
        f"  --output {run} \\",
        "  --bm25" if bm25 else "  --qld",
    ]
    if extra:
        lines.append(f"  {extra}")
    lines.append("")
    if qrels:
        lines += [
            "# trec_eval",
            f"trec_eval -m map -m P.10 -m ndcg_cut.10 {qrels} {run} | tee {run}.eval.txt"
        ]
    return lines

@app.command()
def run(method: str = typer.Option(...),
        queries_tsv: Path = typer.Option(..., "--queries-tsv", exists=True),
        output_tsv: Path = typer.Option(..., "--output-tsv"),
        cfg_path: Optional[Path] = typer.Option(None, "--cfg-path"),
        prompt_bank: Path = typer.Option(Path(__file__).with_name("prompt_bank.yaml"), "--prompt-bank"),
        ctx_jsonl: Optional[Path] = typer.Option(None, "--ctx-jsonl", help="Optional contexts JSONL"),
        output_format: str = typer.Option("both", "--output-format", help="Output format: 'concat', 'plain', or 'both' (default: both)"),
        parallel: Optional[bool] = typer.Option(None, "--parallel", help="Enable parallel generation for methods like MuGI"),
        mode: Optional[str] = typer.Option(None, "--mode", help="Mode for methods: 'zs' (zero-shot) or 'fs' (few-shot)"),
        # Few-shot data loading options
        dataset_type: Optional[str] = typer.Option(None, "--dataset-type", help="Dataset type: 'msmarco', 'beir', or 'generic'"),
        collection_path: Optional[Path] = typer.Option(None, "--collection-path", help="Path to collection file (TSV for MS MARCO/generic)"),
        train_queries_path: Optional[Path] = typer.Option(None, "--train-queries-path", help="Path to training queries file"),
        train_qrels_path: Optional[Path] = typer.Option(None, "--train-qrels-path", help="Path to training qrels file"),
        beir_data_dir: Optional[Path] = typer.Option(None, "--beir-data-dir", help="Path to BEIR dataset directory"),
        train_split: Optional[str] = typer.Option(None, "--train-split", help="BEIR train split: 'train' or 'dev'"),
        num_examples: Optional[int] = typer.Option(None, "--num-examples", help="Number of few-shot examples"),
):
    import yaml
    import os
    import re
    
    def expand_env_vars(text):
        """Expand environment variables in YAML content"""
        def replace_env_var(match):
            var_expr = match.group(1)
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                return os.getenv(var_name, default_value)
            else:
                return os.getenv(var_expr, '')
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, text)
    
    # Default to defaults.yaml if no config path provided
    if cfg_path is None:
        cfg_path = Path(__file__).parent / "config" / "defaults.yaml"
    
    yaml_content = cfg_path.read_text()
    expanded_content = expand_env_vars(yaml_content)
    cfg = yaml.safe_load(expanded_content)
    
    # Override parameters if provided via CLI
    params = cfg.get("params", {})
    if parallel is not None:
        params["parallel"] = parallel
    if mode is not None:
        params["mode"] = mode
    if dataset_type is not None:
        params["dataset_type"] = dataset_type
    if collection_path is not None:
        params["collection_path"] = str(collection_path)
    if train_queries_path is not None:
        params["train_queries_path"] = str(train_queries_path)
    if train_qrels_path is not None:
        params["train_qrels_path"] = str(train_qrels_path)
    if beir_data_dir is not None:
        params["beir_data_dir"] = str(beir_data_dir)
    if train_split is not None:
        params["train_split"] = train_split
    if num_examples is not None:
        params["num_examples"] = num_examples
    
    # Handle context/examples loading based on method
    ctx_map = None
    if ctx_jsonl:
        if method == "query2e":
            # For Q2E: load as examples (query-passage pairs)
            # Auto-set mode to "fs" if not explicitly set to "zs"
            if mode not in ["zs", "zeroshot"]:
                from .data.dataloader import DataLoader
                examples = DataLoader.load_examples(ctx_jsonl)
                params["examples"] = examples
                # Auto-set mode to fs if not already set
                if mode is None:
                    params["mode"] = "fs"
                    typer.echo(f"Auto-setting mode to 'fs' (few-shot) since examples provided")
                typer.echo(f"Loaded {len(examples)} few-shot examples from {ctx_jsonl}")
            else:
                typer.echo(f"Warning: --ctx-jsonl ignored for Q2E in zero-shot mode")
        else:
            # For other methods (LameR, CSQE): load as per-query contexts
            from .data.dataloader import UnifiedContextSource
            ctx_src = UnifiedContextSource(mode="file", path=ctx_jsonl)
            ctx_map = ctx_src.load(list(UnifiedQuerySource(backend="local", format="tsv", path=queries_tsv).iter()))
    
    mc = MethodConfig(name=method, params=params, llm=cfg["llm"],
                      seed=cfg.get("seed",42), retries=cfg.get("retries",2))
    src = UnifiedQuerySource(backend="local", format="tsv", path=queries_tsv)
    queries = list(src.iter())
    
    # Pass ctx_map as-is (None if not provided, dict if provided)
    results = run_method(method_name=method, cfg=mc, queries=queries,
                         prompt_bank_path=str(prompt_bank), ctx_map=ctx_map)
    
    # Show progress summary
    typer.echo(f"Processed {len(results)} queries with {method}")
    
    def write_concat_format(output_path):
        """Write concatenated format to file"""
        with open(output_path, "w", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar='\\')
            for r in results:
                w.writerow([r.qid, r.reformulated])
    
    def write_plain_format(output_path):
        """Write plain format to file"""
        with open(output_path, "w", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar='\\')
            
            # Write header row based on method
            if method == "lamer":
                # LameR: qid \t passage_1 \t passage_2 \t ... \t passage_n
                # Determine number of passages from first result
                if results:
                    passages = results[0].metadata.get("generated_passages", [])
                    if isinstance(passages, list) and passages:
                        num_passages = len(passages)
                    else:
                        # Fallback: estimate from reformulated query
                        original = results[0].original
                        reformulated = results[0].reformulated
                        parts = reformulated.replace(original, "").strip().split()
                        num_passages = len(parts)
                    
                    header = ["qid"] + [f"passage_{i+1}" for i in range(num_passages)]
                    w.writerow(header)
                else:
                    w.writerow(["qid", "passage_1", "passage_2", "passage_3", "passage_4", "passage_5"])
            elif method == "query2doc":
                # Query2Doc: qid \t passage
                w.writerow(["qid", "passage"])
            elif method == "genqr":
                # GenQR: qid \t reformulation_1 \t reformulation_2 \t ... \t reformulation_n
                # Determine number of reformulations from first result
                if results:
                    reformulations = results[0].metadata.get("reformulations", [])
                    if isinstance(reformulations, list) and reformulations:
                        num_reformulations = len(reformulations)
                    else:
                        num_reformulations = 5  # Default estimate
                    header = ["qid"] + [f"reformulation_{i+1}" for i in range(num_reformulations)]
                    w.writerow(header)
                else:
                    w.writerow(["qid", "reformulation_1", "reformulation_2", "reformulation_3", "reformulation_4", "reformulation_5"])
            elif method == "qa_expand":
                # QA-Expand: qid \t final_refined_query
                w.writerow(["qid", "refined_query"])
            elif method == "mugi":
                # MuGI: qid \t pseudo_document
                w.writerow(["qid", "pseudo_document"])
            elif method == "genqr_ensemble":
                # GenQREnsemble: qid \t keyword_1 \t keyword_2 \t ... \t keyword_n
                # Determine number of keywords from first result
                if results:
                    keywords = results[0].metadata.get("keywords", [])
                    if isinstance(keywords, list) and keywords:
                        num_keywords = len(keywords)
                    else:
                        num_keywords = 15  # Default estimate for ensemble
                    header = ["qid"] + [f"keyword_{i+1}" for i in range(num_keywords)]
                    w.writerow(header)
                else:
                    w.writerow(["qid", "keyword_1", "keyword_2", "keyword_3", "keyword_4", "keyword_5", "keyword_6", "keyword_7", "keyword_8", "keyword_9", "keyword_10"])
            elif method == "query2e":
                # Query2E: qid \t keyword_1 \t keyword_2 \t ... \t keyword_n
                # Determine number of keywords from first result
                if results:
                    keywords = results[0].metadata.get("keywords", [])
                    if isinstance(keywords, list) and keywords:
                        num_keywords = len(keywords)
                    else:
                        num_keywords = 10  # Default estimate
                    header = ["qid"] + [f"entity_{i+1}" for i in range(num_keywords)]
                    w.writerow(header)
                else:
                    w.writerow(["qid", "entity_1", "entity_2", "entity_3", "entity_4", "entity_5"])
            else:
                # Default fallback
                w.writerow(["qid", "generated_content"])
            
            # Write data rows
            for r in results:
                if method == "lamer":
                    # LameR: qid \t passage_1 \t passage_2 \t ... \t passage_n
                    passages = r.metadata.get("generated_passages", [])
                    if isinstance(passages, list) and passages:
                        # Clean each passage
                        cleaned_passages = [clean_text(p) for p in passages]
                        w.writerow([r.qid] + cleaned_passages)
                    else:
                        # Fallback: split reformulated query by original query
                        original = r.original
                        reformulated = r.reformulated
                        # Remove original query repetitions and split
                        parts = reformulated.replace(original, "").strip().split()
                        cleaned_parts = [clean_text(p) for p in parts]
                        w.writerow([r.qid] + cleaned_parts)
                elif method == "query2doc":
                    # Query2Doc: qid \t passage
                    passage = r.metadata.get("pseudo_doc", "")
                    cleaned_passage = clean_text(passage)
                    w.writerow([r.qid, cleaned_passage])
                elif method == "genqr":
                    # GenQR: qid \t reformulations
                    reformulations = r.metadata.get("reformulations", [])
                    if isinstance(reformulations, list):
                        cleaned_reformulations = [clean_text(k) for k in reformulations]
                        w.writerow([r.qid] + cleaned_reformulations)
                    else:
                        cleaned_reformulations = clean_text(reformulations)
                        w.writerow([r.qid, cleaned_reformulations])
                elif method == "qa_expand":
                    # QA-Expand: qid \t final_refined_query
                    final_q = r.metadata.get("final_q", "")
                    cleaned_final_q = clean_text(final_q)
                    w.writerow([r.qid, cleaned_final_q])
                elif method == "mugi":
                    # MuGI: qid \t pseudo_document (all 5 joined)
                    pseudo_docs = []
                    for i in range(1, 6):
                        doc = r.metadata.get(f"pseudo_doc_{i}", "")
                        if doc:
                            pseudo_docs.append(doc)
                    # Join all pseudo-docs into single text
                    all_pseudo = " ".join(pseudo_docs)
                    cleaned_pseudo = clean_text(all_pseudo)
                    w.writerow([r.qid, cleaned_pseudo])
                elif method == "genqr_ensemble":
                    # GenQREnsemble: qid \t keyword_1 \t keyword_2 \t ... \t keyword_n
                    keywords = r.metadata.get("keywords", [])
                    if isinstance(keywords, list):
                        cleaned_keywords = [clean_text(k) for k in keywords]
                        w.writerow([r.qid] + cleaned_keywords)
                    else:
                        cleaned_keywords = clean_text(keywords)
                        w.writerow([r.qid, cleaned_keywords])
                elif method == "query2e":
                    # Query2E: qid \t keyword_1 \t keyword_2 \t ... \t keyword_n
                    keywords = r.metadata.get("keywords", [])
                    if isinstance(keywords, list):
                        cleaned_keywords = [clean_text(k) for k in keywords]
                        w.writerow([r.qid] + cleaned_keywords)
                    else:
                        cleaned_keywords = clean_text(keywords)
                        w.writerow([r.qid, cleaned_keywords])
                else:
                    # Default fallback: use reformulated without original query
                    plain_output = r.reformulated.replace(r.original, "").strip()
                    cleaned_output = clean_text(plain_output)
                    w.writerow([r.qid, cleaned_output])
    
    def clean_text(text):
        """Clean text by removing newlines, extra whitespace, and other formatting issues"""
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove newlines and carriage returns
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove backslashes that might be escape characters
        text = text.replace('\\', ' ')
        
        # Replace multiple spaces with single space
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Remove quotes from beginning and end
        text = text.strip('"').strip("'")
        
        return text
    
    # Generate output files based on format
    if output_format == "concat":
        write_concat_format(output_tsv)
        typer.echo(f"Wrote {output_tsv} in concat format")
    elif output_format == "plain":
        write_plain_format(output_tsv)
        typer.echo(f"Wrote {output_tsv} in plain format")
    elif output_format == "both":
        # Generate both formats
        output_dir = output_tsv.parent
        output_stem = output_tsv.stem
        output_suffix = output_tsv.suffix
        
        concat_file = output_dir / f"{output_stem}_concat{output_suffix}"
        plain_file = output_dir / f"{output_stem}_plain{output_suffix}"
        
        write_concat_format(concat_file)
        write_plain_format(plain_file)
        
        typer.echo(f"Wrote both formats:")
        typer.echo(f"  Concat: {concat_file}")
        typer.echo(f"  Plain: {plain_file}")
    else:
        raise ValueError(f"Invalid output_format: {output_format}. Must be 'concat', 'plain', or 'both'")

@app.command("data-to-tsv")
def data_to_tsv(backend: str = typer.Option(...),
                source: str = typer.Option(...),
                out: Path = typer.Option(..., "--out"),
                path: Optional[Path] = typer.Option(None, help="Local TSV/JSONL path"),
                format: Optional[str] = typer.Option(None, help="tsv|jsonl"),
                tsv_qid_col: int = 0, tsv_query_col: int = 1,
                jsonl_qid_key: str = "qid", jsonl_query_key: str = "query",
                msmarco_queries_tsv: Optional[Path] = None,
                hf_name: Optional[str] = None, hf_config: Optional[str] = None,
                split: str = "dev", hf_qid_key: str = "query_id", hf_query_key: str = "query",
                beir_root: Optional[Path] = None, beir_name: Optional[str] = None,
):
    src = UnifiedQuerySource(
        backend=backend if backend in ("local","msmarco","beir") else None,
        source=source if source in ("file","hf","beir") else None,
        path=path, format=format, tsv_qid_col=tsv_qid_col, tsv_query_col=tsv_query_col,
        jsonl_qid_key=jsonl_qid_key, jsonl_query_key=jsonl_query_key,
        msmarco_queries_tsv=msmarco_queries_tsv,
        hf_name=hf_name, hf_config=hf_config, split=split,
        hf_qid_key=hf_qid_key, hf_query_key=hf_query_key,
        beir_root=beir_root, beir_name=beir_name,
    )
    UnifiedQuerySource.export_to_tsv(src.iter(), out)
    typer.echo(f"Wrote {out}")

@app.command("prompts-list")
def prompts_list(prompt_bank: Path = typer.Option(Path(__file__).with_name("prompt_bank.yaml"))):
    pb = PromptBank(prompt_bank)
    for pid in pb.list():
        typer.echo(pid)

@app.command("prompts-show")
def prompts_show(prompt_id: str,
                 prompt_bank: Path = typer.Option(Path(__file__).with_name("prompt_bank.yaml"))):
    pb = PromptBank(prompt_bank)
    meta = pb.get_meta(prompt_id)
    typer.echo(json.dumps(meta, indent=2))

@app.command("script-gen")
def script_gen(index_path: Path = typer.Option(..., exists=False),
               topics: Path = typer.Option(..., help="TSV queries"),
               run: Path = typer.Option(..., help="Output run file"),
               output_bash: Path = typer.Option(Path("run_retrieval.sh")),
               qrels: Optional[Path] = None,
               bm25: bool = True,
               extra: str = typer.Option("", help="Extra Pyserini CLI flags, concatenated")
):
    lines = build_script_lines(str(index_path), str(topics), str(run),
                               str(qrels) if qrels else None, bm25=bm25, extra=extra)
    output_bash.write_text("\n".join(lines))
    typer.echo(f"Generated {output_bash}")

if __name__ == "__main__":
    app()
