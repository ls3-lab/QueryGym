from __future__ import annotations
from ..core.base import BaseReformulator, QueryItem, ReformulationResult
from ..core.registry import register_method
from ..core.utils import strip_think_trace
from typing import Any, Dict, List, Tuple
import random
import os
import csv
import json
import re

# Patterns for removing the conversational and markdown framing that chat-tuned
# models wrap around a keyword list (see Query2E._strip_framing)
_HEADING = re.compile(r"^#{1,6}\s")
_RULE = re.compile(r"^[-*_=]{3,}$")
_FENCE = re.compile(r"^```|^~~~")
_TABLE_ROW = re.compile(r"^\|.*\|$")
_BULLET = re.compile(r"^([\-\*•▪→►✅✔☑]|\(?\d+[\).:]|[a-z]\))\s+")
_BOLD_LABEL = re.compile(r"^(\*\*|__)([^*_]+?)(\*\*|__)\s*(:?)\s*(.*)$")
_HTML_INLINE = re.compile(r"</?(b|strong|em|i|u)\s*/?>", re.IGNORECASE)
_HTML_BLOCK = re.compile(r"</?(li|ul|ol|p|br)\s*/?>", re.IGNORECASE)
_QUERY_LINE = re.compile(r"^\**\s*query\s*\**\s*[:：]", re.IGNORECASE)
_KEYWORDS_LABEL = re.compile(
    r"^\**\s*(keywords?|mots[- ]cl[ée]s|palabras clave|schl[üu]sselw[öo]rter|关键词|キーワード|"
    r"ключевые слова)\s*(\([^)]*\))?\s*\**\s*[:：]\s*\**\s*",
    re.IGNORECASE,
)
_CLOSER_PHRASES = (
    r"note\b|would you like|let me know|i hope|hope this|feel free|"
    r"these (?:keywords|terms|should)|this list"
)
_CLOSER = re.compile(rf"^(?:{_CLOSER_PHRASES})", re.IGNORECASE)
_OPENER = re.compile(
    r"^(?:sure|certainly|of course|okay|absolutely|here(?:'s|’s| is| are)|below (?:is|are|you)|"
    rf"the following|i can help|to find|you (?:can|could|might|may)|if you|{_CLOSER_PHRASES})",
    re.IGNORECASE,
)
_OPENER_OTHER_LANG = re.compile(
    r"^(aqu[ií]|voici|ecco|hier (ist|sind)|a continuaci[oó]n|以下|下面|다음|voil[aà])",
    re.IGNORECASE,
)
_LEXICON = re.compile(r"\b(keyword|keywords|list|query|phrase|phrases|terms?)\b", re.IGNORECASE)
_LIST_VERBS = r"are|include|includes|including|would be|be|use|using"
_VERB = re.compile(rf"\b(?:{_LIST_VERBS}|is|can|should)\b", re.IGNORECASE)
_LIST_TAIL = re.compile(rf"(?:[:：]|\b(?:{_LIST_VERBS}|such as)\b)\s*", re.IGNORECASE)
_PROSE_WORDS = set(
    "you your i we these those can will would should could may might hope help helps let know "
    "feel free need want like please if here below following above some relevant related based "
    "broken".split()
)
_SENTENCE_END = (".", "。", "!", "！", "?", "？")


def _strip_emphasis(s: str) -> str:
    """Remove markdown/HTML emphasis while keeping identifiers like __init__."""
    s = _HTML_BLOCK.sub("", _HTML_INLINE.sub("", s))
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\w)__(?=[^_]*\s)(.+?)__(?!\w)", r"\1", s)
    s = re.sub(r"(?<!\w)\*(?=\S)(.+?)(?<=\S)\*(?!\w)", r"\1", s)
    s = re.sub(r"(?<!\w)`([^`]+)`(?!\w)", r"\1", s)
    return s.strip()


def _is_list_like(s: str) -> bool:
    """At least three short comma-separated items."""
    parts = [p.strip() for p in re.split(r"[,;，；、]", s) if p.strip()]
    return len(parts) >= 3 and sum(len(p.split()) for p in parts) / len(parts) <= 6


def _is_prose(s: str) -> bool:
    """Heuristic: the line reads like a sentence rather than keywords."""
    tokens = re.findall(r"[a-zA-Z']+", s.lower())
    if sum(1 for t in tokens if t in _PROSE_WORDS) >= 2:
        return True
    if _LEXICON.search(" ".join(s.split()[:6])) and _VERB.search(s):
        return True
    if len(tokens) >= 12 and (s.endswith(_SENTENCE_END) or "," not in s):
        return True
    return len(tokens) >= 6 and s.endswith(_SENTENCE_END[2:]) and "," not in s


def _list_tail(s: str) -> str | None:
    """Keyword list embedded at the end of a sentence ("... include a, b, c")."""
    tail = None
    for m in _LIST_TAIL.finditer(s):
        candidate = s[m.end() :].strip().rstrip("".join(_SENTENCE_END))
        if _is_list_like(candidate) and not _is_prose(candidate):
            tail = candidate
    if tail is None:
        return None
    tail = re.sub(r"^(and|or)\s+", "", tail.strip())
    return re.sub(r",\s*(and|or)\s+", ", ", tail)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


@register_method("query2e")
class Query2E(BaseReformulator):
    """
    Query2E: Query to keyword expansion.

    Modes:
        - zs (zero-shot): Simple keyword generation
        - fs (few-shot): Uses training examples for keyword generation

    Formula: (query × 5) + keywords

    Output Cleanup (params["clean_output"], default True):
        Chat-tuned models tend to wrap the keyword list in introductory and closing
        sentences, markdown headings, category labels and <think> reasoning traces.
        With clean_output enabled this framing is removed before keyword parsing.
        Set clean_output=False to parse the raw model output (behavior before 2.0).

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

    VERSION = "2.0"
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
                self.cfg.params.get("collection_path")
                or self.cfg.params.get("msmarco_collection")
                or os.getenv("COLLECTION_PATH")
                or os.getenv("MSMARCO_COLLECTION")
            )
            train_queries_path = (
                self.cfg.params.get("train_queries_path")
                or self.cfg.params.get("msmarco_train_queries")
                or os.getenv("TRAIN_QUERIES_PATH")
                or os.getenv("MSMARCO_TRAIN_QUERIES")
            )
            train_qrels_path = (
                self.cfg.params.get("train_qrels_path")
                or self.cfg.params.get("msmarco_train_qrels")
                or os.getenv("TRAIN_QRELS_PATH")
                or os.getenv("MSMARCO_TRAIN_QRELS")
            )

            # For BEIR, collection_path is actually the BEIR data directory
            if dataset_type == "beir":
                beir_data_dir = (
                    self.cfg.params.get("beir_data_dir")
                    or collection_path
                    or os.getenv("BEIR_DATA_DIR")
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
                with open(collection_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f, delimiter="\t")
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
                "train_qrels": train_qrels,
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
            keywords = [w.strip(".,!?;:") for w in words[:7] if len(w) > 3]
            keywords_str = ", ".join(keywords[:5]) if keywords else "keywords, terms, phrases"

            examples_text += f"Query: {query}\nKeywords: {keywords_str}\n"

        return examples_text

    def _clean_output_enabled(self) -> bool:
        return bool(self.cfg.params.get("clean_output", True))

    def effective_params(self) -> Dict[str, Any]:
        """Record clean_output when enabled; runs without it parsed the raw output."""
        params = super().effective_params()
        if self._clean_output_enabled():
            params["clean_output"] = True
        else:
            params.pop("clean_output", None)
        return params

    def _strip_framing(self, raw_output: str) -> Tuple[str, Dict[str, Any]]:
        """
        Remove the conversational and markdown framing that chat-tuned models add
        around a keyword list:
        - <think> reasoning traces
        - introductory/closing sentences ("Here is a list of keywords...", "I hope this helps!")
        - markdown headings, horizontal rules, code fences, category labels and emphasis
        - "Term: description" explanations (the term is kept)
        - further "Query:/Keywords:" pairs a model adds after its answer in few-shot mode

        Keyword lists, table cells and JSON string arrays are kept.

        Returns:
            (cleaned text in a format accepted by _parse_keywords, cleanup metadata)
        """
        meta: Dict[str, Any] = {}
        if not raw_output:
            return "", meta

        text, unclosed = strip_think_trace(raw_output)
        if unclosed:
            meta["unclosed_think"] = True

        lines = text.split("\n")
        kept: List[str] = []
        seen_content = False
        table_rows = 0
        keyword_columns: List[int] = []

        for i, line in enumerate(lines):
            s = line.strip()
            if not s:
                continue
            if _HEADING.match(s) or _RULE.match(s) or _FENCE.match(s):
                continue

            # Table: skip header and separator rows, keep the keyword cells. The
            # keyword column is taken from the header, else every column but the first
            if _TABLE_ROW.match(s):
                cells = [c.strip() for c in s.strip("|").split("|")]
                if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                    continue
                table_rows += 1
                if table_rows == 1:
                    keyword_columns = [
                        j for j, c in enumerate(cells) if re.search(r"keyword|term|phrase", c, re.I)
                    ]
                    continue
                if keyword_columns:
                    values = [cells[j] for j in keyword_columns if j < len(cells)]
                else:
                    values = cells[1:] if len(cells) > 1 else cells
                kept.append(", ".join(_strip_emphasis(c) for c in values if c))
                seen_content = True
                continue

            # JSON array of strings
            if s.startswith("[") and s.endswith("]"):
                try:
                    items = json.loads(s)
                except ValueError:
                    items = None
                if isinstance(items, list) and all(isinstance(x, str) for x in items):
                    kept.append(", ".join(items))
                    seen_content = True
                    continue

            # Few-shot continuation: the model started another example
            if _QUERY_LINE.match(s):
                if seen_content:
                    meta["truncated_at_query"] = True
                    break
                continue

            # HTML lists: "<li>term</li>" is a list item
            if _HTML_BLOCK.search(s):
                html_item = re.match(r"<li\b", s, re.I)
                s = _HTML_BLOCK.sub("", s).strip()
                if not s:
                    continue
                if html_item:
                    s = "- " + s

            body = _BULLET.sub("", s)
            is_list_item = body != s
            next_line = next((ln for ln in lines[i + 1 :] if ln.strip()), None)
            next_is_nested = next_line is not None and _indent(next_line) > _indent(line)

            # "Keywords: a, b, c"
            label = _KEYWORDS_LABEL.match(body)
            if label:
                body = body[label.end() :].strip()
                if body:
                    kept.append(_strip_emphasis(body))
                    seen_content = True
                continue

            # List item acting as a category header for a nested list
            if (
                is_list_item
                and next_is_nested
                and _BULLET.match(next_line.strip())
                and not re.search(r"[,:：]", _strip_emphasis(body))
            ):
                continue

            # Bold labels: "**Label:** a, b", "**Category**", "- **Term**: description"
            bold = _BOLD_LABEL.match(_HTML_INLINE.sub("**", body))
            if bold:
                name, colon_outside, rest = bold.group(2).strip(), bold.group(4), bold.group(5)
                if name.endswith(":"):
                    if rest:
                        kept.append(_strip_emphasis(rest))
                        seen_content = True
                    continue
                if not rest and not is_list_item:
                    continue
                if colon_outside or (rest and re.match(r"^[\-–—(]", rest)):
                    rest = rest.lstrip(":-–— ").strip()
                    if is_list_item and not _is_list_like(rest):
                        kept.append(name)
                    else:
                        kept.append(_strip_emphasis(rest) if rest else name)
                    seen_content = True
                    continue

            # Standalone "*Category*" line
            if not is_list_item and re.fullmatch(r"\*[^*]+\*|_[^_]+_", body):
                continue

            plain = _strip_emphasis(body)

            # "- term - explanation of the term"
            if is_list_item and re.search(r"\s[-–—]\s", plain):
                head, explanation = re.split(r"\s[-–—]\s", plain, maxsplit=1)
                if len(explanation.split()) >= 4 or _is_prose(explanation):
                    plain = head.strip()

            # Lines ending with a colon are lead-ins or category labels
            if plain.endswith((":", "：")):
                if not is_list_item or next_is_nested:
                    continue
                plain = plain.rstrip(":： ")

            if not is_list_item:
                # A bare keyword list is kept even if it starts like a sentence ("okay google, ...")
                if _is_list_like(plain) and not _VERB.search(plain) and not _CLOSER.match(plain):
                    kept.append(plain)
                    seen_content = True
                    continue
                if (
                    _OPENER.match(plain)
                    or _OPENER_OTHER_LANG.match(plain)
                    or _CLOSER.match(plain)
                    or _is_prose(plain)
                ):
                    tail = None
                    if (_is_list_like(plain) or "," in plain) and not _CLOSER.match(plain):
                        tail = _list_tail(plain)
                    if tail:
                        kept.append(tail)
                        seen_content = True
                    continue

            kept.append(plain)
            seen_content = True

        return "\n".join(kept), meta

    def _extract_keywords(self, raw_output: str) -> Tuple[List[str], Dict[str, Any]]:
        """Clean the model output and parse it into keywords."""
        if not raw_output or not raw_output.strip():
            return [], {}

        cleaned, meta = self._strip_framing(raw_output)
        terms = self._parse_keywords(cleaned, strict_numbering=True)
        if terms or meta.get("unclosed_think"):
            return terms, meta

        # Nothing survived cleanup: fall back to the output (without reasoning
        # traces) only if it is itself a plain keyword list
        raw_terms = self._parse_keywords(strip_think_trace(raw_output)[0], strict_numbering=True)
        if (
            len(raw_terms) >= 2
            and all(len(t.split()) <= 5 for t in raw_terms)
            and not any(_is_prose(t) for t in raw_terms)
        ):
            meta["cleanup_fallback"] = True
            return raw_terms, meta
        meta["cleanup_empty"] = True
        return [], meta

    def _parse_keywords(self, raw_output: str, strict_numbering: bool = False) -> List[str]:
        """
        Parse keywords from LLM output, handling various formats:
        - Comma-separated: "keyword1, keyword2, keyword3"
        - Bullet points: "- keyword1\n- keyword2"
        - Numbered lists: "1. keyword1\n2. keyword2"
        - Mixed formats

        Args:
            raw_output: Model output
            strict_numbering: Only treat a leading number as list numbering when it is
                followed by a delimiter and whitespace ("1. term"), so keywords such as
                "2024" or "401k" are kept intact

        Returns:
            List of cleaned keyword strings
        """
        if not raw_output or not raw_output.strip():
            return []

        # Normalize the text
        text = raw_output.strip()

        # Remove common prefixes like "Keywords:", "Here are the keywords:", etc.
        text = re.sub(
            r"^(keywords|here are|the keywords|list of keywords)[:\s]*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        keywords = []

        # Check if it's a bullet/numbered list format
        if "\n" in text or text.strip().startswith("-") or re.match(r"^\d+\.", text.strip()):
            # Split by newlines
            lines = text.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Remove bullet points: -, *, •, ▪, etc.
                line = re.sub(r"^[\-\*•▪→►]\s*", "", line)

                # Remove numbered prefixes: 1., 1), (1), etc.
                numbering = (
                    r"^[\(\[]?\d+[\)\]\.:\-]\s+"
                    if strict_numbering
                    else r"^[\(\[]?\d+[\)\]\.:\-]?\s*"
                )
                line = re.sub(numbering, "", line)

                # If line contains commas, it might be multiple keywords
                if "," in line:
                    for part in line.split(","):
                        part = part.strip()
                        if part:
                            keywords.append(part)
                elif line:
                    keywords.append(line)
        else:
            # Assume comma-separated format
            for part in text.split(","):
                part = part.strip()
                if part:
                    keywords.append(part)

        # Clean each keyword
        cleaned_keywords = []
        for kw in keywords:
            # Remove quotes
            kw = kw.strip("\"'`")
            # Remove trailing punctuation
            kw = kw.rstrip(".,;:!?")
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
                raise ValueError(
                    f"Invalid mode '{mode}' for Query2E. Must be 'zs' (zero-shot) or 'fs' (few-shot)."
                )

            # Generate keywords
            out = self.llm.chat(msgs, temperature=temperature, max_tokens=max_tokens)

            # Parse keywords, removing chat-model framing unless disabled
            if self._clean_output_enabled():
                terms, cleanup_meta = self._extract_keywords(out)
                metadata.update({"clean_output": True, **cleanup_meta})
            else:
                terms = self._parse_keywords(out)

            # Limit to max 20 keywords
            max_keywords = int(self.cfg.params.get("max_keywords", 20))
            if len(terms) > max_keywords:
                terms = terms[:max_keywords]

            generated_content = " ".join(terms)

            # Concatenate: query + keywords
            reformulated = self.concatenate_result(q.text, generated_content)

            metadata.update(
                {"keywords": terms, "temperature": temperature, "max_tokens": max_tokens}
            )

            return ReformulationResult(q.qid, q.text, reformulated, metadata=metadata)

        except Exception as e:
            # Graceful error handling - fallback to original query
            error_msg = f"Query2E failed: {e}"
            print(f"Error for qid={q.qid}: {error_msg}")

            return ReformulationResult(
                q.qid, q.text, q.text, metadata={"mode": mode, "error": error_msg, "fallback": True}
            )
