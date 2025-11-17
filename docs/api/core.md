# Core API Reference

## Data Structures

### QueryItem

Represents a query with its ID and text.

```python
from querygym import QueryItem

query = QueryItem(qid="q1", text="what causes diabetes")
```

**Attributes:**
- `qid: str` - Query ID
- `text: str` - Query text

### ReformulationResult

Result of a query reformulation.

**Attributes:**
- `qid: str` - Query ID
- `original: str` - Original query text
- `reformulated: str` - Reformulated query text
- `metadata: Dict[str, Any]` - Additional metadata

### MethodConfig

Configuration for a reformulation method.

**Attributes:**
- `name: str` - Method name
- `params: Dict[str, Any]` - Method-specific parameters
- `llm: Dict[str, Any]` - LLM configuration
- `seed: int` - Random seed (default: 42)
- `retries: int` - Number of retries on failure (default: 2)

### BaseReformulator

Base class for all reformulation methods.

**Methods:**
- `reformulate(q: QueryItem, contexts: Optional[List[str]] = None) -> ReformulationResult`
- `reformulate_batch(queries: List[QueryItem], contexts: Optional[Dict[str, List[str]]] = None) -> List[ReformulationResult]`

## High-Level API

### create_reformulator

Create a reformulator instance.

```python
import querygym as qg

reformulator = qg.create_reformulator(
    method_name="genqr_ensemble",
    model="gpt-4",
    params={"repeat_query_weight": 3}
)
```

**Parameters:**
- `method_name: str` - Name of the reformulation method
- `model: str` - LLM model name
- `params: Dict[str, Any]` - Method-specific parameters (optional)
- `llm_config: Dict[str, Any]` - LLM configuration (optional)

### load_queries

Load queries from a file.

```python
import querygym as qg

queries = qg.load_queries("queries.tsv")
```

**Parameters:**
- `path: str` - Path to queries file
- `format: str` - File format ("tsv" or "jsonl", default: "tsv")

**Returns:** `List[QueryItem]`

### load_qrels

Load qrels (relevance judgments) from a file.

```python
qrels = qg.load_qrels("qrels.txt")
```

**Parameters:**
- `path: str` - Path to qrels file

**Returns:** `Dict[str, Dict[str, int]]` - Nested dict: qid → docid → relevance

### load_contexts

Load contexts from a JSONL file.

```python
contexts = qg.load_contexts("contexts.jsonl")
```

**Parameters:**
- `path: str` - Path to contexts file

**Returns:** `Dict[str, List[str]]` - Dict: qid → list of context texts

## LLM Client

### OpenAICompatibleClient

Client for OpenAI-compatible APIs.

```python
from querygym.core.llm import OpenAICompatibleClient

client = OpenAICompatibleClient(
    model="gpt-4",
    base_url="https://api.openai.com/v1",
    api_key="your-key"
)

response = client.chat([
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"}
])
```

## Prompt Bank

### PromptBank

Manages prompts from the YAML prompt bank.

```python
from querygym.core.prompts import PromptBank
from pathlib import Path

pb = PromptBank(Path("querygym/prompt_bank.yaml"))
prompt = pb.get("genqr_keywords")
```
