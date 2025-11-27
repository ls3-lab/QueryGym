# 🎮 QueryGym Hands-On Playground

**Welcome!** This is your interactive guide to exploring QueryGym's query reformulation capabilities. Run the code, modify it, experiment, and see how LLMs can transform short queries into powerful search queries!

---

## 🚀 Quick Start

### Installation

```bash
# Install QueryGym
pip install querygym
```

### Setup Ollama (Local LLM)

All examples in this guide use Ollama with the `qwen2.5:7b` model. To get started:

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Pull the model
ollama pull qwen2.5:7b

# Start Ollama server (usually runs automatically)
# The server will be available at http://127.0.0.1:11434
```

**Note:** All examples use `qwen2.5:7b` via Ollama's OpenAI-compatible API. You can replace it with any other Ollama model you have installed (e.g., `llama3.2:3b`, `mistral:7b`).

## 🎯 Part 1: Your First Reformulation

Let's start simple! Reformulate a single query.

```python
import querygym as qg

# Create a simple query
query = qg.QueryItem(qid="my_query", text="treatments for diabetes")

# Create a reformulator (we'll use GenQR Ensemble - it's great!)
reformulator = qg.create_reformulator(
    method_name="query2doc",
    model="qwen2.5:7b",
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",  # Ollama's OpenAI-compatible endpoint
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)

# Reformulate!
result = reformulator.reformulate(query)

# See the magic ✨
print("Original query:", result.original)
print("Reformulated:", result.reformulated)
print("\nMetadata:", result.metadata)
```

**Try it yourself!** Change the query to something you're curious about:
- "climate change"
- "machine learning"
- "python programming"
- "healthy recipes"

---

## 🎨 Part 2: Compare Different Methods

QueryGym has multiple reformulation methods. Let's see how they differ!

```python
import querygym as qg

# Your query
query = qg.QueryItem("q1", "artificial intelligence")

# Try different methods
methods = ["genqr_ensemble", "query2doc", "mugi"]

print("=" * 60)
print(f"Original Query: {query.text}\n")
print("=" * 60)

for method_name in methods:
    print(f"\n🔹 Method: {method_name.upper()}")
    print("-" * 60)
    
    reformulator = qg.create_reformulator(
        method_name=method_name,
        model="qwen2.5:7b",
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 1.0,
            "max_tokens": 128
        }
    )
    
    result = reformulator.reformulate(query)
    print(f"Reformulated: {result.reformulated}")
    print(f"Length: {len(result.reformulated.split())} words")
```

**What do you notice?**
- Which method produces longer expansions?
- Which adds more context?
- Which do you think would work best for search?

---

## 🎪 Part 3: Batch Processing - Reformulate Multiple Queries

Process multiple queries at once!

```python
import querygym as qg

# Create multiple queries
queries = [
    qg.QueryItem("q1", "python tutorials"),
    qg.QueryItem("q2", "healthy breakfast"),
    qg.QueryItem("q3", "climate solutions"),
    qg.QueryItem("q4", "machine learning basics"),
    qg.QueryItem("q5", "travel tips"),
]

# Create reformulator
reformulator = qg.create_reformulator(
    method_name="genqr_ensemble",
    model="qwen2.5:7b",
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)

# Process all at once (with progress bar!)
results = reformulator.reformulate_batch(queries)

# Display results
print("\n" + "=" * 70)
print("REFORMULATION RESULTS")
print("=" * 70)

for result in results:
    print(f"\n📌 Query ID: {result.qid}")
    print(f"   Original:  {result.original}")
    print(f"   Reformed:  {result.reformulated}")
    print(f"   Words:     {len(result.reformulated.split())} words")
```

**Challenge:** Add your own queries to the list and see how they're transformed!

---

## 🎭 Part 4: Experiment with Different Models

Compare how different LLM models reformulate queries!

```python
import querygym as qg

query = qg.QueryItem("q1", "quantum computing")

# Try different Ollama models (adjust based on what you have installed)
models = ["qwen2.5:7b", "llama3.2:3b", "mistral:7b"]  # Add more if you have access

print(f"Original Query: {query.text}\n")
print("=" * 70)

for model in models:
    print(f"\n🤖 Model: {model}")
    print("-" * 70)
    
    reformulator = qg.create_reformulator(
        method_name="genqr_ensemble",
        model=model,
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 1.0,
            "max_tokens": 128
        }
    )
    
    result = reformulator.reformulate(query)
    print(f"Reformulated: {result.reformulated}")
```

**Note:** Different models have different capabilities. Larger models (7b+) often produce better results but are slower.

---

## 🎨 Part 5: Temperature Playground

Temperature controls creativity. Let's see the difference!

```python
import querygym as qg

query = qg.QueryItem("q1", "sustainable energy")

temperatures = [0.0, 0.5, 0.8, 1.2]

print(f"Original Query: {query.text}\n")
print("=" * 70)

for temp in temperatures:
    print(f"\n🌡️  Temperature: {temp}")
    print("-" * 70)
    
    reformulator = qg.create_reformulator(
        method_name="genqr_ensemble",
        model="qwen2.5:7b",
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": temp,
            "max_tokens": 128
        }
    )
    
    result = reformulator.reformulate(query)
    print(f"Reformulated: {result.reformulated}")
    
    # Run multiple times to see variation (only if temp > 0)
    if temp > 0:
        print("\n   (Variation - run again to see different results)")
```

**What to observe:**
- `temperature=0.0`: Deterministic, same result every time
- `temperature=0.5`: Slightly creative
- `temperature=0.8`: More diverse expansions
- `temperature=1.2`: Very creative, sometimes unpredictable

---

## 🎯 Part 6: Method Deep Dive - Query2Doc

Query2Doc generates pseudo-documents. It's powerful for complex queries!

```python
import querygym as qg

# Query2Doc works great for complex, conceptual queries
queries = [
    qg.QueryItem("q1", "how does photosynthesis work"),
    qg.QueryItem("q2", "what is machine learning"),
    qg.QueryItem("q3", "explain quantum entanglement"),
]

# Create Query2Doc reformulator
reformulator = qg.create_reformulator(
    method_name="query2doc",
    model="qwen2.5:7b",
    params={"mode": "zs"},  # "zs" = zero-shot, "fs" = few-shot, "cot" = chain-of-thought
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 256
    }
)

print("=" * 70)
print("QUERY2DOC - Pseudo-Document Generation")
print("=" * 70)

for query in queries:
    result = reformulator.reformulate(query)
    
    print(f"\n📝 Query: {result.original}")
    print(f"📄 Generated Document: {result.reformulated[:200]}...")
    print(f"📊 Metadata: {result.metadata.get('mode', 'N/A')}")
```

**Try different modes:**
- `"zs"`: Zero-shot (default, fast)
- `"cot"`: Chain-of-thought (more reasoning)
- `"fs"`: Few-shot (requires training data)

---

## 🎪 Part 7: Side-by-Side Comparison

Create a beautiful comparison table!

```python
import querygym as qg
from tabulate import tabulate  # pip install tabulate

query = qg.QueryItem("q1", "renewable energy")

methods = ["genqr", "genqr_ensemble", "query2doc"]

results_data = []

for method in methods:
    reformulator = qg.create_reformulator(
        method_name=method,
        model="qwen2.5:7b",
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 1.0,
            "max_tokens": 128
        }
    )
    
    result = reformulator.reformulate(query)
    
    results_data.append([
        method.upper(),
        result.original,
        result.reformulated[:80] + "..." if len(result.reformulated) > 80 else result.reformulated,
        len(result.reformulated.split()),
    ])

print("\n" + "=" * 100)
print("METHOD COMPARISON")
print("=" * 100)
print(tabulate(
    results_data,
    headers=["Method", "Original", "Reformulated", "Word Count"],
    tablefmt="grid"
))
```

**Install tabulate if needed:**
```bash
pip install tabulate
```

---

## 🎨 Part 8: Interactive Query Reformulator

Create a simple interactive tool!

```python
import querygym as qg

def interactive_reformulator():
    """Interactive query reformulation tool"""
    
    print("🎮 QueryGym Interactive Reformulator")
    print("=" * 50)
    print("Type 'quit' to exit\n")
    
    # Create reformulator once (more efficient)
    reformulator = qg.create_reformulator(
        method_name="genqr_ensemble",
        model="qwen2.5:7b",
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 1.0,
            "max_tokens": 128
        }
    )
    
    while True:
        query_text = input("Enter your query: ").strip()
        
        if query_text.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not query_text:
            continue
        
        # Reformulate
        query = qg.QueryItem("interactive", query_text)
        result = reformulator.reformulate(query)
        
        print(f"\n✨ Original:  {result.original}")
        print(f"🚀 Reformed:  {result.reformulated}\n")
        print("-" * 50 + "\n")

# Run it!
if __name__ == "__main__":
    interactive_reformulator()
```

**Try it:** Run this and reformulate queries on the fly!

---

## 🎯 Part 9: Save and Load Results

Save your reformulations for later analysis!

```python
import querygym as qg
import json
from pathlib import Path

# Create queries
queries = [
    qg.QueryItem("q1", "python data science"),
    qg.QueryItem("q2", "healthy cooking"),
    qg.QueryItem("q3", "sustainable living"),
]

# Reformulate
reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="qwen2.5:7b",
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)
results = reformulator.reformulate_batch(queries)

# Save to JSON
output_file = Path("reformulations.json")
with open(output_file, "w") as f:
    json.dump([
        {
            "qid": r.qid,
            "original": r.original,
            "reformulated": r.reformulated,
            "metadata": r.metadata
        }
        for r in results
    ], f, indent=2)

print(f"✅ Saved {len(results)} reformulations to {output_file}")

# Load and display
print("\n📖 Loading saved results...")
with open(output_file, "r") as f:
    loaded = json.load(f)

for item in loaded:
    print(f"\n{item['qid']}: {item['original']} → {item['reformulated'][:60]}...")
```

---

## 🎪 Part 10: Method Showcase - All Methods

See all available methods in action!

```python
import querygym as qg

query = qg.QueryItem("showcase", "artificial intelligence applications")

# All available methods
all_methods = [
    "genqr",
    "genqr_ensemble", 
    "query2doc",
    "qa_expand",
    "mugi",
    "lamer",
    "query2e",
    "csqe",  # Note: CSQE requires contexts (see advanced section)
]

print("=" * 80)
print("ALL QUERYGYM METHODS SHOWCASE")
print("=" * 80)
print(f"Original Query: {query.text}\n")

for method in all_methods:
    try:
        print(f"\n{'='*80}")
        print(f"🔹 {method.upper()}")
        print(f"{'='*80}")
        
        reformulator = qg.create_reformulator(
            method_name=method,
            model="qwen2.5:7b",
            llm_config={
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "temperature": 1.0,
                "max_tokens": 128
            }
        )
        
        result = reformulator.reformulate(query)
        print(f"Reformulated: {result.reformulated}")
        print(f"Words: {len(result.reformulated.split())}")
        
    except Exception as e:
        print(f"❌ Error with {method}: {e}")
        print("   (Some methods may require additional setup)")

print("\n" + "=" * 80)
print("✅ Showcase complete!")
```

---

## 🎨 Part 11: Custom Configuration

Fine-tune your reformulator with custom parameters!

```python
import querygym as qg

query = qg.QueryItem("custom", "machine learning")

# Custom configuration
reformulator = qg.create_reformulator(
    method_name="genqr_ensemble",
    model="qwen2.5:7b",
    params={
        "repeat_query_weight": 5,  # Repeat query 5 times (default: 3)
        # Method-specific params go here
    },
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 200
    },
    seed=42  # For reproducibility
)

result = reformulator.reformulate(query)

print("Original:", result.original)
print("Reformulated:", result.reformulated)
print("\nFull metadata:", json.dumps(result.metadata, indent=2))
```

---

## 🎯 Part 12: Compare Original vs Reformulated

Visual comparison with word counts and analysis!

```python
import querygym as qg

queries = [
    qg.QueryItem("q1", "python"),
    qg.QueryItem("q2", "climate change"),
    qg.QueryItem("q3", "machine learning tutorial"),
]

reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="qwen2.5:7b",
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)
results = reformulator.reformulate_batch(queries)

print("\n" + "=" * 90)
print("ORIGINAL vs REFORMULATED COMPARISON")
print("=" * 90)

for result in results:
    orig_words = len(result.original.split())
    reform_words = len(result.reformulated.split())
    expansion_ratio = reform_words / orig_words if orig_words > 0 else 0
    
    print(f"\n📌 Query ID: {result.qid}")
    print(f"   Original:     {result.original}")
    print(f"   Words:        {orig_words}")
    print(f"   Reformulated: {result.reformulated}")
    print(f"   Words:        {reform_words}")
    print(f"   Expansion:    {expansion_ratio:.1f}x")
    print(f"   Added:        {reform_words - orig_words} words")

print("\n" + "=" * 90)
```

---

## 🎪 Part 13: Fun Queries Challenge

Try reformulating these fun and challenging queries!

```python
import querygym as qg

# Fun and challenging queries
fun_queries = [
    qg.QueryItem("fun1", "why is the sky blue"),
    qg.QueryItem("fun2", "how do birds fly"),
    qg.QueryItem("fun3", "what is the meaning of life"),
    qg.QueryItem("fun4", "best programming language"),
    qg.QueryItem("fun5", "time travel possible"),
    qg.QueryItem("fun6", "how to learn guitar"),
    qg.QueryItem("fun7", "sustainable fashion"),
    qg.QueryItem("fun8", "quantum physics explained"),
]

reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="qwen2.5:7b",
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)
results = reformulator.reformulate_batch(fun_queries)

print("\n" + "🎉" * 40)
print("FUN QUERIES CHALLENGE")
print("🎉" * 40 + "\n")

for result in results:
    print(f"❓ {result.original}")
    print(f"✨ {result.reformulated}\n")
    print("-" * 80 + "\n")
```

**Challenge:** Add your own fun queries and see how they're transformed!

---

## 🎨 Part 14: Domain-Specific Queries

Test how QueryGym handles different domains!

```python
import querygym as qg

# Queries from different domains
domain_queries = {
    "Science": [
        qg.QueryItem("sci1", "photosynthesis"),
        qg.QueryItem("sci2", "black holes"),
    ],
    "Technology": [
        qg.QueryItem("tech1", "blockchain"),
        qg.QueryItem("tech2", "neural networks"),
    ],
    "Health": [
        qg.QueryItem("health1", "meditation benefits"),
        qg.QueryItem("health2", "healthy diet"),
    ],
    "General": [
        qg.QueryItem("gen1", "travel tips"),
        qg.QueryItem("gen2", "cooking recipes"),
    ],
}

reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="qwen2.5:7b",
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)

print("\n" + "=" * 90)
print("DOMAIN-SPECIFIC REFORMULATION")
print("=" * 90)

for domain, queries in domain_queries.items():
    print(f"\n📚 Domain: {domain}")
    print("-" * 90)
    
    results = reformulator.reformulate_batch(queries)
    
    for result in results:
        print(f"\n  Query: {result.original}")
        print(f"  → {result.reformulated}")
```

---

## 🎯 Part 15: Advanced - Context-Aware Reformulation (CSQE)

Some methods like CSQE need retrieved contexts. Here's how to use them!

```python
import querygym as qg

# CSQE requires contexts (retrieved documents)
query = qg.QueryItem("csqe1", "machine learning")

# Option 1: Provide contexts manually
contexts = [
    "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
    "Deep learning uses neural networks with multiple layers to process complex patterns.",
    "Supervised learning trains models on labeled data to make predictions.",
    "Unsupervised learning finds patterns in data without labels.",
]

# Create CSQE reformulator
reformulator = qg.create_reformulator(
    method_name="csqe",
    model="qwen2.5:7b",
    params={
        "retrieval_k": 4,  # Use top 4 contexts
    },
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)

# Reformulate with contexts
result = reformulator.reformulate(query, contexts=contexts)

print("Original:", result.original)
print("Reformulated:", result.reformulated)
print("\nMetadata:", json.dumps(result.metadata, indent=2))
```

**Note:** For production use, you'd retrieve contexts using Pyserini or another search system.

---

## 🎯 Part 15.5: Pyserini Integration with QueryGym Wrapper

Use Pyserini to retrieve contexts automatically! This example shows how to wrap a Pyserini searcher and pass it to QueryGym methods.

### Prerequisites

```bash
# Install Pyserini (if not already installed)
pip install pyserini

# Note: Pyserini will auto-download prebuilt indices when first used
# Common prebuilt indices: "msmarco-v1-passage", "beir-v1.0.0-nfcorpus"
```

### Example 1: CSQE with Wrapped Pyserini Searcher

```python
import querygym as qg
from pyserini.search.lucene import LuceneSearcher

# Step 1: Create Pyserini searcher
print("🔍 Setting up Pyserini searcher...")
pyserini_searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")

# Step 2: Configure BM25 parameters (optional, but recommended)
pyserini_searcher.set_bm25(k1=0.82, b=0.68)  # MS MARCO passage defaults

# Step 3: Wrap the searcher with QueryGym's wrapper
wrapped_searcher = qg.wrap_pyserini_searcher(
    pyserini_searcher,
    answer_key="contents"  # Field to extract content from
)

print("✅ Searcher wrapped and ready!\n")

# Step 4: Create queries
queries = [
    qg.QueryItem("q1", "machine learning"),
    qg.QueryItem("q2", "climate change"),
    qg.QueryItem("q3", "quantum computing"),
]

# Step 5: Create CSQE reformulator with wrapped searcher
reformulator = qg.create_reformulator(
    method_name="csqe",
    model="qwen2.5:7b",
    params={
        "searcher": wrapped_searcher,  # Pass the wrapped searcher here!
        "retrieval_k": 10,  # Retrieve top 10 documents per query
        "gen_num": 2,  # Generate 2 expansions (KEQE + CSQE)
    },
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)

print("=" * 90)
print("CSQE WITH PYSERINI - AUTOMATIC CONTEXT RETRIEVAL")
print("=" * 90)
print("\nReformulating queries with automatically retrieved contexts...\n")

# Step 6: Reformulate - contexts are retrieved automatically!
results = reformulator.reformulate_batch(queries)

for result in results:
    print(f"📌 Query: {result.original}")
    print(f"✨ Reformulated: {result.reformulated[:150]}...")
    print(f"📊 Metadata:")
    print(f"   - Used contexts: {result.metadata.get('used_ctx', 'N/A')}")
    print(f"   - Total generations: {result.metadata.get('total_generations', 'N/A')}")
    print(f"   - KEQE passages: {len(result.metadata.get('keqe_passages', []))}")
    print(f"   - CSQE sentences: {len(result.metadata.get('csqe_sentences', []))}")
    print("-" * 90 + "\n")
```

### Example 2: LameR with Wrapped Pyserini Searcher

```python
import querygym as qg
from pyserini.search.lucene import LuceneSearcher

# Step 1: Create and wrap Pyserini searcher
pyserini_searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
pyserini_searcher.set_bm25(k1=0.82, b=0.68)

wrapped_searcher = qg.wrap_pyserini_searcher(
    pyserini_searcher,
    answer_key="contents"
)

# Step 2: Create queries
queries = [
    qg.QueryItem("q1", "artificial intelligence applications"),
    qg.QueryItem("q2", "renewable energy solutions"),
]

# Step 3: Create LameR reformulator with wrapped searcher
reformulator = qg.create_reformulator(
    method_name="lamer",
    model="qwen2.5:7b",
    params={
        "searcher": wrapped_searcher,  # Pass the wrapped searcher!
        "retrieval_k": 10,  # Use top 10 contexts
        "gen_passages": 5,  # Generate 5 passages per query
    },
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)

print("=" * 90)
print("LAMER WITH PYSERINI - AUTOMATIC CONTEXT RETRIEVAL")
print("=" * 90)
print("\nReformulating queries with automatically retrieved contexts...\n")

# Step 4: Reformulate - contexts are retrieved automatically!
results = reformulator.reformulate_batch(queries)

for result in results:
    print(f"📌 Query: {result.original}")
    print(f"✨ Reformulated: {result.reformulated[:150]}...")
    print(f"📊 Metadata:")
    print(f"   - Generated passages: {result.metadata.get('generated_passages_count', 'N/A')}")
    print(f"   - Used contexts: {result.metadata.get('used_ctx', 'N/A')}")
    print("-" * 90 + "\n")
```

### Example 3: Using Local Pyserini Index

If you have a local Lucene index instead of a prebuilt one:

```python
import querygym as qg
from pyserini.search.lucene import LuceneSearcher

# Step 1: Create searcher from local index path
local_index_path = "/path/to/your/lucene/index"
pyserini_searcher = LuceneSearcher(local_index_path)

# Step 2: Configure BM25 with custom parameters
pyserini_searcher.set_bm25(k1=0.9, b=0.4)  # Custom parameters

# Step 3: Wrap the searcher
wrapped_searcher = qg.wrap_pyserini_searcher(
    pyserini_searcher,
    answer_key="contents"
)

# Step 4: Create reformulator
reformulator = qg.create_reformulator(
    method_name="csqe",
    model="qwen2.5:7b",
    params={
        "searcher": wrapped_searcher,
        "retrieval_k": 10,
    },
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)

# Step 5: Use it!
query = qg.QueryItem("q1", "information retrieval")
result = reformulator.reformulate(query)

print(f"Query: {result.original}")
print(f"Reformulated: {result.reformulated}")
```

### Example 4: Error Handling

```python
import querygym as qg

try:
    from pyserini.search.lucene import LuceneSearcher
    
    # Create and wrap searcher
    pyserini_searcher = LuceneSearcher.from_prebuilt_index("msmarco-v1-passage")
    wrapped_searcher = qg.wrap_pyserini_searcher(pyserini_searcher)
    
    # Create reformulator
    reformulator = qg.create_reformulator(
        method_name="csqe",
        model="qwen2.5:7b",
        params={
            "searcher": wrapped_searcher,
            "retrieval_k": 10,
        },
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 1.0,
            "max_tokens": 128
        }
    )
    
    # Use it
    query = qg.QueryItem("q1", "test query")
    result = reformulator.reformulate(query)
    print(f"Success! Reformulated: {result.reformulated}")
    
except ImportError:
    print("❌ Pyserini not installed. Install with: pip install pyserini")
except Exception as e:
    print(f"❌ Error: {e}")
    print("   Falling back to manual contexts...")
    
    # Fallback: use manual contexts
    contexts = ["Context 1", "Context 2", "Context 3"]
    reformulator = qg.create_reformulator(
        method_name="csqe",
        model="qwen2.5:7b",
        params={"retrieval_k": 3},
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 1.0,
            "max_tokens": 128
        }
    )
    result = reformulator.reformulate(query, contexts=contexts)
    print(f"Fallback result: {result.reformulated}")
```

### Tips for Pyserini Integration

1. **Prebuilt Indices**: Pyserini provides many prebuilt indices:
   - `msmarco-v1-passage`: MS MARCO passage corpus (recommended for most use cases)
   - `msmarco-v1-doc`: MS MARCO document corpus
   - `beir-v1.0.0-{dataset}`: BEIR datasets (e.g., `beir-v1.0.0-nfcorpus`)

2. **BM25 Parameters**: Different indices may need different BM25 parameters:
   - MS MARCO passage: `k1=0.82, b=0.68` (default)
   - MS MARCO document: `k1=4.46, b=0.82`
   - You can experiment with different values for your use case

3. **Performance**: 
   - Batch retrieval is automatically used when calling `reformulate_batch()`
   - The wrapped searcher supports parallel processing
   - First run may be slower as Pyserini downloads the index

4. **Answer Key**: The `answer_key` parameter tells the wrapper which field to extract:
   - `"contents"` is common for MS MARCO indices
   - For custom indices, check your document structure

5. **Reusability**: You can reuse the same wrapped searcher for multiple reformulators:
   ```python
   # Create once
   wrapped_searcher = qg.wrap_pyserini_searcher(pyserini_searcher)
   
   # Use with multiple methods
   csqe_reformulator = qg.create_reformulator("csqe", ..., params={"searcher": wrapped_searcher})
   lamer_reformulator = qg.create_reformulator("lamer", ..., params={"searcher": wrapped_searcher})
   ```

---

## 🎪 Part 16: Experiment with Different Prompt Strategies

Query2Doc has multiple modes - let's compare them!

```python
import querygym as qg

query = qg.QueryItem("modes", "how does the internet work")

modes = ["zs", "cot"]  # Zero-shot and Chain-of-Thought

print("=" * 90)
print("QUERY2DOC MODE COMPARISON")
print("=" * 90)
print(f"Query: {query.text}\n")

for mode in modes:
    print(f"\n{'='*90}")
    print(f"Mode: {mode.upper()}")
    if mode == "zs":
        print("(Zero-shot: Direct generation)")
    elif mode == "cot":
        print("(Chain-of-Thought: With reasoning)")
    print(f"{'='*90}")
    
    reformulator = qg.create_reformulator(
        method_name="query2doc",
        model="qwen2.5:7b",
        params={"mode": mode},
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 1.0,
            "max_tokens": 256
        }
    )
    
    result = reformulator.reformulate(query)
    print(f"\nReformulated:\n{result.reformulated}\n")
```

---

## 🎨 Part 17: Create Your Own Test Suite

Build a test suite to evaluate reformulation quality!

```python
import querygym as qg
import json

# Your test queries
test_queries = [
    {"qid": "t1", "text": "python", "expected_keywords": ["programming", "language"]},
    {"qid": "t2", "text": "climate", "expected_keywords": ["change", "warming"]},
    {"qid": "t3", "text": "ai", "expected_keywords": ["artificial", "intelligence"]},
]

reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="qwen2.5:7b",
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)

print("=" * 90)
print("REFORMULATION TEST SUITE")
print("=" * 90)

for test in test_queries:
    query = qg.QueryItem(test["qid"], test["text"])
    result = reformulator.reformulate(query)
    
    # Simple check: do expected keywords appear?
    reformulated_lower = result.reformulated.lower()
    found_keywords = [
        kw for kw in test["expected_keywords"]
        if kw.lower() in reformulated_lower
    ]
    
    print(f"\n✅ Query: {test['text']}")
    print(f"   Reformulated: {result.reformulated}")
    print(f"   Expected keywords found: {found_keywords}")
    print(f"   Coverage: {len(found_keywords)}/{len(test['expected_keywords'])}")
```

---

## 🎯 Part 18: Export Reformulations for Search

Export reformulated queries in search engine format!

```python
import querygym as qg
from pathlib import Path

queries = [
    qg.QueryItem("q1", "python tutorial"),
    qg.QueryItem("q2", "healthy recipes"),
    qg.QueryItem("q3", "travel guide"),
]

reformulator = qg.create_reformulator(
    "genqr_ensemble",
    model="qwen2.5:7b",
    llm_config={
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "ollama",
        "temperature": 1.0,
        "max_tokens": 128
    }
)
results = reformulator.reformulate_batch(queries)

# Export as TSV (query_id \t reformulated_query)
output_file = Path("reformulated_queries.tsv")
with open(output_file, "w") as f:
    for result in results:
        f.write(f"{result.qid}\t{result.reformulated}\n")

print(f"✅ Exported {len(results)} reformulated queries to {output_file}")

# Display
print("\n📄 Exported queries:")
with open(output_file, "r") as f:
    print(f.read())
```

---

## 🎪 Part 19: Performance Comparison

Compare speed and quality across methods!

```python
import querygym as qg
import time

query = qg.QueryItem("perf", "machine learning")

methods = ["genqr", "genqr_ensemble", "query2doc"]

print("=" * 90)
print("PERFORMANCE COMPARISON")
print("=" * 90)
print(f"Query: {query.text}\n")

results = []

for method in methods:
    print(f"Testing {method}...")
    
    reformulator = qg.create_reformulator(
        method_name=method,
        model="qwen2.5:7b",
        llm_config={
            "base_url": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
            "temperature": 1.0,
            "max_tokens": 128
        }
    )
    
    # Time it
    start = time.time()
    result = reformulator.reformulate(query)
    elapsed = time.time() - start
    
    results.append({
        "method": method,
        "time": elapsed,
        "words": len(result.reformulated.split()),
        "result": result.reformulated
    })
    
    print(f"  ⏱️  Time: {elapsed:.2f}s")
    print(f"  📊 Words: {len(result.reformulated.split())}\n")

# Summary
print("=" * 90)
print("SUMMARY")
print("=" * 90)
for r in results:
    print(f"{r['method']:20} | {r['time']:6.2f}s | {r['words']:4} words")
```

---

## 🎨 Part 20: Final Challenge - Build Your Own Tool

Combine everything into a useful tool!

```python
import querygym as qg
import json
from pathlib import Path
from datetime import datetime

class QueryReformulationTool:
    """Your custom query reformulation tool"""
    
    def __init__(self, method="genqr_ensemble", model="qwen2.5:7b"):
        self.reformulator = qg.create_reformulator(
            method,
            model=model,
            llm_config={
                "base_url": "http://127.0.0.1:11434/v1",
                "api_key": "ollama",
                "temperature": 1.0,
                "max_tokens": 128
            }
        )
        self.history = []
    
    def reformulate(self, query_text, save=True):
        """Reformulate a query and optionally save to history"""
        query = qg.QueryItem(f"q{len(self.history)+1}", query_text)
        result = self.reformulator.reformulate(query)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "original": result.original,
            "reformulated": result.reformulated,
            "metadata": result.metadata
        }
        
        if save:
            self.history.append(entry)
        
        return result
    
    def batch_reformulate(self, query_texts):
        """Reformulate multiple queries"""
        queries = [qg.QueryItem(f"q{i+1}", text) for i, text in enumerate(query_texts)]
        results = self.reformulator.reformulate_batch(queries)
        
        for result in results:
            self.history.append({
                "timestamp": datetime.now().isoformat(),
                "original": result.original,
                "reformulated": result.reformulated,
                "metadata": result.metadata
            })
        
        return results
    
    def save_history(self, filepath="reformulation_history.json"):
        """Save reformulation history"""
        with open(filepath, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ Saved {len(self.history)} reformulations to {filepath}")
    
    def show_stats(self):
        """Show statistics about reformulations"""
        if not self.history:
            print("No reformulations yet!")
            return
        
        total = len(self.history)
        avg_original = sum(len(h["original"].split()) for h in self.history) / total
        avg_reformed = sum(len(h["reformulated"].split()) for h in self.history) / total
        
        print(f"\n📊 Statistics:")
        print(f"   Total reformulations: {total}")
        print(f"   Avg original words: {avg_original:.1f}")
        print(f"   Avg reformed words: {avg_reformed:.1f}")
        print(f"   Avg expansion: {avg_reformed/avg_original:.1f}x")

# Use it!
tool = QueryReformulationTool(method="genqr_ensemble", model="qwen2.5:7b")

# Reformulate some queries
tool.reformulate("python programming")
tool.reformulate("healthy eating")
tool.batch_reformulate(["machine learning", "climate change", "travel tips"])

# Show stats
tool.show_stats()

# Save history
tool.save_history()

# Display recent reformulations
print("\n📝 Recent reformulations:")
for entry in tool.history[-3:]:
    print(f"\n  {entry['original']} → {entry['reformulated'][:60]}...")
```

---

## 🎯 Tips & Tricks

### 1. **Performance Optimization**
- Use smaller models (e.g., `llama3.2:3b`) for faster results
- Lower `max_tokens` to reduce processing time
- Batch process to amortize overhead

### 2. **Quality Optimization**
- Use larger models (e.g., `qwen2.5:7b`, `mistral:7b`) for better quality
- Adjust `temperature` (0.8-1.2 for balance)
- Try different methods for different query types

### 3. **Reproducibility**
- Set `seed=42` for deterministic results
- Save metadata with results
- Version your configurations

### 4. **Error Handling**
```python
try:
    result = reformulator.reformulate(query)
except Exception as e:
    print(f"Error: {e}")
    # Fallback to original query
    result = qg.ReformulationResult(query.qid, query.text, query.text)
```

---

## 🎪 Next Steps

1. **Experiment**: Try different queries, methods, and parameters
2. **Compare**: See which methods work best for your use case
3. **Integrate**: Use reformulated queries in your search system
4. **Extend**: Build your own reformulation methods!

---

## 📚 Resources

- **Documentation**: https://querygym.readthedocs.io/
- **GitHub**: https://github.com/ls3-lab/QueryGym
- **Paper**: Check the README for citation

---

## 🎉 Have Fun!

Experiment, break things, learn, and most importantly - **have fun exploring query reformulation!** 🚀

**Pro Tip**: Start with simple queries, then gradually try more complex ones. Compare methods side-by-side to understand their strengths!

---

*Happy Reformulating!* ✨

