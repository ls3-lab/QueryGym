# QueryGym + Pyserini Pipeline

End-to-end pipeline for LLM-based query reformulation with Pyserini retrieval and evaluation.

## 🎯 **Overview**

This pipeline combines:
- **QueryGym**: LLM-based query reformulation methods
- **Pyserini**: BM25 sparse retrieval with prebuilt indices and evaluation

## 📋 **Pipeline Steps**

1. **Reformulate**: Load queries from Pyserini topics and reformulate using QueryGym
2. **Retrieve**: Retrieve documents using Pyserini with BM25
3. **Evaluate**: Evaluate results using trec_eval

## 🚀 **Quick Start**

### **Full Pipeline**

```bash
python scripts/querygym_pyserini/pipeline.py \
  --dataset msmarco-v1-passage.trecdl2019 \
  --method query2doc \
  --model your-model-name \
  --base-url http://your-llm-endpoint/v1 \
  --api-key your-api-key \
  --output-dir outputs/dl19_query2doc
```

### **List Available Datasets**

```bash
python scripts/querygym_pyserini/pipeline.py --list-datasets
```

### **Show Dataset Info**

```bash
python scripts/querygym_pyserini/pipeline.py \
  --dataset-info msmarco-v1-passage.trecdl2019
```

## 📖 **Individual Steps**

### **1. Query Reformulation Only**

```bash
python scripts/querygym_pyserini/reformulate_queries.py \
  --dataset msmarco-v1-passage.trecdl2019 \
  --method query2doc \
  --model your-model-name \
  --base-url http://your-llm-endpoint/v1 \
  --api-key your-api-key \
  --output-dir outputs/dl19_query2doc
```

**Options:**
- `--method`: QueryGym method (genqr, genqr_ensemble, query2doc, qa_expand, mugi, lamer, query2e, csqe)
- `--model`: LLM model name (e.g., qwen2.5:7b, llama3.1:8b, gpt-4, etc.)
- `--base-url`: LLM API endpoint (e.g., http://localhost:11434/v1)
- `--api-key`: LLM API key
- `--temperature`: LLM temperature (default: 1.0)
- `--max-tokens`: Max tokens (default: 128)
- `--retrieval-k`: Number of documents to retrieve for methods that need context (default: 10)

**Note:** If `--base-url` and `--api-key` are not provided, they will be read from `querygym/config/defaults.yaml`

### **2. Document Retrieval Only**

```bash
python scripts/querygym_pyserini/retrieve.py \
  --dataset msmarco-v1-passage.trecdl2019 \
  --queries outputs/dl19_query2doc/queries/reformulated_queries.tsv \
  --output-dir outputs/dl19_query2doc \
  --k 1000 \
  --threads 16
```

**Options:**
- `--k`: Number of documents to retrieve per query (default: 1000)
- `--threads`: Number of threads for parallel retrieval (default: 16)

### **3. Evaluation Only**

```bash
python scripts/querygym_pyserini/evaluate.py \
  --dataset msmarco-v1-passage.trecdl2019 \
  --run outputs/dl19_query2doc/runs/run.txt \
  --output-dir outputs/dl19_query2doc
```

## 🔄 **Advanced Usage**

### **Run Specific Steps**

```bash
# Only reformulation and retrieval
python scripts/querygym_pyserini/pipeline.py \
  --dataset beir-v1.0.0-nfcorpus \
  --method query2doc \
  --model your-model-name \
  --base-url http://your-llm-endpoint/v1 \
  --api-key your-api-key \
  --steps reformulate,retrieve \
  --output-dir outputs/nfcorpus_q2d
```

### **Resume Pipeline**

```bash
# Skip reformulation, run retrieval and evaluation
python scripts/querygym_pyserini/pipeline.py \
  --dataset msmarco-v1-passage.trecdl2019 \
  --method query2doc \
  --model your-model-name \
  --base-url http://your-llm-endpoint/v1 \
  --api-key your-api-key \
  --steps retrieve,evaluate \
  --output-dir outputs/dl19_query2doc
```

### **Batch Experiments**

```bash
# Test multiple methods on same dataset
for method in genqr genqr_ensemble query2doc; do
  python scripts/querygym_pyserini/pipeline.py \
    --dataset beir-v1.0.0-nfcorpus \
    --method $method \
    --model your-model-name \
    --base-url http://your-llm-endpoint/v1 \
    --api-key your-api-key \
    --output-dir outputs/nfcorpus_$method
done
```

## 📁 **Output Structure**

```
outputs/<dataset>_<method>/
├── logs/
│   └── pipeline_YYYYMMDD_HHMMSS.log
├── queries/
│   ├── original_queries.tsv
│   └── reformulated_queries.tsv
├── runs/
│   ├── run.txt                        # TREC run file
│   └── retrieval_log.txt
├── eval/
│   ├── eval_results.txt               # Full Pyserini eval output
│   ├── eval_results.json              # Parsed metrics
│   └── eval_summary.txt               # Human-readable summary
├── reformulation_metadata.json
├── retrieval_metadata.json
├── evaluation_metadata.json
├── pipeline_summary.json
├── pipeline_summary.txt
└── reformulation_samples.txt          # Sample reformulations
```

## 📊 **Available Datasets**

### **MS MARCO**
- `msmarco-v1-passage.dev` - MS MARCO Passage Dev
- `msmarco-v1-passage.trecdl2019` - TREC DL 2019 Passage
- `msmarco-v1-passage.trecdl2020` - TREC DL 2020 Passage

### **BEIR (v1.0.0)**
- `beir-v1.0.0-trec-covid` - TREC-COVID
- `beir-v1.0.0-bioasq` - BioASQ
- `beir-v1.0.0-nfcorpus` - NFCorpus
- `beir-v1.0.0-nq` - Natural Questions
- `beir-v1.0.0-hotpotqa` - HotpotQA
- `beir-v1.0.0-fiqa` - FiQA
- `beir-v1.0.0-scifact` - SciFact
- `beir-v1.0.0-fever` - FEVER
- ... and more (see `dataset_registry.yaml`)

## 🔧 **Requirements**

### **Python Packages**
```bash
pip install querygym pyserini pyyaml
```

### **System Requirements**
- **Java 21**: Required by Pyserini for evaluation
  ```bash
  # Ubuntu/Debian
  sudo apt install openjdk-21-jdk
  
  # Verify installation
  java -version  # Should show version 21
  ```

**Note:** Pyserini includes its own evaluation tools, so no separate trec_eval installation is needed!

### **LLM Server**
Any OpenAI-compatible API endpoint:
- **Ollama**: https://ollama.com
- **vLLM**: For high-performance serving
- **OpenAI API**: For GPT models
- **Other**: Any service with OpenAI-compatible chat completions API

Configure your LLM endpoint in `querygym/config/defaults.yaml`:
```yaml
llm:
  model: "your-model-name"
  base_url: "http://your-llm-endpoint/v1"
  api_key: "your-api-key"
```

## 💡 **Tips**

1. **Start Small**: Test with a BEIR dataset (smaller, faster) before MS MARCO
2. **Verify LLM**: Ensure your LLM endpoint is running and accessible
3. **Monitor Resources**: Large models (70B+) need significant RAM/VRAM
4. **Save Outputs**: All intermediate files are saved for debugging
5. **Resume Failed Runs**: Use `--steps` to skip completed steps
6. **Check Java**: Evaluation requires Java 21 (`java -version`)

## 🐛 **Troubleshooting**

### **Java Version Error**
```bash
# Error: UnsupportedClassVersionError
# Solution: Upgrade to Java 21

# Check current Java version
java -version

# Find Java 21 installation
update-alternatives --list java

# Set Java 21 for current session (adjust path to match your Java 21 installation)
export JAVA_HOME=/path/to/your/java-21-installation
export PATH=$JAVA_HOME/bin:$PATH

# Verify
java -version  # Should show version 21
```

### **Pyserini Index Not Found**
```bash
# Pyserini will auto-download prebuilt indices and qrels
# First run may take time to download
```

### **LLM Connection Error**
```bash
# Verify your LLM endpoint is accessible
curl http://your-llm-endpoint/v1/models

# Check configuration in querygym/config/defaults.yaml
# Ensure base_url, api_key, and model are correctly set
```

### **Out of Memory**
```bash
# Use smaller model
--model smaller-model-name  # e.g., 7B instead of 70B

# Reduce threads
--threads 4  # instead of 16

# Reduce max tokens
--max-tokens 64  # instead of 128
```

## 📚 **Examples**

### **Example 1: TREC DL 2019 with Query2Doc**
```bash
python scripts/querygym_pyserini/pipeline.py \
  --dataset msmarco-v1-passage.trecdl2019 \
  --method query2doc \
  --model your-model-name \
  --base-url http://your-llm-endpoint/v1 \
  --api-key your-api-key \
  --output-dir outputs/dl19_query2doc
```

**Expected output:**
```
map                     : 0.4567
ndcg_cut.10             : 0.6789
recall.1000             : 0.8234
```

### **Example 2: NFCorpus with Query2Doc**
```bash
python scripts/querygym_pyserini/pipeline.py \
  --dataset beir-v1.0.0-nfcorpus \
  --method query2doc \
  --model your-model-name \
  --base-url http://your-llm-endpoint/v1 \
  --api-key your-api-key \
  --temperature 0.7 \
  --output-dir outputs/nfcorpus_q2d
```

### **Example 3: Compare Methods**
```bash
# Create a comparison script
cat > run_comparison.sh << 'EOF'
#!/bin/bash
DATASET="beir-v1.0.0-scifact"
MODEL="your-model-name"
BASE_URL="http://your-llm-endpoint/v1"
API_KEY="your-api-key"

for METHOD in genqr genqr_ensemble query2doc; do
  echo "Running $METHOD..."
  python scripts/querygym_pyserini/pipeline.py \
    --dataset $DATASET \
    --method $METHOD \
    --model $MODEL \
    --base-url $BASE_URL \
    --api-key $API_KEY \
    --output-dir outputs/${DATASET}_${METHOD}
done

echo "Comparison complete!"
EOF

chmod +x run_comparison.sh
./run_comparison.sh
```

