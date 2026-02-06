# VFinal4 - Production Medical Coding Pipeline

**Created**: January 12, 2026  
**Accuracy**: 90% CPT on 10-sample tests

## Overview

This is the production-ready version of the NyxMed medical coding pipeline, achieving 90% CPT accuracy with the following key components:

- **RAG Retrieval**: FAISS + BM25 hybrid search with CrossEncoder reranking
- **LLM**: Fine-tuned Llama 3.3 70B (`vineetdaniels/NYXMed-Final-1-7`) on HuggingFace
- **Section Extractor**: Extracts INDICATION, IMPRESSION, LATERALITY for better RAG matching

## Key Fixes in VFinal4

1. **Fixed prompt building** - Now includes 3 few-shot examples from RAG
2. **Fixed RAG content parsing** - Strips dashes from `parsed_compact` to avoid split issues
3. **Added section extractor** - Uses compact query format for better RAG matching
4. **All prompt sections present**:
   - Few-shot examples
   - SIMILAR CASES CPT CODES
   - Available ICD codes
   - Rules
   - Output format

## Directory Structure

```
VFinal4/
├── src/
│   ├── preprocessor.py      # Main preprocessing + prompt building
│   ├── rag_retriever.py     # RAG with FAISS + BM25 + CrossEncoder
│   ├── llm_predictor.py     # HuggingFace endpoint caller
│   ├── postprocessor.py     # Modifier and ICD validation
│   ├── confidence_scorer.py # Confidence scoring
│   ├── pipeline.py          # Full pipeline wrapper
│   └── report_section_extractor.py  # Extract sections from reports
├── tests/
│   ├── test_5_samples.py
│   ├── test_10_samples.py
│   ├── test_50_samples.py
│   └── test_250_samples.py
├── scripts/
│   ├── run_production_batch.py  # Production batch processing
│   └── ...
├── indices/
│   ├── nyxmed_71k_bm25.pkl      # BM25 index (86 MB)
│   ├── nyxmed_71k_faiss.index   # FAISS index (210 MB)
│   └── nyxmed_71k_metadata.pkl  # Metadata (166 MB)
├── config.py
└── requirements.txt
```

## Usage

### Running Tests

```bash
# On RunPod with GPU
cd /workspace/VFinal4

# 10-sample test
python3 tests/test_10_samples.py

# 50-sample test with reranker
python3 tests/test_50_samples.py --use-reranker
```

### Production Batch

```bash
python3 scripts/run_production_batch.py --input /path/to/records.csv --output results/
```

## HuggingFace Endpoint

- **URL**: `https://cdii8l8gfc0bn4fl.us-east-2.aws.endpoints.huggingface.cloud`
- **Token**: Set `HF_API_TOKEN` environment variable
- **Temperature**: 0.1
- **Max Tokens**: 100

## Performance

| Metric | Value |
|--------|-------|
| CPT Accuracy | 90% |
| GT in RAG | 100% |
| Few-shot Examples | 3 per prompt |
| Index Size | 71,654 documents |

## Dependencies

- Python 3.11+
- PyTorch 2.4+
- sentence-transformers
- faiss-cpu
- rank_bm25
- requests

## Notes

- CrossEncoder reranking is enabled by default (`rag_use_reranker=True`)
- Section extractor uses compact format matching the indexed `parsed_compact` field
- Prompt matches `PRODUCTION_PROMPT_BASELINE.md` format exactly
