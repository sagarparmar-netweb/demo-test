#!/usr/bin/env python3
"""
VFinal3 - 50 Sample Validation Test with CrossEncoder Reranking
================================================================

Quick validation before production run.
Tests the exact same pipeline that will be used in production.

Usage:
    export HF_API_TOKEN="hf_xxxxx"
    python scripts/test_50_with_reranker.py
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict

import pandas as pd
import numpy as np

# Add paths
SCRIPT_DIR = Path(__file__).parent
VFINAL_DIR = SCRIPT_DIR.parent
PROJECT_DIR = VFINAL_DIR.parent
sys.path.insert(0, str(VFINAL_DIR))
sys.path.insert(0, str(VFINAL_DIR / "src"))

# Configuration
NUM_SAMPLES = 50
RANDOM_STATE = 123
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")


@dataclass
class TestResult:
    sample_num: int
    gt_cpt: str
    pred_cpt: str
    cpt_correct: bool
    gt_modifier: str
    pred_modifier: str
    modifier_correct: bool
    gt_icd: str
    pred_icd: str
    icd_recall: float
    rag_match_score: float
    latency_ms: float
    error: str = ""


def load_models():
    """Load all models once (with timing)."""
    import pickle
    import faiss
    from sentence_transformers import SentenceTransformer, CrossEncoder
    import requests
    
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)
    
    total_start = time.time()
    
    # Indices
    start = time.time()
    index_dir = VFINAL_DIR / "indices"
    with open(index_dir / "nyxmed_71k_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    faiss_index = faiss.read_index(str(index_dir / "nyxmed_71k_faiss.index"))
    with open(index_dir / "nyxmed_71k_metadata.pkl", "rb") as f:
        data = pickle.load(f)
        documents = data["documents"]
        metadata = data["metadata"]
    print(f"✓ Indices: {time.time()-start:.1f}s ({len(documents):,} docs)")
    
    # Embedding model
    start = time.time()
    embed_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")
    print(f"✓ Embedding model: {time.time()-start:.1f}s")
    
    # CrossEncoder
    start = time.time()
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    print(f"✓ CrossEncoder: {time.time()-start:.1f}s")
    
    # Postprocessor
    from postprocessor import Postprocessor
    postprocessor = Postprocessor(bill_type='P')
    
    total = time.time() - total_start
    print(f"\n✅ All models loaded in {total:.1f}s")
    print("=" * 60)
    
    return {
        'bm25': bm25,
        'faiss_index': faiss_index,
        'documents': documents,
        'metadata': metadata,
        'embed_model': embed_model,
        'reranker': reranker,
        'postprocessor': postprocessor,
        'requests': requests
    }


def search_with_rerank(query: str, models: dict, top_k: int = 15) -> tuple:
    """Hybrid search with CrossEncoder reranking. Returns (hits, top_score)."""
    import faiss as faiss_module
    
    # BM25
    tokenized = query.lower().split()
    bm25_scores = models['bm25'].get_scores(tokenized)
    bm25_indices = np.argsort(bm25_scores)[::-1][:100]
    
    # Vector
    query_vec = models['embed_model'].encode([query])
    faiss_module.normalize_L2(query_vec)
    _, vector_indices = models['faiss_index'].search(query_vec, 100)
    vector_indices = vector_indices[0]
    
    # Combine
    candidates = list(set(bm25_indices) | set(vector_indices))
    
    # Filter exact matches
    query_norm = query[:500].lower().strip()
    candidates = [idx for idx in candidates if models['documents'][idx][:500].lower().strip() != query_norm]
    
    # Rerank
    if len(candidates) > 0:
        pairs = [[query, models['documents'][idx]] for idx in candidates[:50]]
        scores = models['reranker'].predict(pairs)
        sorted_idx = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank_idx in sorted_idx:
            orig_idx = candidates[rank_idx]
            results.append({
                "score": float(scores[rank_idx]),
                "content": models['documents'][orig_idx],
                "metadata": models['metadata'][orig_idx]
            })
        
        top_score = float(scores[sorted_idx[0]]) if len(sorted_idx) > 0 else 0.0
        return results, top_score
    
    return [], 0.0


def build_prompt(report: str, hits: List[Dict]) -> str:
    """Build LLM prompt with few-shot examples."""
    from collections import Counter
    
    cpt_codes, icd_codes = [], []
    for hit in hits:
        meta = hit['metadata']
        cpt = str(meta.get('gt_cpt', ''))
        if cpt.isdigit() and len(cpt) == 5:
            cpt_codes.append(cpt)
        icd_str = str(meta.get('gt_icd', ''))
        if icd_str and icd_str not in ['nan', 'None']:
            for code in icd_str.split(','):
                if code.strip():
                    icd_codes.append(code.strip())
    
    cpt_candidates = [c for c, _ in Counter(cpt_codes).most_common(10)]
    icd_candidates = [c for c, _ in Counter(icd_codes).most_common(30)]
    
    examples = []
    for hit in hits[:3]:
        meta = hit['metadata']
        cpt = str(meta.get('gt_cpt', ''))
        icd = str(meta.get('gt_icd', ''))
        modifier = str(meta.get('modifier', ''))
        
        if not cpt.isdigit() or len(cpt) != 5:
            continue
        
        parsed = meta.get('parsed_compact', '')
        report_content = str(parsed)[:700] if parsed and str(parsed) not in ['nan', 'None'] else hit['content'][:400]
        
        output = f"{cpt}, {modifier}, {icd}" if modifier and modifier not in ['nan', 'None'] else f"{cpt}, {icd}"
        examples.append(f"Report: {report_content}\nOutput: {output}")
    
    examples_str = '\n---\n'.join(examples)
    
    return f"""Study these similar cases:

{examples_str}

---

Code this report:
{report[:2500]}

SIMILAR CASES CPT CODES (most likely):
{chr(10).join(cpt_candidates)}

SIMILAR CASES ICD CODES (most likely):
{chr(10).join(icd_candidates)}

Output format: CPT_CODE, MODIFIER (if applicable), ICD_CODE1, ICD_CODE2, ...
Output:"""


def call_hf_endpoint(prompt: str, requests_module) -> str:
    """Call HuggingFace endpoint."""
    response = requests_module.post(
        "https://cdii8l8gfc0bn4fl.us-east-2.aws.endpoints.huggingface.cloud/v1/chat/completions",
        headers={"Authorization": f"Bearer {HF_API_TOKEN}", "Content-Type": "application/json"},
        json={
            "messages": [
                {"role": "system", "content": "You are an expert radiology coder specializing in ICD-10 and CPT coding for radiology reports."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,
            "temperature": 0.1
        },
        timeout=120
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"API Error: {response.status_code}")


def parse_response(response: str) -> tuple:
    """Parse LLM response."""
    parts = [p.strip() for p in response.replace('\n', ',').split(',')]
    cpt_code, modifier, icd_codes = "", "", []
    valid_modifiers = {'26', 'TC', 'LT', 'RT', '50', '76', '77'}
    
    for i, part in enumerate(parts):
        if not part.strip():
            continue
        if i == 0:
            cpt_code = part
        elif i == 1:
            if part in valid_modifiers or any(m in part for m in valid_modifiers):
                modifier = part
            else:
                icd_codes.append(part)
        else:
            if part and len(part) >= 3:
                icd_codes.append(part)
    
    return cpt_code, modifier, icd_codes


def calculate_icd_recall(gt_icd: str, pred_icd: List[str]) -> float:
    """Calculate ICD recall."""
    if not gt_icd or gt_icd == 'nan':
        return 1.0
    
    gt_set = set(gt_icd.replace(' ', '').split(','))
    pred_set = set(','.join(pred_icd).replace(' ', '').split(','))
    
    if not gt_set:
        return 1.0
    
    return len(gt_set & pred_set) / len(gt_set)


def main():
    if not HF_API_TOKEN:
        print("❌ ERROR: HF_API_TOKEN not set!")
        print("   Run: export HF_API_TOKEN='your_token'")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("VFINAL3 - 50 SAMPLE VALIDATION TEST (WITH CROSSENCODER)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load models
    models = load_models()
    
    # Load test data
    data_path = PROJECT_DIR / "data" / "consolidated_cleaned.csv"
    if not data_path.exists():
        # Try alternative paths
        for alt in [PROJECT_DIR / "Training_Data_True_Source_20251231" / "Raw_Data_Consolidated.csv"]:
            if alt.exists():
                data_path = alt
                break
    
    print(f"\n📂 Loading test data: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filter valid records
    valid = df[
        df['Report'].notna() & 
        df['Procedure'].notna() & 
        (df['Report'].str.len() > 300) &
        (df['Procedure'].astype(str).str.match(r'^[0-9]{5}$'))
    ]
    print(f"   Valid records: {len(valid):,}")
    
    # Sample
    samples = valid.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
    
    print(f"\n🚀 Running {NUM_SAMPLES} tests...\n")
    
    results = []
    cpt_correct = 0
    mod_correct = 0
    icd_recalls = []
    
    start_time = time.time()
    
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        report = str(row['Report'])
        gt_cpt = str(row['Procedure'])
        gt_icd = str(row.get('ICD10 - Diagnosis', ''))
        gt_modifier = str(row.get('Modifier', '')).replace('nan', '').replace('.0', '').strip()
        
        sample_start = time.time()
        
        try:
            # 1. RAG search with reranking
            hits, top_score = search_with_rerank(report, models)
            
            # 2. Build prompt
            prompt = build_prompt(report, hits)
            
            # 3. Call LLM
            response = call_hf_endpoint(prompt, models['requests'])
            
            # 4. Parse
            pred_cpt, pred_modifier, pred_icds = parse_response(response)
            
            # Apply disambiguation
            from preprocessor import apply_cpt_disambiguation
            pred_cpt = apply_cpt_disambiguation(pred_cpt, report, "")
            
            # Add modifier 26 if not interventional
            if not pred_modifier and pred_cpt not in models['postprocessor'].INTERVENTIONAL_CPT_CODES:
                pred_modifier = "26"
            
            latency = (time.time() - sample_start) * 1000
            
            # Calculate accuracy
            cpt_match = pred_cpt == gt_cpt
            if cpt_match:
                cpt_correct += 1
            
            is_interventional = gt_cpt in models['postprocessor'].INTERVENTIONAL_CPT_CODES
            if is_interventional:
                mod_match = '26' not in pred_modifier
            else:
                mod_match = '26' in pred_modifier
            if mod_match:
                mod_correct += 1
            
            icd_recall = calculate_icd_recall(gt_icd, pred_icds)
            icd_recalls.append(icd_recall)
            
            results.append(TestResult(
                sample_num=i,
                gt_cpt=gt_cpt,
                pred_cpt=pred_cpt,
                cpt_correct=cpt_match,
                gt_modifier=gt_modifier,
                pred_modifier=pred_modifier,
                modifier_correct=mod_match,
                gt_icd=gt_icd,
                pred_icd=', '.join(pred_icds),
                icd_recall=icd_recall,
                rag_match_score=top_score,
                latency_ms=latency
            ))
            
        except Exception as e:
            results.append(TestResult(
                sample_num=i, gt_cpt=gt_cpt, pred_cpt="ERROR", cpt_correct=False,
                gt_modifier=gt_modifier, pred_modifier="", modifier_correct=False,
                gt_icd=gt_icd, pred_icd="", icd_recall=0.0,
                rag_match_score=0.0, latency_ms=0.0, error=str(e)
            ))
        
        # Progress
        if i % 10 == 0 or i == NUM_SAMPLES:
            elapsed = time.time() - start_time
            rate = i / elapsed
            eta = (NUM_SAMPLES - i) / rate if rate > 0 else 0
            cpt_acc = cpt_correct / i
            mod_acc = mod_correct / i
            avg_icd = sum(icd_recalls) / len(icd_recalls) if icd_recalls else 0
            
            print(f"[{i:2d}/{NUM_SAMPLES}] CPT: {cpt_acc:.1%} | MOD: {mod_acc:.1%} | ICD: {avg_icd:.1%} | ETA: {eta:.0f}s")
    
    # Summary
    total_time = time.time() - start_time
    cpt_acc = cpt_correct / NUM_SAMPLES
    mod_acc = mod_correct / NUM_SAMPLES
    avg_icd = sum(icd_recalls) / len(icd_recalls)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"CPT Accuracy:     {cpt_acc:.1%} ({cpt_correct}/{NUM_SAMPLES})")
    print(f"Modifier Correct: {mod_acc:.1%} ({mod_correct}/{NUM_SAMPLES})")
    print(f"ICD Recall:       {avg_icd:.1%}")
    print(f"Total Time:       {total_time:.1f}s ({total_time/NUM_SAMPLES:.1f}s/sample)")
    
    # Show failures
    failures = [r for r in results if not r.cpt_correct]
    if failures:
        print(f"\n❌ CPT FAILURES ({len(failures)}):")
        for f in failures[:10]:
            print(f"   #{f.sample_num}: GT={f.gt_cpt} → Pred={f.pred_cpt}")
    
    # Validation
    print("\n" + "=" * 70)
    if cpt_acc >= 0.88 and mod_acc >= 0.95:
        print("✅ VALIDATION PASSED - Ready for production!")
        return 0
    else:
        print("⚠️ VALIDATION NEEDS REVIEW")
        if cpt_acc < 0.88:
            print(f"   - CPT accuracy {cpt_acc:.1%} < 88% threshold")
        if mod_acc < 0.95:
            print(f"   - Modifier accuracy {mod_acc:.1%} < 95% threshold")
        return 1


if __name__ == "__main__":
    sys.exit(main())
