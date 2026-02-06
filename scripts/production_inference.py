#!/usr/bin/env python3
"""
NYXMed Production Inference Script
===================================

This script implements the EXACT prompt format that achieves 100% CPT accuracy.
DO NOT modify the prompt template without re-validation.

Usage:
    # Single prediction
    python scripts/production_inference.py --report "path/to/report.txt"
    
    # Batch prediction
    python scripts/production_inference.py --csv "path/to/batch.csv" --output "results.csv"

Reference: PRODUCTION_PROMPT_BASELINE.md
"""

import os
import sys
import json
import pickle
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import re

# Add paths
SCRIPT_DIR = Path(__file__).parent
VFINAL_DIR = SCRIPT_DIR.parent
PROJECT_DIR = VFINAL_DIR.parent

sys.path.insert(0, str(VFINAL_DIR))
sys.path.insert(0, str(PROJECT_DIR))

# =============================================================================
# CONFIGURATION - DO NOT CHANGE
# =============================================================================

# HuggingFace Endpoint
HF_ENDPOINT_URL = os.getenv(
    "HF_ENDPOINT_URL",
    "https://cdii8l8gfc0bn4fl.us-east-2.aws.endpoints.huggingface.cloud/v1/chat/completions"
)
HF_TOKEN = os.getenv("HF_TOKEN", "")

# System prompt - EXACT match to training
SYSTEM_PROMPT = "You are an expert radiology coder specializing in ICD-10 and CPT coding for radiology reports."

# RAG Configuration
INDEX_DIR = VFINAL_DIR / "indices"
INDEX_PREFIX = "nyxmed_71k"

# LLM Parameters
TEMPERATURE = 0.1
MAX_TOKENS = 100


# =============================================================================
# PROMPT TEMPLATE - DO NOT CHANGE
# =============================================================================

PROMPT_TEMPLATE = """Study these similar cases:

{few_shot_examples}

---

Code this report:
{report_header}

{report_body}

SIMILAR CASES CPT CODES (most likely):
{cpt_candidates}

Available ICD codes (from similar cases):
{icd_candidates}

Rules:
- Determine the CPT code from the procedure/exam description
- Add appropriate modifiers (26=professional, LT=left, RT=right, 50=bilateral)
- Code the clinical indication (R/Z/S code)
- Code confirmed findings from impression
- Use laterality info to select correct modifier (LT/RT/50)

Output format: CPT, MODIFIER, ICD1, ICD2, ...
Example: 71046, 26, R91.8, J18.9
Example: 73030, 26 LT, M25.511, S43.401A"""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PredictionResult:
    """Result from production inference."""
    cpt_code: str
    modifier: str
    icd_codes: List[str]
    raw_response: str
    confidence: float = 1.0
    
    def to_string(self) -> str:
        """Format as output string."""
        parts = [self.cpt_code]
        if self.modifier:
            parts.append(self.modifier)
        parts.extend(self.icd_codes)
        return ", ".join(parts)


# =============================================================================
# RAG RETRIEVER
# =============================================================================

class ProductionRAG:
    """RAG retriever for production inference."""
    
    def __init__(self):
        self.loaded = False
        self.bm25 = None
        self.faiss_index = None
        self.documents = None
        self.metadata = None
        self.embed_model = None
        
        self._load()
    
    def _load(self):
        """Load RAG indices."""
        bm25_path = INDEX_DIR / f"{INDEX_PREFIX}_bm25.pkl"
        faiss_path = INDEX_DIR / f"{INDEX_PREFIX}_faiss.index"
        meta_path = INDEX_DIR / f"{INDEX_PREFIX}_metadata.pkl"
        
        if not all(p.exists() for p in [bm25_path, faiss_path, meta_path]):
            print(f"⚠ RAG indices not found at {INDEX_DIR}")
            print(f"  Expected prefix: {INDEX_PREFIX}")
            return
        
        print("Loading RAG indices...")
        
        # Load BM25
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        
        # Load FAISS
        import faiss
        self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load metadata
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
        
        # Load embedding model
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m")
        
        self.loaded = True
        print(f"  ✓ Loaded {len(self.documents):,} documents")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar cases."""
        if not self.loaded:
            return []
        
        import faiss
        
        # BM25 search
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        bm25_top = np.argsort(bm25_scores)[-top_k*2:][::-1]
        
        # FAISS search
        query_vec = self.embed_model.encode([query])
        faiss.normalize_L2(query_vec)
        _, faiss_top = self.faiss_index.search(query_vec.astype('float32'), top_k*2)
        
        # Combine
        combined = list(set(bm25_top.tolist() + faiss_top[0].tolist()))
        
        # Sort by BM25 score
        scored = [(idx, bm25_scores[idx]) for idx in combined if idx < len(self.metadata)]
        scored.sort(key=lambda x: -x[1])
        
        results = []
        for idx, score in scored[:top_k]:
            results.append({
                "score": float(score),
                "content": self.documents[idx],
                "metadata": self.metadata[idx]
            })
        
        return results
    
    def get_candidates_and_examples(self, report: str, top_k: int = 5) -> Tuple[List[str], List[str], str]:
        """
        Get CPT candidates, ICD candidates, and formatted few-shot examples.
        
        This is the key function that builds the prompt components.
        """
        if not self.loaded:
            return [], [], ""
        
        hits = self.search(report, top_k=top_k)
        
        # Extract CPT and ICD candidates
        cpt_codes = []
        icd_codes = []
        
        for hit in hits:
            meta = hit['metadata']
            
            cpt = meta.get('gt_cpt') or meta.get('cpt_code')
            if cpt and str(cpt) not in ['nan', 'None', '']:
                cpt_codes.append(str(cpt))
            
            icd_str = meta.get('gt_icd') or meta.get('icd_codes') or ''
            for code in str(icd_str).split(','):
                code = code.strip()
                if code and code not in ['nan', 'None']:
                    icd_codes.append(code)
        
        # Unique, ordered by frequency
        from collections import Counter
        unique_cpts = [c for c, _ in Counter(cpt_codes).most_common()]
        unique_icds = [c for c, _ in Counter(icd_codes).most_common(30)]
        
        # Format few-shot examples - MUST match training format
        examples = []
        for hit in hits[:3]:
            meta = hit['metadata']
            
            # Build report content using parsed fields
            parts = []
            
            indication = meta.get('indication', '')
            if indication and str(indication) not in ['nan', 'None', '']:
                parts.append(f"INDICATION: {indication}")
            
            laterality = meta.get('laterality', '')
            if laterality and str(laterality) not in ['nan', 'None', '']:
                parts.append(f"LATERALITY: {laterality}")
            
            impression = meta.get('impression', '')
            if impression and str(impression) not in ['nan', 'None', '']:
                parts.append(f"IMPRESSION: {impression}")
            
            # Fallback to parsed_compact or document content
            if not parts:
                content = meta.get('parsed_compact', hit['content'][:400])
                parts.append(content)
            
            report_content = '\n'.join(parts)[:600]
            
            # Build output line
            cpt = meta.get('gt_cpt') or meta.get('cpt_code') or ''
            icd = meta.get('gt_icd') or meta.get('icd_codes') or ''
            modifier = meta.get('modifier', '')
            
            if modifier and str(modifier) not in ['nan', 'None', '']:
                output = f"{cpt}, {modifier}, {icd}"
            else:
                output = f"{cpt}, {icd}"
            
            examples.append(f"Report: {report_content}\nOutput: {output}")
        
        examples_str = "\n---\n".join(examples)
        
        return unique_cpts, unique_icds, examples_str


# =============================================================================
# PRODUCTION PREDICTOR
# =============================================================================

class ProductionPredictor:
    """
    Production predictor using the exact prompt format from training.
    
    DO NOT modify the prompt building logic.
    """
    
    def __init__(self, hf_token: str = None):
        self.hf_token = hf_token or HF_TOKEN
        if not self.hf_token:
            print("⚠ HF_TOKEN not set. Set via environment variable or pass to __init__")
        
        self.rag = ProductionRAG()
    
    def _extract_report_header(self, report: str) -> Tuple[str, str]:
        """Extract header fields and body from report."""
        # Try to extract structured header
        header_fields = []
        
        # MRN
        mrn_match = re.search(r'MRN[:\s]+([^\n]+)', report, re.I)
        if mrn_match:
            header_fields.append(f"MRN: {mrn_match.group(1).strip()}")
        
        # Order No
        order_match = re.search(r'Order\s*No[\.:]?\s*([^\n]+)', report, re.I)
        if order_match:
            header_fields.append(f"Order No.: {order_match.group(1).strip()}")
        
        # Exam Date
        date_match = re.search(r'Exam\s*Date[:\s]+([^\n]+)', report, re.I)
        if date_match:
            header_fields.append(f"Exam Date: {date_match.group(1).strip()}")
        
        # Exam Description - CRITICAL for CPT
        exam_match = re.search(r'Exam\s*Description[:\s]+([^\n]+)', report, re.I)
        if not exam_match:
            exam_match = re.search(r'EXAM[:\s]+([^\n]+)', report, re.I)
        if exam_match:
            header_fields.append(f"Exam Description: {exam_match.group(1).strip()}")
        
        # Bill Type
        bill_match = re.search(r'Bill\s*Type[:\s]+([^\n]+)', report, re.I)
        if bill_match:
            header_fields.append(f"Bill Type: {bill_match.group(1).strip()}")
        
        # Reason for Exam
        reason_match = re.search(r'Reason\s*for\s*Exam[:\s]+([^\n]+)', report, re.I)
        if reason_match:
            header_fields.append(f"\nReason for Exam: {reason_match.group(1).strip()}")
        
        header = '\n'.join(header_fields) if header_fields else ""
        
        # Body is everything after FINDINGS or the full report
        body_match = re.search(r'(FINDINGS.*)', report, re.I | re.DOTALL)
        if body_match:
            body = body_match.group(1)[:2500]
        else:
            body = report[:2500]
        
        return header, body
    
    def _build_prompt(
        self,
        report: str,
        cpt_candidates: List[str],
        icd_candidates: List[str],
        examples: str
    ) -> str:
        """
        Build the production prompt.
        
        Uses the EXACT format from training data.
        """
        header, body = self._extract_report_header(report)
        
        # Format candidates
        cpt_str = '\n'.join(cpt_candidates[:10])
        icd_str = '\n'.join(icd_candidates[:30])
        
        prompt = PROMPT_TEMPLATE.format(
            few_shot_examples=examples,
            report_header=header,
            report_body=body,
            cpt_candidates=cpt_str,
            icd_candidates=icd_str
        )
        
        return prompt
    
    def _call_endpoint(self, prompt: str) -> str:
        """Call HuggingFace endpoint."""
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": MAX_TOKENS,
            "temperature": TEMPERATURE
        }
        
        response = requests.post(
            HF_ENDPOINT_URL,
            headers=headers,
            json=payload,
            timeout=120
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content'].strip()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def _parse_response(self, response: str) -> PredictionResult:
        """Parse LLM response into structured result."""
        parts = [p.strip() for p in response.replace('\n', ',').split(',')]
        
        cpt_code = ""
        modifier = ""
        icd_codes = []
        
        valid_modifiers = {'26', 'TC', 'LT', 'RT', '50', '76', '77', 'XU', 'XP', 'XE', 'PI', 'JW', 'JZ'}
        
        for i, part in enumerate(parts):
            if i == 0:
                # First part is CPT
                cpt_code = part
            elif i == 1:
                # Check if modifier or ICD
                if part in valid_modifiers or any(m in part for m in valid_modifiers):
                    modifier = part
                else:
                    icd_codes.append(part)
            else:
                # Rest are ICD codes
                if part and len(part) >= 3:
                    icd_codes.append(part)
        
        return PredictionResult(
            cpt_code=cpt_code,
            modifier=modifier,
            icd_codes=icd_codes,
            raw_response=response
        )
    
    def predict(self, report: str) -> PredictionResult:
        """
        Make a production prediction.
        
        This uses the exact prompt format from training.
        """
        # 1. Get RAG candidates and examples
        cpt_candidates, icd_candidates, examples = self.rag.get_candidates_and_examples(report)
        
        if not examples:
            print("⚠ No RAG examples found - prediction quality may be degraded")
        
        # 2. Build prompt
        prompt = self._build_prompt(report, cpt_candidates, icd_candidates, examples)
        
        # 3. Call endpoint
        response = self._call_endpoint(prompt)
        
        # 4. Parse response
        result = self._parse_response(response)
        
        return result
    
    def predict_batch(self, reports: List[str], show_progress: bool = True) -> List[PredictionResult]:
        """Predict on multiple reports."""
        results = []
        
        iterator = tqdm(reports, desc="Predicting") if show_progress else reports
        
        for report in iterator:
            try:
                result = self.predict(report)
                results.append(result)
            except Exception as e:
                print(f"Error: {e}")
                results.append(PredictionResult(
                    cpt_code="ERROR",
                    modifier="",
                    icd_codes=[],
                    raw_response=str(e)
                ))
        
        return results


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NYXMed Production Inference")
    parser.add_argument("--report", type=str, help="Path to report text file")
    parser.add_argument("--csv", type=str, help="Path to CSV with reports")
    parser.add_argument("--output", type=str, help="Output CSV path")
    parser.add_argument("--report-column", type=str, default="Report", help="Column name for reports in CSV")
    parser.add_argument("--token", type=str, help="HuggingFace API token")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ProductionPredictor(hf_token=args.token)
    
    if args.report:
        # Single report
        with open(args.report, 'r') as f:
            report_text = f.read()
        
        result = predictor.predict(report_text)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(f"CPT:      {result.cpt_code}")
        print(f"Modifier: {result.modifier}")
        print(f"ICD:      {', '.join(result.icd_codes)}")
        print(f"\nOutput:   {result.to_string()}")
        print("=" * 60)
    
    elif args.csv:
        # Batch processing
        df = pd.read_csv(args.csv)
        
        if args.report_column not in df.columns:
            print(f"Error: Column '{args.report_column}' not found in CSV")
            print(f"Available columns: {list(df.columns)}")
            return
        
        reports = df[args.report_column].tolist()
        results = predictor.predict_batch(reports)
        
        # Add results to dataframe
        df['Predicted_CPT'] = [r.cpt_code for r in results]
        df['Predicted_Modifier'] = [r.modifier for r in results]
        df['Predicted_ICD'] = [','.join(r.icd_codes) for r in results]
        df['Predicted_Output'] = [r.to_string() for r in results]
        
        # Save
        output_path = args.output or args.csv.replace('.csv', '_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
