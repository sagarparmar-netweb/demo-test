#!/usr/bin/env python3
"""
NYXMed Production Batch Runner
==============================

Uses the SAME pipeline as test_50_samples.py:
  Preprocessor → LLMPredictor → Postprocessor

Outputs frontend-compatible columns with real-time CSV updates.

Usage:
    export HF_API_TOKEN="hf_xxxxx"
    python scripts/run_production_batch.py \
        --input /path/to/input.csv \
        --output /path/to/output.csv
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np
from tqdm import tqdm

# Add paths
SCRIPT_DIR = Path(__file__).parent
VFINAL_DIR = SCRIPT_DIR.parent
PROJECT_DIR = VFINAL_DIR.parent
sys.path.insert(0, str(VFINAL_DIR))
sys.path.insert(0, str(VFINAL_DIR / "src"))

# Configuration
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
SAVE_EVERY_N = 25
MAX_RETRIES = 3
RETRY_DELAY = 5


def find_training_data():
    """Find training data for Preprocessor initialization."""
    candidates = [
        PROJECT_DIR / 'data' / 'consolidated_cleaned.csv',
        PROJECT_DIR / 'Training_Data_True_Source_20251231' / 'Raw_Data_Consolidated.csv',
        PROJECT_DIR / 'archive' / 'old_repository_files' / 'Training_Data_True_Source_20251231' / 'Raw_Data_Consolidated.csv',
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


class ProductionBatchRunner:
    """Production batch runner using VFinal3 pipeline components."""
    
    def __init__(self, verbose: bool = True):
        # CrossEncoder reranking is REQUIRED (not optional).
        self.use_reranker = True
        self.verbose = verbose
        
        if not HF_API_TOKEN:
            raise ValueError("HF_API_TOKEN environment variable not set!")
        
        self._load_pipeline()
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _load_pipeline(self):
        """Load VFinal3 pipeline components."""
        from src.preprocessor import Preprocessor
        from src.llm_predictor import LLMPredictor
        from src.postprocessor import Postprocessor
        
        total_start = time.time()
        
        self._log("\n" + "=" * 60)
        self._log("LOADING VFINAL3 PIPELINE")
        self._log("=" * 60)
        self._log("CrossEncoder Reranking: ENABLED (required)")
        
        # Load training data for preprocessor
        start = time.time()
        training_data_path = find_training_data()
        if training_data_path:
            self._log(f"\nLoading training data: {training_data_path.name}")
            self.training_df = pd.read_csv(training_data_path, low_memory=False)
            self._log(f"✓ Training data: {time.time()-start:.1f}s ({len(self.training_df):,} records)")
        else:
            self._log("⚠ Training data not found, using RAG indices only")
            self.training_df = None
        
        # Initialize Preprocessor (includes RAG)
        start = time.time()
        self.preprocessor = Preprocessor(
            df=self.training_df, 
            use_rag=True, 
            rag_use_reranker=self.use_reranker
        )
        self._log(f"✓ Preprocessor + RAG: {time.time()-start:.1f}s")
        
        # Initialize LLM Predictor
        start = time.time()
        self.llm = LLMPredictor(backend="huggingface")
        self._log(f"✓ LLM Predictor: {time.time()-start:.1f}s")
        
        # Initialize Postprocessor
        start = time.time()
        self.postprocessor = Postprocessor(bill_type='P')
        self._log(f"✓ Postprocessor: {time.time()-start:.1f}s")
        
        total_time = time.time() - total_start
        self._log(f"\n✅ Pipeline loaded in {total_time:.1f}s")
        self._log("=" * 60 + "\n")
    
    def _get_rag_info(self, prep) -> tuple:
        """Extract RAG info from preprocessed data for frontend."""
        # Get CPT candidates
        cpt_candidates = ""
        icd_candidates = ""
        closest_samples = "[]"
        rag_score = 0.0
        
        if hasattr(prep, 'cpt_candidates') and prep.cpt_candidates:
            cpt_candidates = ', '.join([str(c) for c in prep.cpt_candidates[:5]])
        
        if hasattr(prep, 'icd_candidates') and prep.icd_candidates:
            icd_candidates = ', '.join([str(c) for c, _ in prep.icd_candidates[:10]])
        
        # Try to get RAG hits from preprocessor
        if hasattr(self.preprocessor, 'rag_retriever') and self.preprocessor.rag_retriever:
            if hasattr(self.preprocessor, '_last_rag_hits'):
                hits = self.preprocessor._last_rag_hits
                if hits:
                    rag_score = hits[0].get('score', 0) if isinstance(hits[0], dict) else 0
                    closest = []
                    for hit in hits[:3]:
                        if isinstance(hit, dict):
                            meta = hit.get('metadata', {})
                            closest.append({
                                "score": round(hit.get('score', 0), 2),
                                "cpt": str(meta.get('gt_cpt', '')),
                                "modifier": str(meta.get('modifier', '')).replace('nan', ''),
                                "icd": str(meta.get('gt_icd', ''))[:50]
                            })
                    closest_samples = json.dumps(closest)
        
        return cpt_candidates, icd_candidates, closest_samples, rag_score
    
    def _calculate_confidence(self, prep, final) -> tuple:
        """Calculate confidence scores."""
        # Base confidence on whether CPT was extracted vs predicted
        if hasattr(prep, 'extracted_cpt') and prep.extracted_cpt:
            cpt_conf = 0.95  # High confidence if CPT was extracted from report
        else:
            cpt_conf = 0.85  # Good confidence from LLM prediction
        
        icd_conf = 0.80
        mod_conf = 0.95
        overall = (cpt_conf + icd_conf + mod_conf) / 3
        
        return round(overall, 2), round(cpt_conf, 2), round(icd_conf, 2), round(mod_conf, 2)
    
    def predict_single(self, report: str, exam_desc: str = "") -> dict:
        """Process a single report through the full pipeline."""
        start = time.time()
        
        try:
            # Step 1: Preprocess
            prep = self.preprocessor.preprocess(report)
            
            # Step 2: LLM Prediction
            llm_result = self.llm.predict(
                prompt=prep.prompt,
                extracted_cpt=prep.extracted_cpt,
                valid_icd_codes=set(c for c, _ in prep.icd_candidates) if prep.icd_candidates else set(),
                report_text=report,
                exam_desc=exam_desc
            )
            
            # Step 3: Postprocess
            final = self.postprocessor.postprocess(
                cpt_code=llm_result.cpt_code,
                icd_codes=llm_result.icd_codes,
                laterality=prep.laterality,
                cpt_was_extracted=bool(prep.extracted_cpt),
                valid_icd_codes=None,
                llm_suggested_modifiers=llm_result.modifiers
            )
            
            # Get RAG info for frontend
            cpt_cands, icd_cands, closest, rag_score = self._get_rag_info(prep)
            
            # Calculate confidence
            overall, cpt_conf, icd_conf, mod_conf = self._calculate_confidence(prep, final)
            
            return {
                'predicted_cpt': final.procedure,
                'predicted_modifier': final.modifier,
                'predicted_icd': final.icd10_diagnosis,
                'confidence_score': overall,
                'cpt_confidence': cpt_conf,
                'icd_confidence': icd_conf,
                'modifier_confidence': mod_conf,
                'rag_match_score': rag_score,
                'rag_cpt_candidates': cpt_cands,
                'rag_icd_candidates': icd_cands,
                'closest_samples': closest,
                'prediction_latency_ms': (time.time() - start) * 1000,
                'prediction_error': '',
                'status': 'processed'
            }
            
        except Exception as e:
            return {
                'predicted_cpt': 'ERROR',
                'predicted_modifier': '',
                'predicted_icd': '',
                'confidence_score': 0,
                'cpt_confidence': 0,
                'icd_confidence': 0,
                'modifier_confidence': 0,
                'rag_match_score': 0,
                'rag_cpt_candidates': '',
                'rag_icd_candidates': '',
                'closest_samples': '[]',
                'prediction_latency_ms': (time.time() - start) * 1000,
                'prediction_error': str(e),
                'status': 'error'
            }
    
    def run_batch(self, input_csv: str, output_csv: str, report_column: str = "report", resume: bool = True) -> pd.DataFrame:
        """Run batch prediction with frontend-compatible output."""
        self._log(f"\n📂 Loading input: {input_csv}")
        df = pd.read_csv(input_csv, low_memory=False)
        total_records = len(df)
        self._log(f"   Total records: {total_records:,}")
        
        output_path = Path(output_csv)
        start_idx = 0
        
        # Resume logic
        if resume and output_path.exists():
            existing_df = pd.read_csv(output_csv, low_memory=False)
            if 'status' in existing_df.columns:
                completed = (existing_df['status'] == 'processed').sum()
                if completed > 0:
                    self._log(f"   Resuming from record {completed}")
                    start_idx = completed
                    df = existing_df
        
        # Initialize columns
        new_columns = [
            'predicted_cpt', 'predicted_icd', 'predicted_modifier',
            'confidence_score', 'cpt_confidence', 'icd_confidence', 'modifier_confidence',
            'rag_match_score', 'rag_cpt_candidates', 'rag_icd_candidates',
            'closest_samples', 'prediction_latency_ms', 'prediction_error', 'status'
        ]
        for col in new_columns:
            if col not in df.columns:
                df[col] = None
        
        if 'status' not in df.columns or df['status'].isna().all():
            df['status'] = 'pending'
        
        self._log(f"\n🚀 Processing {total_records - start_idx:,} records...")
        self._log(f"   Saving every {SAVE_EVERY_N} records\n")
        
        start_time = time.time()
        errors = 0
        
        for i in tqdm(range(start_idx, total_records), initial=start_idx, total=total_records):
            report = str(df.iloc[i][report_column])
            exam_desc = str(df.iloc[i].get('Exam Description', '')) if 'Exam Description' in df.columns else ''
            
            if pd.isna(report) or len(report.strip()) < 50:
                df.at[i, 'status'] = 'error'
                df.at[i, 'prediction_error'] = "Report too short"
                errors += 1
                continue
            
            # Predict with retries
            result = None
            for attempt in range(MAX_RETRIES):
                result = self.predict_single(report, exam_desc)
                if result['prediction_error'] == '':
                    break
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
            
            # Store results
            for key, value in result.items():
                df.at[i, key] = value
            
            if result['prediction_error']:
                errors += 1
            
            # Save frequently
            if (i + 1) % SAVE_EVERY_N == 0:
                df.to_csv(output_csv, index=False)
                elapsed = time.time() - start_time
                processed = i + 1 - start_idx
                rate = processed / elapsed if elapsed > 0 else 0
                eta = (total_records - i - 1) / rate if rate > 0 else 0
                self._log(f"\n   💾 {i+1}/{total_records} | {rate:.2f}/sec | ETA: {eta/60:.1f}min")
        
        # Final save
        df.to_csv(output_csv, index=False)
        
        elapsed = time.time() - start_time
        processed_count = (df['status'] == 'processed').sum()
        
        self._log("\n" + "=" * 60)
        self._log("✅ BATCH COMPLETE")
        self._log("=" * 60)
        self._log(f"   Total: {total_records:,} | Processed: {processed_count:,} | Errors: {errors}")
        self._log(f"   Time: {elapsed/60:.1f}min")
        self._log(f"   Output: {output_csv}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="NYXMed Production Batch Runner")
    parser.add_argument("--input", "-i", required=True, help="Input CSV path")
    parser.add_argument("--output", "-o", required=True, help="Output CSV path")
    parser.add_argument("--report-column", default="report", help="Column name for report text")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    
    args = parser.parse_args()
    
    if not HF_API_TOKEN:
        print("❌ ERROR: HF_API_TOKEN not set!")
        print("   Run: export HF_API_TOKEN='your_token'")
        sys.exit(1)
    
    runner = ProductionBatchRunner(verbose=True)
    runner.run_batch(args.input, args.output, args.report_column, resume=not args.no_resume)


if __name__ == "__main__":
    main()
