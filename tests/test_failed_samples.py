#!/usr/bin/env python3
"""
VFinal4 - Test Only Failed Samples

Re-runs ONLY the samples that failed in the previous test run.
Uses the same random_state=123 to get identical samples.

Failed samples from previous run: 6, 8, 23, 27, 29, 45, 46

Usage:
    export HF_API_TOKEN="your_token_here"
    python tests/test_failed_samples.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add VFinal4 to path
VFINAL4_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(VFINAL4_DIR))
sys.path.insert(0, str(VFINAL4_DIR / 'src'))

# Also add project root for data access
PROJECT_DIR = VFINAL4_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.preprocessor import Preprocessor, apply_cpt_disambiguation
from src.llm_predictor import LLMPredictor
from src.postprocessor import Postprocessor

# Configuration
RANDOM_STATE = 123  # Same as original test for reproducibility
NUM_SAMPLES = 50    # Original sample count
OUTPUT_CSV = PROJECT_DIR / 'reports' / 'vfinal4-failed-samples.csv'

# Failed sample indices (1-based from previous test)
FAILED_SAMPLES = [6, 8, 23, 27, 29, 45, 46]


def main():
    # Check HF token
    hf_token = os.environ.get('HF_API_TOKEN')
    if not hf_token:
        print("ERROR: HF_API_TOKEN environment variable not set")
        print("Usage: export HF_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    print("=" * 80)
    print("VFINAL4 - FAILED SAMPLES RE-TEST")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing samples: {FAILED_SAMPLES}")
    print(f"Output CSV: {OUTPUT_CSV}")
    
    # Load data
    data_path = PROJECT_DIR / 'data' / 'consolidated_cleaned.csv'
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filter valid records
    valid = df[
        df['Report'].notna() & 
        df['Procedure'].notna() & 
        (df['Report'].str.len() > 300) &
        (df['Procedure'].astype(str).str.match(r'^[0-9]{5}$'))
    ]
    print(f"Valid records: {len(valid):,}")
    
    # Get the SAME fixed samples as original test
    all_samples = valid.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
    
    # Initialize components
    print("\nInitializing pipeline components...")
    preprocessor = Preprocessor(df=df, use_rag=True, rag_use_reranker=True)
    llm = LLMPredictor(backend="huggingface")
    postprocessor = Postprocessor(bill_type='P')
    print("✓ All components initialized")
    
    print(f"\nRunning {len(FAILED_SAMPLES)} failed samples...\n")
    
    rows = []
    cpt_correct = 0
    fixed_samples = []
    still_failing = []
    
    for sample_idx in FAILED_SAMPLES:
        # Get the specific sample (0-indexed)
        row = all_samples.iloc[sample_idx - 1]
        
        report = str(row['Report'])
        gt_cpt = str(row['Procedure'])
        gt_icd = str(row.get('ICD10 - Diagnosis', ''))
        gt_modifier = str(row.get('Modifier', '')).replace('nan', '').replace('.0', '').strip()
        exam_desc = str(row.get('Exam Description', ''))
        
        is_interventional = gt_cpt in Postprocessor.INTERVENTIONAL_CPT_CODES
        
        # Step 1: Preprocess
        prep = preprocessor.preprocess(report)
        
        # Step 2: LLM Prediction
        llm_result = llm.predict(
            prompt=prep.prompt,
            extracted_cpt=prep.extracted_cpt,
            valid_icd_codes=set(c for c, _ in prep.icd_candidates),
            report_text=report,
            exam_desc=exam_desc
        )
        
        # Step 3: Apply CPT disambiguation (this is where our fixes are)
        disambiguated_cpt = apply_cpt_disambiguation(
            llm_result.cpt_code, 
            report, 
            exam_desc
        )
        
        # Step 4: Postprocess
        final = postprocessor.postprocess(
            cpt_code=disambiguated_cpt,
            icd_codes=llm_result.icd_codes,
            laterality=prep.laterality,
            cpt_was_extracted=bool(prep.extracted_cpt),
            valid_icd_codes=None,
            llm_suggested_modifiers=llm_result.modifiers
        )
        
        # Calculate accuracy
        cpt_match = final.procedure == gt_cpt
        if cpt_match:
            cpt_correct += 1
            fixed_samples.append(sample_idx)
        else:
            still_failing.append(sample_idx)
        
        gt_icd_set = set(gt_icd.replace(' ', '').split(',')) if gt_icd and gt_icd != 'nan' else set()
        pred_icd_set = set(final.icd10_diagnosis.replace(' ', '').split(',')) if final.icd10_diagnosis else set()
        recall = len(gt_icd_set & pred_icd_set) / len(gt_icd_set) if gt_icd_set else 1.0
        
        # Print detailed result
        status = "✅ FIXED" if cpt_match else "❌ STILL FAILING"
        print(f"\nSample {sample_idx}: {status}")
        print(f"   Exam: {exam_desc[:60]}")
        print(f"   GT CPT:   {gt_cpt}")
        print(f"   LLM Raw:  {llm_result.cpt_code}")
        print(f"   Disambig: {disambiguated_cpt}")
        print(f"   Final:    {final.procedure}")
        
        rows.append({
            'sample': sample_idx,
            'exam_description': exam_desc[:60],
            'gt_cpt': gt_cpt,
            'llm_cpt': llm_result.cpt_code,
            'disambig_cpt': disambiguated_cpt,
            'pred_cpt': final.procedure,
            'cpt_match': cpt_match,
            'is_interventional': is_interventional,
            'gt_modifier': gt_modifier,
            'pred_modifier': final.modifier,
            'gt_icd': gt_icd,
            'pred_icd': final.icd10_diagnosis,
            'icd_recall': f"{recall:.0%}",
            'llm_raw_response': llm_result.raw_response,
            'llm_latency_ms': int(llm_result.latency_ms),
        })
    
    # Save CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ FIXED: {len(fixed_samples)}/{len(FAILED_SAMPLES)}")
    if fixed_samples:
        print(f"   Samples: {fixed_samples}")
    
    print(f"\n❌ STILL FAILING: {len(still_failing)}/{len(FAILED_SAMPLES)}")
    if still_failing:
        print(f"   Samples: {still_failing}")
    
    fix_rate = len(fixed_samples) / len(FAILED_SAMPLES) * 100
    print(f"\n📊 Fix Rate: {fix_rate:.1f}%")
    
    print(f"\n📁 CSV saved: {OUTPUT_CSV}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return 0 if len(still_failing) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
