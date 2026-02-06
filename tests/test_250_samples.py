#!/usr/bin/env python3
"""
VFinal4 - 250 Sample Validation Test

Larger scale test to validate pipeline accuracy.
Uses fixed random_state=456 for reproducible results.

Usage:
    export HF_API_TOKEN="your_token_here"
    python tests/test_250_samples.py
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
from src.preprocessor import Preprocessor
from src.llm_predictor import LLMPredictor
from src.postprocessor import Postprocessor

# Configuration
RANDOM_STATE = 456  # Different from 50-sample test for variety
NUM_SAMPLES = 250
OUTPUT_CSV = PROJECT_DIR / 'reports' / 'vfinal4-test250.csv'


def main():
    # Check HF token
    hf_token = os.environ.get('HF_API_TOKEN')
    if not hf_token:
        print("ERROR: HF_API_TOKEN environment variable not set")
        print("Usage: export HF_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    print("=" * 80)
    print("VFINAL4 - 250 SAMPLE VALIDATION TEST")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Random State: {RANDOM_STATE}")
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
    
    # Get fixed samples
    samples = valid.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
    
    # Initialize components
    print("\nInitializing pipeline components...")
    preprocessor = Preprocessor(df=df, use_rag=True, rag_use_reranker=True)
    llm = LLMPredictor(backend="huggingface")
    postprocessor = Postprocessor(bill_type='P')
    print("✓ All components initialized")
    
    print(f"\nRunning {NUM_SAMPLES} tests...\n")
    
    rows = []
    cpt_correct = 0
    mod_correct = 0
    icd_recalls = []
    icd_exact = 0
    
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        report = str(row['Report'])
        gt_cpt = str(row['Procedure'])
        gt_icd = str(row.get('ICD10 - Diagnosis', ''))
        gt_modifier = str(row.get('Modifier', '')).replace('nan', '').replace('.0', '').strip()
        exam_desc = str(row.get('Exam Description', ''))
        
        is_interventional = gt_cpt in Postprocessor.INTERVENTIONAL_CPT_CODES
        
        # Step 1: Preprocess
        prep = preprocessor.preprocess(report)
        
        # Step 2: LLM Prediction (with exam_desc for disambiguation)
        llm_result = llm.predict(
            prompt=prep.prompt,
            extracted_cpt=prep.extracted_cpt,
            valid_icd_codes=set(c for c, _ in prep.icd_candidates),
            report_text=report,
            exam_desc=exam_desc
        )
        
        # Step 3: Postprocess (NO ICD filtering - trust the LLM)
        final = postprocessor.postprocess(
            cpt_code=llm_result.cpt_code,
            icd_codes=llm_result.icd_codes,
            laterality=prep.laterality,
            cpt_was_extracted=bool(prep.extracted_cpt),
            valid_icd_codes=None,  # Don't filter ICD codes
            llm_suggested_modifiers=llm_result.modifiers
        )
        
        # Calculate accuracy
        cpt_match = final.procedure == gt_cpt
        if cpt_match:
            cpt_correct += 1
        
        gt_icd_set = set(gt_icd.replace(' ', '').split(',')) if gt_icd and gt_icd != 'nan' else set()
        pred_icd_set = set(final.icd10_diagnosis.replace(' ', '').split(',')) if final.icd10_diagnosis else set()
        recall = len(gt_icd_set & pred_icd_set) / len(gt_icd_set) if gt_icd_set else 1.0
        icd_recalls.append(recall)
        
        is_icd_exact = gt_icd_set == pred_icd_set
        if is_icd_exact:
            icd_exact += 1
        
        # Modifier correctness
        pred_mod_set = set(final.modifier.split()) if final.modifier else set()
        if is_interventional:
            is_mod_correct = '26' not in pred_mod_set  # Should NOT have 26
        else:
            is_mod_correct = '26' in pred_mod_set  # Should have 26
        if is_mod_correct:
            mod_correct += 1
        
        rows.append({
            'sample': i,
            'exam_description': exam_desc[:60],
            'gt_cpt': gt_cpt,
            'pred_cpt': final.procedure,
            'cpt_match': cpt_match,
            'is_interventional': is_interventional,
            'gt_modifier': gt_modifier,
            'pred_modifier': final.modifier,
            'modifier_correct': is_mod_correct,
            'gt_icd': gt_icd,
            'pred_icd': final.icd10_diagnosis,
            'icd_recall': f"{recall:.0%}",
            'icd_exact': is_icd_exact,
            'llm_raw_response': llm_result.raw_response,
            'llm_latency_ms': int(llm_result.latency_ms),
        })
        
        # Print progress every 25 samples
        if i % 25 == 0 or i == NUM_SAMPLES:
            curr_cpt_acc = cpt_correct / i
            curr_mod_acc = mod_correct / i
            curr_icd_recall = sum(icd_recalls) / len(icd_recalls)
            print(f"Progress: {i}/{NUM_SAMPLES} | CPT: {curr_cpt_acc:.1%} | MOD: {curr_mod_acc:.1%} | ICD Recall: {curr_icd_recall:.1%}")
    
    # Save CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY")
    print("=" * 80)
    
    cpt_acc = cpt_correct / NUM_SAMPLES
    mod_acc = mod_correct / NUM_SAMPLES
    avg_icd_recall = sum(icd_recalls) / len(icd_recalls)
    icd_exact_acc = icd_exact / NUM_SAMPLES
    
    print(f"CPT Accuracy:        {cpt_acc:.1%} ({cpt_correct}/{NUM_SAMPLES})")
    print(f"Modifier Correct:    {mod_acc:.1%} ({mod_correct}/{NUM_SAMPLES})")
    print(f"ICD Recall (avg):    {avg_icd_recall:.1%}")
    print(f"ICD Exact Match:     {icd_exact_acc:.1%} ({icd_exact}/{NUM_SAMPLES})")
    
    # Show failures
    failures = [r for r in rows if not r['cpt_match']]
    if failures:
        print(f"\n❌ CPT FAILURES ({len(failures)}):")
        for f in failures[:20]:  # Show first 20
            print(f"   Sample {f['sample']}: GT={f['gt_cpt']} → Pred={f['pred_cpt']} | {f['exam_description'][:40]}")
        if len(failures) > 20:
            print(f"   ... and {len(failures) - 20} more failures")
    
    # Group failures by pattern
    print(f"\n📊 FAILURE ANALYSIS:")
    failure_patterns = {}
    for f in failures:
        key = f"{f['gt_cpt']} → {f['pred_cpt']}"
        if key not in failure_patterns:
            failure_patterns[key] = []
        failure_patterns[key].append(f['exam_description'][:30])
    
    sorted_patterns = sorted(failure_patterns.items(), key=lambda x: -len(x[1]))
    for pattern, examples in sorted_patterns[:10]:
        print(f"   {pattern}: {len(examples)} cases")
    
    print(f"\n📁 CSV saved: {OUTPUT_CSV}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Validation check
    print("\n" + "=" * 80)
    print("VALIDATION CHECK")
    print("=" * 80)
    
    if cpt_acc >= 0.90 and mod_acc >= 0.95:
        print("✅ TESTS PASSED - Ready for production!")
        return 0
    else:
        print("⚠️ REVIEW NEEDED - Some metrics below threshold")
        if cpt_acc < 0.90:
            print(f"   - CPT accuracy {cpt_acc:.1%} < 90%")
        if mod_acc < 0.95:
            print(f"   - Modifier accuracy {mod_acc:.1%} < 95%")
        return 1


if __name__ == "__main__":
    sys.exit(main())
