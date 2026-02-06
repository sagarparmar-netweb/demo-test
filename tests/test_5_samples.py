#!/usr/bin/env python3
"""
VFinal2 - 5 Sample Validation Test

Run this before production to verify pipeline accuracy.
Uses fixed random_state=789 for reproducible results.

Expected Results (as of Jan 8, 2026):
- CPT Accuracy: 100% (5/5)
- Modifier Correct: 100% (5/5)
- ICD Recall: 100%
- ICD Exact Match: 100% (5/5)

Usage:
    export HF_API_TOKEN="your_token_here"
    python tests/test_5_samples.py
"""

import sys
import os
from pathlib import Path

# Add VFinal2 to path
VFINAL2_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(VFINAL2_DIR))
sys.path.insert(0, str(VFINAL2_DIR / 'src'))

# Also add project root for data access
PROJECT_DIR = VFINAL2_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.preprocessor import Preprocessor
from src.llm_predictor import LLMPredictor
from src.postprocessor import Postprocessor

# Configuration
RANDOM_STATE = 789  # Fixed for reproducibility
NUM_SAMPLES = 5
OUTPUT_CSV = PROJECT_DIR / 'reports' / 'vfinal2-test1.csv'


def main():
    # Check HF token
    hf_token = os.environ.get('HF_API_TOKEN')
    if not hf_token:
        print("ERROR: HF_API_TOKEN environment variable not set")
        print("Usage: export HF_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    print("=" * 80)
    print("VFINAL2 - 5 SAMPLE VALIDATION TEST")
    print("=" * 80)
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
    preprocessor = Preprocessor(df=df, use_rag=True)
    llm = LLMPredictor(backend="huggingface")
    postprocessor = Postprocessor(bill_type='P')
    print("✓ All components initialized")
    
    print(f"\nRunning {NUM_SAMPLES} tests...\n")
    
    rows = []
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
        
        gt_icd_set = set(gt_icd.replace(' ', '').split(',')) if gt_icd and gt_icd != 'nan' else set()
        pred_icd_set = set(final.icd10_diagnosis.replace(' ', '').split(',')) if final.icd10_diagnosis else set()
        icd_recall = len(gt_icd_set & pred_icd_set) / len(gt_icd_set) if gt_icd_set else 1.0
        icd_exact = gt_icd_set == pred_icd_set
        
        # Modifier correctness
        pred_mod_set = set(final.modifier.split()) if final.modifier else set()
        if is_interventional:
            mod_correct = '26' not in pred_mod_set  # Should NOT have 26
        else:
            mod_correct = '26' in pred_mod_set  # Should have 26
        
        rows.append({
            'sample': i,
            'exam_description': exam_desc[:60],
            'gt_cpt': gt_cpt,
            'pred_cpt': final.procedure,
            'cpt_match': cpt_match,
            'is_interventional': is_interventional,
            'gt_modifier': gt_modifier,
            'pred_modifier': final.modifier,
            'modifier_correct': mod_correct,
            'gt_icd': gt_icd,
            'pred_icd': final.icd10_diagnosis,
            'icd_recall': f"{icd_recall:.0%}",
            'icd_exact': icd_exact,
            'llm_raw_response': llm_result.raw_response,
            'llm_latency_ms': int(llm_result.latency_ms),
            'full_prompt': prep.prompt.replace('\n', '\\n')[:2000]
        })
        
        # Print progress
        cpt_status = "✅" if cpt_match else "❌"
        mod_status = "✅" if mod_correct else "❌"
        icd_status = "✅" if icd_recall >= 0.5 else "❌"
        tag = " [INTERVENTIONAL]" if is_interventional else ""
        
        print(f"Sample {i}/5: {exam_desc[:40]}...")
        print(f"  CPT: {cpt_status} GT={gt_cpt} → Pred={final.procedure}{tag}")
        print(f"  MOD: {mod_status} GT='{gt_modifier}' → Pred='{final.modifier}'")
        print(f"  ICD: {icd_status} Recall={icd_recall:.0%}")
        print()
    
    # Save CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    
    # Print summary
    print("=" * 80)
    print("ACCURACY SUMMARY")
    print("=" * 80)
    
    cpt_acc = sum(1 for r in rows if r['cpt_match']) / len(rows)
    mod_acc = sum(1 for r in rows if r['modifier_correct']) / len(rows)
    icd_recalls = [float(r['icd_recall'].replace('%', '')) / 100 for r in rows]
    avg_icd_recall = sum(icd_recalls) / len(icd_recalls)
    icd_exact_acc = sum(1 for r in rows if r['icd_exact']) / len(rows)
    
    print(f"CPT Accuracy:        {cpt_acc:.0%} ({sum(1 for r in rows if r['cpt_match'])}/{NUM_SAMPLES})")
    print(f"Modifier Correct:    {mod_acc:.0%} ({sum(1 for r in rows if r['modifier_correct'])}/{NUM_SAMPLES})")
    print(f"ICD Recall (avg):    {avg_icd_recall:.0%}")
    print(f"ICD Exact Match:     {icd_exact_acc:.0%} ({sum(1 for r in rows if r['icd_exact'])}/{NUM_SAMPLES})")
    
    print(f"\n📁 CSV saved: {OUTPUT_CSV}")
    
    # Verify expected results
    print("\n" + "=" * 80)
    print("VALIDATION CHECK")
    print("=" * 80)
    
    all_pass = (
        cpt_acc == 1.0 and 
        mod_acc == 1.0 and 
        avg_icd_recall == 1.0 and 
        icd_exact_acc == 1.0
    )
    
    if all_pass:
        print("✅ ALL TESTS PASSED - Ready for production!")
    else:
        print("❌ SOME TESTS FAILED - Review before production")
        if cpt_acc < 1.0:
            print(f"   - CPT accuracy below 100%")
        if mod_acc < 1.0:
            print(f"   - Modifier accuracy below 100%")
        if avg_icd_recall < 1.0:
            print(f"   - ICD recall below 100%")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
