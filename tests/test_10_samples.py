#!/usr/bin/env python3
"""
VFinal3 - 10 Sample Test for Iterative Debugging
"""
import sys
import os
from pathlib import Path

VFINAL3_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(VFINAL3_DIR))
sys.path.insert(0, str(VFINAL3_DIR / 'src'))
PROJECT_DIR = VFINAL3_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from src.preprocessor import Preprocessor
from src.llm_predictor import LLMPredictor
from src.postprocessor import Postprocessor

RANDOM_STATE = 999  # Different seed for variety
NUM_SAMPLES = 10


def main():
    hf_token = os.environ.get('HF_API_TOKEN')
    if not hf_token:
        print("ERROR: HF_API_TOKEN not set")
        sys.exit(1)
    
    print("=" * 80)
    print("VFINAL3 - 10 SAMPLE TEST (with reranker)")
    print("=" * 80)
    
    # Load data
    data_path = VFINAL3_DIR / 'data' / 'consolidated_cleaned_v1.csv'
    print(VFINAL3_DIR)
    print(PROJECT_DIR)
    df = pd.read_csv(data_path, low_memory=False)
    
    valid = df[
        df['Report'].notna() & 
        df['Procedure'].notna() & 
        (df['Report'].str.len() > 300) &
        (df['Procedure'].astype(str).str.match(r'^[0-9]{5}$'))
    ]
    print(f"Valid records: {len(valid):,}")
    
    samples = valid.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
    
    print("\nInitializing with reranker...")
    preprocessor = Preprocessor(df=df, use_rag=True, rag_use_reranker=True)
    llm = LLMPredictor(backend="huggingface")
    postprocessor = Postprocessor(bill_type='P')
    print("Ready\n")
    
    results = []
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        report = str(row['Report'])
        gt_cpt = str(row['Procedure'])
        gt_icd = str(row.get('ICD10 - Diagnosis', ''))
        exam_desc = str(row.get('Exam Description', ''))
        
        # Preprocess
        prep = preprocessor.preprocess(report)
        
        # LLM Predict
        llm_result = llm.predict(
            prompt=prep.prompt,
            extracted_cpt=prep.extracted_cpt,
            valid_icd_codes=set(c for c, _ in prep.icd_candidates),
            report_text=report,
            exam_desc=exam_desc
        )
        
        # Postprocess
        final = postprocessor.postprocess(
            cpt_code=llm_result.cpt_code,
            icd_codes=llm_result.icd_codes,
            laterality=prep.laterality,
            cpt_was_extracted=bool(prep.extracted_cpt),
            valid_icd_codes=None,
            llm_suggested_modifiers=llm_result.modifiers
        )
        
        cpt_match = final.procedure == gt_cpt
        status = "✅" if cpt_match else "❌"
        
        print(f"{status} Sample {i}/10: {exam_desc[:50]}...")
        print(f"   GT CPT: {gt_cpt} | Pred: {final.procedure}")
        print(f"   RAG CPT candidates: {prep.rag_cpt_candidates[:5]}")
        print(f"   Extracted CPT: {prep.extracted_cpt or 'None'}")
        if not cpt_match:
            print(f"   LLM raw: {llm_result.raw_response}")
        print()
        
        results.append({
            'sample': i,
            'exam_desc': exam_desc[:60],
            'gt_cpt': gt_cpt,
            'pred_cpt': final.procedure,
            'match': cpt_match,
            'rag_candidates': prep.rag_cpt_candidates[:5],
            'extracted_cpt': prep.extracted_cpt,
            'llm_raw': llm_result.raw_response
        })
    
    # Summary
    correct = sum(1 for r in results if r['match'])
    print("=" * 80)
    print(f"CPT ACCURACY: {correct}/{NUM_SAMPLES} ({100*correct/NUM_SAMPLES:.0f}%)")
    print("=" * 80)
    
    if correct < NUM_SAMPLES:
        print("\nFAILURES:")
        for r in results:
            if not r['match']:
                print(f"  {r['exam_desc']}")
                print(f"    GT={r['gt_cpt']} Pred={r['pred_cpt']}")
                print(f"    RAG candidates: {r['rag_candidates']}")
                print(f"    GT in candidates: {r['gt_cpt'] in r['rag_candidates']}")
    
    return 0 if correct == NUM_SAMPLES else 1

if __name__ == "__main__":
    sys.exit(main())
