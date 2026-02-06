#!/usr/bin/env python3
"""
VFinal3 - 5 Sample Verbose Test
Shows input/output for each pipeline step.
"""

import sys
import os
import argparse
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

RANDOM_STATE = 789
NUM_SAMPLES = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-reranker', action='store_true')
    args = parser.parse_args()
    
    hf_token = os.environ.get('HF_API_TOKEN')
    if not hf_token:
        print("ERROR: HF_API_TOKEN not set")
        sys.exit(1)
    
    print("=" * 100)
    print("VFINAL3 - VERBOSE 5 SAMPLE TEST")
    print("=" * 100)
    print(f"Use Reranker: {args.use_reranker}")
    
    # Load data
    data_path = PROJECT_DIR / 'data' / 'consolidated_cleaned.csv'
    df = pd.read_csv(data_path, low_memory=False)
    
    valid = df[
        df['Report'].notna() & 
        df['Procedure'].notna() & 
        (df['Report'].str.len() > 300) &
        (df['Procedure'].astype(str).str.match(r'^[0-9]{5}$'))
    ]
    
    samples = valid.sample(n=NUM_SAMPLES, random_state=RANDOM_STATE)
    
    # Initialize
    print("\nInitializing components...")
    preprocessor = Preprocessor(df=df, use_rag=True, rag_use_reranker=args.use_reranker)
    llm = LLMPredictor(backend="huggingface")
    postprocessor = Postprocessor(bill_type='P')
    print("Ready\n")
    
    for i, (idx, row) in enumerate(samples.iterrows(), 1):
        report = str(row['Report'])
        gt_cpt = str(row['Procedure'])
        gt_icd = str(row.get('ICD10 - Diagnosis', ''))
        gt_modifier = str(row.get('Modifier', '')).replace('nan', '').replace('.0', '').strip()
        exam_desc = str(row.get('Exam Description', ''))
        
        print("\n" + "#" * 100)
        print(f"# SAMPLE {i}/5")
        print("#" * 100)
        
        # ============ GROUND TRUTH ============
        print("\n" + "=" * 50)
        print("GROUND TRUTH")
        print("=" * 50)
        print(f"Exam Description: {exam_desc}")
        print(f"CPT: {gt_cpt}")
        print(f"Modifier: '{gt_modifier}'")
        print(f"ICD: {gt_icd}")
        
        # ============ STEP 1: INPUT ============
        print("\n" + "=" * 50)
        print("STEP 1: INPUT REPORT (first 500 chars)")
        print("=" * 50)
        print(report[:500])
        print("...")
        
        # ============ STEP 2: PREPROCESSOR ============
        print("\n" + "=" * 50)
        print("STEP 2: PREPROCESSOR OUTPUT")
        print("=" * 50)
        
        prep = preprocessor.preprocess(report)
        
        print(f"\n[Extracted CPT]: {prep.extracted_cpt or 'None'}")
        print(f"[Modality]: {prep.modality}")
        print(f"[Body Part]: {prep.body_part}")
        print(f"[Laterality]: {prep.laterality}")
        print(f"[Has Contrast]: {prep.has_contrast}")
        
        print(f"\n[RAG CPT Candidates]: {prep.rag_cpt_candidates[:5] if prep.rag_cpt_candidates else 'None'}")
        print(f"[ICD Candidates (top 5)]: {[c for c,_ in prep.icd_candidates[:5]]}")
        
        num_examples = len(prep.examples)
        print(f"\n[Few-Shot Examples ({num_examples})]:")
        for j, ex in enumerate(prep.examples[:3], 1):
            print(f"  Example {j}: CPT={ex.get('cpt')}, MOD={ex.get('modifier')}, ICD={ex.get('icd', '')[:30]}...")
            print(f"    Report: {ex.get('report', '')[:100]}...")
        
        print(f"\n[Generated Prompt (first 1500 chars)]:")
        print("-" * 40)
        print(prep.prompt[:1500])
        print("...")
        print("-" * 40)
        
        # ============ STEP 3: LLM PREDICTOR ============
        print("\n" + "=" * 50)
        print("STEP 3: LLM PREDICTOR")
        print("=" * 50)
        
        llm_result = llm.predict(
            prompt=prep.prompt,
            extracted_cpt=prep.extracted_cpt,
            valid_icd_codes=set(c for c, _ in prep.icd_candidates),
            report_text=report,
            exam_desc=exam_desc
        )
        
        print(f"\n[LLM Raw Response]: {llm_result.raw_response}")
        print(f"[Parsed CPT]: {llm_result.cpt_code}")
        print(f"[Parsed Modifiers]: {llm_result.modifiers}")
        print(f"[Parsed ICD]: {llm_result.icd_codes}")
        print(f"[Latency]: {llm_result.latency_ms:.0f}ms")
        
        # ============ STEP 4: POSTPROCESSOR ============
        print("\n" + "=" * 50)
        print("STEP 4: POSTPROCESSOR")
        print("=" * 50)
        
        final = postprocessor.postprocess(
            cpt_code=llm_result.cpt_code,
            icd_codes=llm_result.icd_codes,
            laterality=prep.laterality,
            cpt_was_extracted=bool(prep.extracted_cpt),
            valid_icd_codes=None,
            llm_suggested_modifiers=llm_result.modifiers
        )
        
        print(f"\n[Final CPT]: {final.procedure}")
        print(f"[Final Modifier]: '{final.modifier}'")
        print(f"[Final ICD]: {final.icd10_diagnosis}")
        
        # ============ ACCURACY CHECK ============
        print("\n" + "=" * 50)
        print("ACCURACY CHECK")
        print("=" * 50)
        
        cpt_match = final.procedure == gt_cpt
        gt_icd_set = set(gt_icd.replace(' ', '').split(',')) if gt_icd and gt_icd != 'nan' else set()
        pred_icd_set = set(final.icd10_diagnosis.replace(' ', '').split(',')) if final.icd10_diagnosis else set()
        icd_recall = len(gt_icd_set & pred_icd_set) / len(gt_icd_set) if gt_icd_set else 1.0
        
        cpt_status = "MATCH" if cpt_match else "MISMATCH"
        print(f"\nCPT: {cpt_status}")
        print(f"  Ground Truth: {gt_cpt}")
        print(f"  Predicted:    {final.procedure}")
        
        print(f"\nICD Recall: {icd_recall:.0%}")
        print(f"  Ground Truth: {gt_icd_set}")
        print(f"  Predicted:    {pred_icd_set}")
        print(f"  Overlap:      {gt_icd_set & pred_icd_set}")
    
    print("\n" + "=" * 100)
    print("TEST COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
