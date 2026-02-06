#!/usr/bin/env python3
"""
TRAINING DATA GENERATOR - HYBRID APPROACH

Generates training data with:
1. STABLE prompt template (never changes)
2. DYNAMIC reference codes (from CPT-ICD priors, for guidance only)

KEY INSIGHT: The model learns to code based on the REPORT CONTENT,
with reference codes as helpful guidance, NOT as constraints.

This approach allows:
- RAG index to grow/change without retraining
- Model to predict correct codes even when not in candidates
- Reference codes to help when multiple valid options exist

The model learns:
1. How to extract clinical info from reports (PRIMARY)
2. How to use reference codes as sanity checks (SECONDARY)
"""

import json
import sys
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import random
from collections import Counter

# =============================================================================
# PROMPT TEMPLATES - MUST MATCH V13 PREPROCESSOR EXACTLY
# =============================================================================
# These are the EXACT formats used in src/preprocessor.py
# The model learns these formats - they must be identical at training & inference

# Template 1: CPT is known (extracted from report)
PROMPT_TEMPLATE_CPT_KNOWN = """You are an expert radiology coder. Study these examples:

{examples}

---

Code this report:
{report}

Available codes (sorted by relevance):
{icd_candidates}

Rules:
- Match coding patterns from examples
- Code the clinical indication
- Code confirmed findings from impression
- Use laterality-specific codes when applicable
- For normal studies, code only the indication

Output ONLY codes (comma-separated):"""


# Template 2: CPT unknown, but have RAG candidates
PROMPT_TEMPLATE_WITH_RAG = """You are an expert radiology coder. Study these similar cases:

{examples}

---

Code this report:
{report}

SIMILAR CASES CPT CODES (most likely):
{cpt_candidates}

Available ICD codes (from similar cases):
{icd_candidates}

Rules:
- The CPT code should match one from the similar cases above
- Code the clinical indication (R/Z/S code)
- Code confirmed findings from impression
- For normal studies, code only the indication

Output format: CPT, ICD1, ICD2, ...
Example: 71046, R91.8, J18.9"""


# Template 3: CPT unknown, no RAG (fallback - rare)
PROMPT_TEMPLATE_FALLBACK = """You are an expert radiology coder.

Code this report:
{report}

DETECTED EXAM INFO:
- Modality: {modality}
- Body Part: {body_part}

CPT CODE OPTIONS:
{cpt_candidates}

Available ICD codes (sorted by relevance):
{icd_candidates}

CPT SELECTION RULES:
1. Match the imaging modality (X-ray, CT, MRI, Ultrasound, Mammography)
2. Match the body part (head, chest, abdomen, pelvis, extremity)
3. Check for contrast (with, without, or both)

ICD SELECTION RULES:
- Code the clinical indication (R/Z/S code)
- Code confirmed findings from impression
- For normal studies, code only the indication

Output format: CPT, ICD1, ICD2, ...
Example: 71046, R91.8, J18.9"""


def build_cpt_icd_priors(df: pd.DataFrame) -> dict:
    """
    Build CPT → ICD mappings from training data.
    
    This is STABLE - based on training data, not RAG.
    """
    print("Building CPT-ICD priors from training data...")
    
    cpt_to_icd = {}
    
    for _, row in df.iterrows():
        cpt = str(row.get('Procedure', ''))
        icd = str(row.get('ICD10 - Diagnosis', ''))
        
        if cpt == 'nan' or icd == 'nan':
            continue
        
        if cpt not in cpt_to_icd:
            cpt_to_icd[cpt] = Counter()
        
        for code in icd.split(','):
            code = code.strip()
            if code:
                cpt_to_icd[cpt][code] += 1
    
    # Convert to sorted lists (most common first)
    result = {}
    for cpt, counter in cpt_to_icd.items():
        result[cpt] = [code for code, _ in counter.most_common(50)]
    
    print(f"  Built priors for {len(result)} CPT codes")
    return result


def get_cpt_candidates(cpt: str, cpt_to_icd: dict, all_cpts: list) -> list:
    """
    Get CPT candidates for training.
    
    Strategy:
    - Always include the ground truth CPT
    - Add 5-10 similar CPTs (same prefix = same body area)
    """
    candidates = [cpt]
    
    # Add CPTs with same first 2 digits (similar procedures)
    prefix = cpt[:2] if len(cpt) >= 2 else cpt
    similar = [c for c in all_cpts if c.startswith(prefix) and c != cpt]
    candidates.extend(similar[:7])
    
    # Add some random CPTs for diversity
    other = [c for c in all_cpts if c not in candidates]
    if other:
        candidates.extend(random.sample(other, min(3, len(other))))
    
    return candidates[:10]


def get_icd_candidates(cpt: str, icd_codes: list, cpt_to_icd: dict) -> list:
    """
    Get ICD candidates for training.
    
    Strategy:
    - Always include the ground truth ICD codes
    - Add top codes for this CPT from priors
    """
    candidates = list(icd_codes)  # Ground truth first
    
    # Add prior codes for this CPT
    if cpt in cpt_to_icd:
        for code in cpt_to_icd[cpt]:
            if code not in candidates:
                candidates.append(code)
            if len(candidates) >= 30:
                break
    
    return candidates[:30]


def get_examples_for_cpt(cpt: str, df: pd.DataFrame, n: int = 3) -> str:
    """Get few-shot examples for a CPT code."""
    same_cpt = df[df['Procedure'] == cpt].head(n + 1)  # +1 in case we need to skip current
    
    examples = []
    for _, row in same_cpt.iterrows():
        rep = str(row.get('Report', ''))[:400]
        icd = str(row.get('ICD10 - Diagnosis', ''))
        if icd and 'nan' not in icd.lower():
            examples.append(f'Report: {rep}...\nICD: {icd}')
        if len(examples) >= n:
            break
    
    return '\n---\n'.join(examples) if examples else "No examples available."


def generate_training_data(
    data_path: str,
    output_path: str,
    max_samples: int = None,
    max_report_length: int = 3000,
    train_split: float = 0.9,
    cpt_known_ratio: float = 0.5  # 50% with CPT known, 50% predict CPT
):
    """
    Generate training data matching V13 preprocessor formats exactly.
    
    Generates two types of examples:
    1. CPT known (Template 1) - model just predicts ICD
    2. CPT unknown with candidates (Template 2) - model predicts both
    """
    print("=" * 70)
    print("GENERATING TRAINING DATA (V13 FORMAT)")
    print("=" * 70)
    
    # Load data
    print(f"\n1. Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filter valid records
    valid_df = df[
        df['Report'].notna() &
        (df['Report'].str.len() > 200) &
        df['Procedure'].notna() &
        (df['Procedure'] != 'nan') &
        df['ICD10 - Diagnosis'].notna() &
        (df['ICD10 - Diagnosis'] != 'nan')
    ].copy()
    
    print(f"   Total records: {len(df):,}")
    print(f"   Valid records: {len(valid_df):,}")
    
    if max_samples and max_samples < len(valid_df):
        valid_df = valid_df.sample(n=max_samples, random_state=42)
        print(f"   Sampled: {len(valid_df):,}")
    
    # Build CPT-ICD priors (STABLE - from training data only)
    print("\n2. Building CPT-ICD priors...")
    cpt_to_icd = build_cpt_icd_priors(df)
    all_cpts = list(cpt_to_icd.keys())
    
    # Generate training examples
    print("\n3. Generating training examples...")
    print(f"   CPT known ratio: {cpt_known_ratio:.0%}")
    
    training_data = []
    cpt_known_count = 0
    cpt_unknown_count = 0
    
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
        try:
            report = str(row['Report'])[:max_report_length]
            cpt = str(row['Procedure'])
            icd = str(row['ICD10 - Diagnosis'])
            
            # Parse ICD codes
            icd_codes = [c.strip() for c in icd.split(',') if c.strip()]
            
            # Get candidates
            cpt_candidates = get_cpt_candidates(cpt, cpt_to_icd, all_cpts)
            icd_candidates = get_icd_candidates(cpt, icd_codes, cpt_to_icd)
            
            # Get examples
            examples_str = get_examples_for_cpt(cpt, df, n=3)
            
            # Randomly decide: CPT known or unknown
            # This teaches the model to handle both scenarios
            if random.random() < cpt_known_ratio:
                # Template 1: CPT is known - just predict ICD
                prompt = PROMPT_TEMPLATE_CPT_KNOWN.format(
                    examples=examples_str,
                    report=report,
                    icd_candidates='\n'.join([f'{c}' for c in icd_candidates])
                )
                expected_output = icd  # Just ICD codes
                cpt_known_count += 1
            else:
                # Template 2: CPT unknown - predict both
                cpt_str = '\n'.join([f'{c}' for c in cpt_candidates])
                prompt = PROMPT_TEMPLATE_WITH_RAG.format(
                    examples=examples_str,
                    report=report,
                    cpt_candidates=cpt_str,
                    icd_candidates='\n'.join([f'{c}' for c in icd_candidates])
                )
                expected_output = f"{cpt}, {icd}"  # CPT + ICD codes
                cpt_unknown_count += 1
            
            training_data.append({
                "input": prompt,
                "output": expected_output
            })
            
        except Exception as e:
            continue
    
    print(f"   CPT known examples:   {cpt_known_count:,}")
    print(f"   CPT unknown examples: {cpt_unknown_count:,}")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(training_data)
    
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]
    
    # Save files
    print(f"\n4. Saving files...")
    
    # Training data
    train_path = output_path.replace('.jsonl', '_train.jsonl')
    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    print(f"   Train: {len(train_data):,} examples → {train_path}")
    
    # Validation data
    val_path = output_path.replace('.jsonl', '_val.jsonl')
    with open(val_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    print(f"   Val:   {len(val_data):,} examples → {val_path}")
    
    # Save the prompt template separately (for reference)
    template_path = output_path.replace('.jsonl', '_template.txt')
    with open(template_path, 'w') as f:
        f.write("# STABLE PROMPT TEMPLATE\n")
        f.write("# This exact format is used for training AND inference\n\n")
        f.write(PROMPT_TEMPLATE)
    print(f"   Template: {template_path}")
    
    # Save priors (can be used at inference if RAG unavailable)
    priors_path = output_path.replace('.jsonl', '_priors.json')
    with open(priors_path, 'w') as f:
        json.dump(cpt_to_icd, f)
    print(f"   Priors: {priors_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING DATA GENERATED")
    print("=" * 70)
    print(f"Train examples: {len(train_data):,}")
    print(f"Val examples:   {len(val_data):,}")
    print(f"Total:          {len(training_data):,}")
    
    # Show sample
    print("\n" + "=" * 70)
    print("SAMPLE TRAINING EXAMPLE")
    print("=" * 70)
    sample = random.choice(training_data)
    print(f"\n[INPUT] ({len(sample['input'])} chars)")
    print("-" * 70)
    print(sample['input'][:1500])
    if len(sample['input']) > 1500:
        print("... (truncated)")
    print("-" * 70)
    print(f"\n[OUTPUT]")
    print(sample['output'])
    
    return training_data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training data (hybrid approach)")
    parser.add_argument("--data", default="Training_Data_True_Source_20251231/Raw_Data_Consolidated.csv")
    parser.add_argument("--output", default="training/data/v13_training.jsonl")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-report-length", type=int, default=3000)
    
    args = parser.parse_args()
    
    # Resolve paths
    project_dir = Path(__file__).parent.parent.parent.parent
    data_path = project_dir / args.data
    output_path = project_dir / args.output
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generate_training_data(
        str(data_path),
        str(output_path),
        args.max_samples,
        args.max_report_length
    )
