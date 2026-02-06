#!/usr/bin/env python3
"""
CREATE DIVERSE TRAINING DATASET

Requirements:
1. Only records from Dec 24, 2024 onwards (skip initial ~35K records)
2. Cover top 95% of CPT codes
3. Cover top 75-80% of ICD codes
4. Ensure variety and quality

Output:
- training_dataset.csv - The curated dataset
- dataset_stats.json - Statistics about coverage
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import random

# =============================================================================
# CONFIGURATION
# =============================================================================

# Date filter
MIN_DATE = "2024-12-24"

# Coverage targets
CPT_COVERAGE_TARGET = 0.95  # 95% of CPT codes
ICD_COVERAGE_TARGET = 0.80  # 80% of ICD codes

# Minimum samples per CPT code (to ensure we have examples for few-shot)
MIN_SAMPLES_PER_CPT = 3

# Maximum samples per CPT code (to prevent over-representation)
MAX_SAMPLES_PER_CPT = 500


def analyze_code_distribution(df: pd.DataFrame) -> dict:
    """Analyze CPT and ICD code distributions."""
    
    # CPT distribution
    cpt_counts = df['Procedure'].value_counts()
    total_records = len(df)
    
    # Calculate cumulative coverage
    cpt_cumsum = cpt_counts.cumsum() / total_records
    cpts_for_95 = (cpt_cumsum < CPT_COVERAGE_TARGET).sum() + 1
    
    # ICD distribution
    all_icds = []
    for icd_str in df['ICD10 - Diagnosis'].dropna():
        for code in str(icd_str).split(','):
            code = code.strip()
            if code and code != 'nan':
                all_icds.append(code)
    
    icd_counts = Counter(all_icds)
    total_icds = len(all_icds)
    
    # Calculate ICDs needed for 80%
    cumsum = 0
    icds_for_80 = 0
    for i, (icd, count) in enumerate(icd_counts.most_common()):
        cumsum += count
        if cumsum / total_icds >= ICD_COVERAGE_TARGET:
            icds_for_80 = i + 1
            break
    
    return {
        'total_records': total_records,
        'unique_cpts': len(cpt_counts),
        'cpts_for_95': cpts_for_95,
        'top_cpts': cpt_counts.head(cpts_for_95).index.tolist(),
        'cpt_counts': cpt_counts.to_dict(),
        'unique_icds': len(icd_counts),
        'icds_for_80': icds_for_80,
        'top_icds': [icd for icd, _ in icd_counts.most_common(icds_for_80)],
        'icd_counts': dict(icd_counts.most_common(500))
    }


def create_diverse_dataset(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Create a diverse dataset that covers required code distributions.
    
    Strategy:
    1. Include ALL records with top CPT codes (ensure coverage)
    2. Ensure minimum samples per CPT code
    3. Cap maximum samples per CPT code (prevent over-fitting)
    4. Ensure ICD coverage by including rare ICD code samples
    """
    
    target_cpts = set(stats['top_cpts'])
    target_icds = set(stats['top_icds'])
    
    print(f"\n   Target CPT codes: {len(target_cpts)}")
    print(f"   Target ICD codes: {len(target_icds)}")
    
    # Step 1: Get all records with target CPT codes
    selected_indices = set()
    cpt_sample_counts = Counter()
    
    for idx, row in df.iterrows():
        cpt = str(row['Procedure'])
        if cpt in target_cpts:
            if cpt_sample_counts[cpt] < MAX_SAMPLES_PER_CPT:
                selected_indices.add(idx)
                cpt_sample_counts[cpt] += 1
    
    print(f"   After CPT filter: {len(selected_indices):,} records")
    
    # Step 2: Ensure minimum samples per CPT
    for cpt in target_cpts:
        if cpt_sample_counts[cpt] < MIN_SAMPLES_PER_CPT:
            cpt_records = df[df['Procedure'] == cpt].index.tolist()
            needed = MIN_SAMPLES_PER_CPT - cpt_sample_counts[cpt]
            for idx in cpt_records[:needed]:
                selected_indices.add(idx)
    
    print(f"   After min samples check: {len(selected_indices):,} records")
    
    # Step 3: Ensure ICD coverage by including records with rare but important ICDs
    current_icds = set()
    for idx in selected_indices:
        icd_str = str(df.loc[idx, 'ICD10 - Diagnosis'])
        for code in icd_str.split(','):
            current_icds.add(code.strip())
    
    missing_icds = target_icds - current_icds
    print(f"   Missing target ICDs: {len(missing_icds)}")
    
    # Find records with missing ICDs
    for idx, row in df.iterrows():
        if idx in selected_indices:
            continue
        icd_str = str(row.get('ICD10 - Diagnosis', ''))
        record_icds = set(c.strip() for c in icd_str.split(','))
        if record_icds & missing_icds:
            selected_indices.add(idx)
            current_icds.update(record_icds)
            missing_icds -= record_icds
        if not missing_icds:
            break
    
    print(f"   After ICD coverage: {len(selected_indices):,} records")
    print(f"   Final missing ICDs: {len(missing_icds)}")
    
    # Create final dataset
    result_df = df.loc[list(selected_indices)].copy()
    
    return result_df


def calculate_coverage(df: pd.DataFrame, stats: dict) -> dict:
    """Calculate actual coverage achieved."""
    
    # CPT coverage
    dataset_cpts = set(df['Procedure'].unique())
    target_cpts = set(stats['top_cpts'])
    cpt_coverage = len(dataset_cpts & target_cpts) / len(target_cpts)
    
    # ICD coverage
    dataset_icds = set()
    for icd_str in df['ICD10 - Diagnosis'].dropna():
        for code in str(icd_str).split(','):
            dataset_icds.add(code.strip())
    
    target_icds = set(stats['top_icds'])
    icd_coverage = len(dataset_icds & target_icds) / len(target_icds)
    
    return {
        'cpt_coverage': cpt_coverage,
        'icd_coverage': icd_coverage,
        'unique_cpts_in_dataset': len(dataset_cpts),
        'unique_icds_in_dataset': len(dataset_icds),
        'target_cpts_covered': len(dataset_cpts & target_cpts),
        'target_icds_covered': len(dataset_icds & target_icds)
    }


def main():
    print("=" * 70)
    print("CREATING DIVERSE TRAINING DATASET")
    print("=" * 70)
    
    # Paths
    project_dir = Path(__file__).parent.parent.parent.parent
    data_path = project_dir / "Training_Data_True_Source_20251231" / "Raw_Data_Consolidated.csv"
    output_dir = project_dir / "training" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n1. Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"   Total records: {len(df):,}")
    
    # Parse dates
    print(f"\n2. Filtering by date (>= {MIN_DATE})...")
    df['date_parsed'] = pd.to_datetime(df['Service Date'], errors='coerce')
    min_date = pd.Timestamp(MIN_DATE)
    
    df_filtered = df[df['date_parsed'] >= min_date].copy()
    print(f"   Records after date filter: {len(df_filtered):,}")
    print(f"   Skipped records: {len(df) - len(df_filtered):,}")
    
    # Filter valid records (have CPT, ICD, and Report)
    df_valid = df_filtered[
        df_filtered['Report'].notna() &
        (df_filtered['Report'].str.len() > 200) &
        df_filtered['Procedure'].notna() &
        (df_filtered['Procedure'] != 'nan') &
        df_filtered['ICD10 - Diagnosis'].notna() &
        (df_filtered['ICD10 - Diagnosis'] != 'nan')
    ].copy()
    print(f"   Valid records (with CPT, ICD, Report): {len(df_valid):,}")
    
    # Analyze code distribution
    print(f"\n3. Analyzing code distribution...")
    stats = analyze_code_distribution(df_valid)
    print(f"   Unique CPT codes: {stats['unique_cpts']}")
    print(f"   CPTs needed for 95% coverage: {stats['cpts_for_95']}")
    print(f"   Unique ICD codes: {stats['unique_icds']}")
    print(f"   ICDs needed for 80% coverage: {stats['icds_for_80']}")
    
    # Create diverse dataset
    print(f"\n4. Creating diverse dataset...")
    training_df = create_diverse_dataset(df_valid, stats)
    
    # Calculate coverage
    print(f"\n5. Calculating coverage...")
    coverage = calculate_coverage(training_df, stats)
    print(f"   CPT coverage: {coverage['cpt_coverage']:.1%}")
    print(f"   ICD coverage: {coverage['icd_coverage']:.1%}")
    
    # Shuffle and save
    print(f"\n6. Saving dataset...")
    training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    dataset_path = output_dir / "training_dataset.csv"
    training_df.to_csv(dataset_path, index=False)
    print(f"   Saved: {dataset_path}")
    print(f"   Records: {len(training_df):,}")
    
    # Save stats
    stats_output = {
        'created_at': datetime.now().isoformat(),
        'source_file': str(data_path),
        'min_date_filter': MIN_DATE,
        'original_records': len(df),
        'after_date_filter': len(df_filtered),
        'valid_records': len(df_valid),
        'final_dataset_size': len(training_df),
        'coverage': coverage,
        'targets': {
            'cpt_coverage_target': CPT_COVERAGE_TARGET,
            'icd_coverage_target': ICD_COVERAGE_TARGET,
            'target_cpt_count': len(stats['top_cpts']),
            'target_icd_count': len(stats['top_icds'])
        },
        'top_cpts': stats['top_cpts'][:50],
        'top_icds': stats['top_icds'][:100]
    }
    
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_output, f, indent=2)
    print(f"   Saved: {stats_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Records: {len(training_df):,}")
    print(f"CPT Coverage: {coverage['cpt_coverage']:.1%} ({coverage['target_cpts_covered']}/{len(stats['top_cpts'])} codes)")
    print(f"ICD Coverage: {coverage['icd_coverage']:.1%} ({coverage['target_icds_covered']}/{len(stats['top_icds'])} codes)")
    
    # Show CPT distribution in final dataset
    print(f"\nTop 10 CPT codes in dataset:")
    cpt_dist = training_df['Procedure'].value_counts()
    for cpt, count in cpt_dist.head(10).items():
        print(f"  {cpt}: {count:,} ({100*count/len(training_df):.1f}%)")
    
    return training_df


if __name__ == "__main__":
    main()

