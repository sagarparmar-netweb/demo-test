#!/usr/bin/env python3
"""
Quick Start Script
==================

Sets up the pipeline with RAG index from training data.

Usage:
    python scripts/quick_start.py --data path/to/data.csv --token your_hf_token
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="Quick start NYXMed pipeline")
    parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to training data CSV (optional, for building index)"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run test after setup"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("NYXMed Quick Start")
    print("=" * 60)
    
    # Set token
    if args.token:
        os.environ["HF_API_TOKEN"] = args.token
        print(f"✓ HF token set")
    elif os.getenv("HF_API_TOKEN"):
        print(f"✓ HF token found in environment")
    else:
        print("⚠ No HF token provided. Set with --token or HF_API_TOKEN env var")
    
    # Build index if data provided
    if args.data:
        print(f"\nBuilding RAG index from {args.data}...")
        
        import pandas as pd
        from src.rag_retriever import RAGRetriever
        from config import INDICES_DIR
        
        df = pd.read_csv(args.data, low_memory=False)
        df = df[
            df['Report'].notna() &
            df['Procedure'].notna() &
            df['ICD10 - Diagnosis'].notna()
        ]
        
        print(f"Found {len(df)} valid records")
        
        documents = []
        for _, row in df.iterrows():
            documents.append({
                "report": str(row["Report"]),
                "cpt_code": str(row["Procedure"]),
                "icd_codes": str(row["ICD10 - Diagnosis"]),
                "modifier": str(row.get("Modifier", ""))
            })
        
        retriever = RAGRetriever()
        retriever.build_index(documents, save_dir=str(INDICES_DIR))
        print(f"✓ Index built with {len(documents)} documents")
    
    # Test if requested
    if args.test:
        print("\nRunning test...")
        from src.pipeline import NYXMedPipeline
        
        pipeline = NYXMedPipeline(verbose=True)
        
        test_report = """
EXAM: CT CHEST WITH CONTRAST
INDICATION: Shortness of breath, rule out PE
FINDINGS: No pulmonary embolism. Mild emphysema.
IMPRESSION: No PE. Mild emphysema.
"""
        
        result = pipeline.predict(test_report)
        
        print(f"\nTest Result:")
        print(f"  CPT: {result.cpt_code}")
        print(f"  Modifier: {result.modifier}")
        print(f"  ICD: {', '.join(result.icd_codes)}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Latency: {result.latency_ms:.0f}ms")
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Set HF_API_TOKEN environment variable")
    print("  2. Run: python scripts/test_pipeline.py")
    print("  3. Use in your code:")
    print("     from src.pipeline import NYXMedPipeline")
    print("     pipeline = NYXMedPipeline()")
    print("     result = pipeline.predict(report)")


if __name__ == "__main__":
    main()

