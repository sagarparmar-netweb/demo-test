#!/usr/bin/env python3
"""
Build RAG Index
===============

Builds the FAISS + BM25 indices from training data.
Run this once before using the pipeline.

Usage:
    python scripts/build_index.py --data path/to/data.csv
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.rag_retriever import RAGRetriever
from config import INDICES_DIR


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from training data")
    parser.add_argument(
        "--data", "-d",
        type=str,
        required=True,
        help="Path to CSV file with columns: Report, Procedure, ICD10 - Diagnosis, Modifier"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(INDICES_DIR),
        help=f"Output directory for indices (default: {INDICES_DIR})"
    )
    parser.add_argument(
        "--max-docs", "-n",
        type=int,
        default=None,
        help="Maximum number of documents to index (default: all)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data, low_memory=False)
    
    # Filter valid records
    df = df[
        df['Report'].notna() &
        df['Procedure'].notna() &
        df['ICD10 - Diagnosis'].notna()
    ]
    
    if args.max_docs:
        df = df.head(args.max_docs)
    
    print(f"Found {len(df)} valid records")
    
    # Prepare documents
    documents = []
    for _, row in df.iterrows():
        documents.append({
            "report": str(row["Report"]),
            "cpt_code": str(row["Procedure"]),
            "icd_codes": str(row["ICD10 - Diagnosis"]),
            "modifier": str(row.get("Modifier", ""))
        })
    
    # Build index
    print(f"\nBuilding index...")
    retriever = RAGRetriever()
    success = retriever.build_index(documents, save_dir=args.output)
    
    if success:
        print(f"\n✓ Index built successfully!")
        print(f"  Saved to: {args.output}")
        print(f"  Documents: {len(documents)}")
    else:
        print("\n✗ Failed to build index")
        sys.exit(1)


if __name__ == "__main__":
    main()

