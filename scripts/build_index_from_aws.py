#!/usr/bin/env python3
"""
Build RAG Index from AWS S3 Data
=================================

Downloads the latest consolidated data from AWS S3 and builds
FAISS + BM25 indices for the VFinal pipeline.

Features:
- Uses `Parsed_Compact` for document text (69% token savings)
- Stores `Parsed_Indication`, `Parsed_Impression` for few-shot examples
- Includes all 60k+ records with enriched metadata

Usage:
    # Build from AWS (downloads automatically)
    python scripts/build_index_from_aws.py
    
    # Build from local file
    python scripts/build_index_from_aws.py --local /path/to/data.csv
    
    # Quick test with fewer documents
    python scripts/build_index_from_aws.py --max-docs 5000
"""

import sys
import os
import argparse
import pickle
import tempfile
from pathlib import Path
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

# RAG dependencies
try:
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False
    print("Error: RAG dependencies not installed. Run:")
    print("  pip install faiss-cpu rank_bm25 sentence-transformers")
    sys.exit(1)

from config import INDICES_DIR, PROJECT_ROOT

# AWS Configuration
AWS_BUCKET = "nyxmed-data-backup-20251229"
AWS_KEY = "consolidated_data/Raw_Data_Consolidated_LATEST.csv"

# Embedding model
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m"


def download_from_s3(bucket: str, key: str, local_path: str) -> bool:
    """Download file from S3."""
    try:
        import boto3
        s3 = boto3.client('s3')
        print(f"Downloading s3://{bucket}/{key}...")
        s3.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        print("\nMake sure you have AWS credentials configured:")
        print("  aws configure")
        print("Or set environment variables:")
        print("  export AWS_ACCESS_KEY_ID=xxx")
        print("  export AWS_SECRET_ACCESS_KEY=xxx")
        return False


def build_index(
    data_path: str,
    output_dir: str,
    max_docs: int = None,
    use_parsed_compact: bool = True
) -> bool:
    """
    Build FAISS + BM25 indices from training data.
    
    Args:
        data_path: Path to CSV file with training data
        output_dir: Directory to save indices
        max_docs: Maximum documents to index (None = all)
        use_parsed_compact: Use Parsed_Compact instead of full Report
    
    Returns:
        True if successful
    """
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Filter valid records
    df = df[
        df['Report'].notna() &
        df['Procedure'].notna() &
        df['ICD10 - Diagnosis'].notna()
    ]
    
    print(f"Found {len(df)} valid records")
    
    # Check for enriched columns
    has_parsed_compact = 'Parsed_Compact' in df.columns
    has_parsed_indication = 'Parsed_Indication' in df.columns
    has_parsed_impression = 'Parsed_Impression' in df.columns
    
    print(f"\nEnriched columns available:")
    print(f"  Parsed_Compact: {'✓' if has_parsed_compact else '✗'}")
    print(f"  Parsed_Indication: {'✓' if has_parsed_indication else '✗'}")
    print(f"  Parsed_Impression: {'✓' if has_parsed_impression else '✗'}")
    
    if max_docs:
        df = df.head(max_docs)
        print(f"\nLimited to {max_docs} documents for testing")
    
    # Prepare documents
    print("\nPreparing documents...")
    documents = []
    metadata = []
    
    for _, row in df.iterrows():
        # Choose document text
        if use_parsed_compact and has_parsed_compact and pd.notna(row.get('Parsed_Compact')):
            doc_text = str(row['Parsed_Compact'])
        else:
            doc_text = str(row['Report'])[:2000]  # Limit full report
        
        documents.append(doc_text)
        
        # Build rich metadata for few-shot examples
        meta = {
            'cpt_code': str(row['Procedure']),
            'icd_codes': str(row['ICD10 - Diagnosis']),
            'modifier': str(row.get('Modifier', '')) if pd.notna(row.get('Modifier')) else '',
        }
        
        # Add parsed sections if available
        if has_parsed_indication and pd.notna(row.get('Parsed_Indication')):
            meta['indication'] = str(row['Parsed_Indication'])[:500]
        
        if has_parsed_impression and pd.notna(row.get('Parsed_Impression')):
            meta['impression'] = str(row['Parsed_Impression'])[:500]
        
        if has_parsed_compact and pd.notna(row.get('Parsed_Compact')):
            meta['compact'] = str(row['Parsed_Compact'])[:800]
        
        # Add full report snippet for reference
        meta['report_snippet'] = str(row['Report'])[:500]
        
        metadata.append(meta)
    
    print(f"Prepared {len(documents)} documents")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build BM25 index
    print("\nBuilding BM25 index...")
    tokenized_docs = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    
    bm25_path = output_path / "nyxmed_60k_bm25.pkl"
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"  Saved: {bm25_path}")
    
    # Build FAISS index
    print("\nBuilding FAISS index (this may take a few minutes)...")
    print(f"  Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL)
    
    print(f"  Encoding {len(documents)} documents...")
    embeddings = embed_model.encode(
        documents, 
        show_progress_bar=True,
        batch_size=32
    )
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
    index.add(embeddings)
    
    faiss_path = output_path / "nyxmed_60k_faiss.index"
    faiss.write_index(index, str(faiss_path))
    print(f"  Saved: {faiss_path}")
    
    # Save metadata
    print("\nSaving metadata...")
    meta_path = output_path / "nyxmed_60k_metadata.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump({
            "documents": documents,
            "metadata": metadata,
            "version": "60k_enriched",
            "embed_model": EMBED_MODEL,
            "use_parsed_compact": use_parsed_compact
        }, f)
    print(f"  Saved: {meta_path}")
    
    # Calculate stats
    print("\n" + "=" * 60)
    print("INDEX BUILD COMPLETE")
    print("=" * 60)
    print(f"\nDocuments indexed: {len(documents)}")
    print(f"Embedding dimension: {dimension}")
    print(f"Index files saved to: {output_path}")
    print(f"\nFiles created:")
    print(f"  - nyxmed_60k_bm25.pkl")
    print(f"  - nyxmed_60k_faiss.index")
    print(f"  - nyxmed_60k_metadata.pkl")
    
    # Sample few-shot example
    print("\n" + "=" * 60)
    print("SAMPLE FEW-SHOT EXAMPLE")
    print("=" * 60)
    sample = metadata[0]
    print(f"\nCPT: {sample['cpt_code']}")
    print(f"ICD: {sample['icd_codes']}")
    print(f"Modifier: {sample['modifier']}")
    if 'indication' in sample:
        print(f"\nIndication: {sample['indication'][:200]}...")
    if 'impression' in sample:
        print(f"\nImpression: {sample['impression'][:200]}...")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG index from AWS S3 data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from AWS (downloads latest data automatically)
  python build_index_from_aws.py
  
  # Build from local CSV file
  python build_index_from_aws.py --local /path/to/data.csv
  
  # Quick test with 5000 documents
  python build_index_from_aws.py --max-docs 5000
  
  # Use full report text instead of Parsed_Compact
  python build_index_from_aws.py --no-compact
        """
    )
    parser.add_argument(
        "--local", "-l",
        type=str,
        default=None,
        help="Use local CSV file instead of downloading from AWS"
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
    parser.add_argument(
        "--no-compact",
        action="store_true",
        help="Use full Report text instead of Parsed_Compact"
    )
    
    args = parser.parse_args()
    
    # Determine data source
    if args.local:
        data_path = args.local
        if not Path(data_path).exists():
            print(f"Error: File not found: {data_path}")
            sys.exit(1)
    else:
        # Download from AWS
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            data_path = tmp.name
        
        if not download_from_s3(AWS_BUCKET, AWS_KEY, data_path):
            sys.exit(1)
        
        # Show file size
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"Downloaded: {size_mb:.1f} MB")
    
    # Build index
    success = build_index(
        data_path=data_path,
        output_dir=args.output,
        max_docs=args.max_docs,
        use_parsed_compact=not args.no_compact
    )
    
    # Cleanup temp file
    if not args.local:
        try:
            os.unlink(data_path)
        except:
            pass
    
    if not success:
        sys.exit(1)
    
    print("\n✓ Done! Update rag_retriever.py to use 'nyxmed_60k' prefix.")


if __name__ == "__main__":
    main()

