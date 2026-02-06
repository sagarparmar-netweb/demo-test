#!/usr/bin/env python3
"""
Rebuild RAG Index with Latest Data

This script rebuilds the BM25 and FAISS indices using the latest
consolidated data (53K+ records).

Run this when new data is added to the training dataset.
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Paths
PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_DIR / "data" / "consolidated_cleaned.csv"
OUTPUT_DIR = PROJECT_DIR / "VFinal" / "indices"
INDEX_PREFIX = "nyxmed_71k"  # Updated for 71K records with parsed fields

# Model
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m"


def main():
    print("=" * 70)
    print("REBUILDING RAG INDEX")
    print("=" * 70)
    
    # 1. Load data
    print(f"\n1. Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    # Filter to valid records
    valid = df[
        df['Report'].notna() &
        (df['Report'].str.len() > 100) &
        df['Procedure'].notna() &
        df['ICD10 - Diagnosis'].notna()
    ].copy()
    
    print(f"   Total records: {len(df)}")
    print(f"   Valid records: {len(valid)}")
    
    # 2. Prepare documents and metadata
    print("\n2. Preparing documents and metadata...")
    documents = []
    metadata = []
    
    for idx, row in tqdm(valid.iterrows(), total=len(valid)):
        # Document = EXAM DESCRIPTION + Parsed_Compact for proper matching
        # This allows RAG to match on BOTH procedure type AND clinical content
        exam_desc = str(row.get('Exam Description', ''))
        parsed_compact = str(row.get('Parsed_Compact', ''))
        
        # Build document with exam description at the start
        doc_parts = []
        if exam_desc and exam_desc not in ['nan', 'None', '']:
            doc_parts.append(f"EXAM: {exam_desc}")
        
        if parsed_compact and len(parsed_compact) > 50:
            doc_parts.append(parsed_compact)
        else:
            # Fallback to full report if no parsed compact
            doc_parts.append(str(row['Report'])[:1500])
        
        doc = '\n'.join(doc_parts)
        documents.append(doc)
        
        # Metadata - include parsed fields for few-shot examples
        meta = {
            'id': row.get('MRN', idx),  # Use MRN if available, else index
            'original_index': idx,
            'gt_cpt': str(row['Procedure']),
            'gt_icd': str(row['ICD10 - Diagnosis']),
            'gt_codes': str(row['ICD10 - Diagnosis']),  # Alias
            'cpt_code': str(row['Procedure']),  # Alias
            'modifier': str(row.get('Modifier', '')),
            # Parsed fields for rich few-shot examples
            'indication': str(row.get('Parsed_Indication', '')),
            'impression': str(row.get('Parsed_Impression', '')),
            'parsed_compact': parsed_compact,
            'laterality': str(row.get('Parsed_Laterality', '')),
            'exam_description': str(row.get('Exam Description', '')),
        }
        metadata.append(meta)
    
    print(f"   Documents: {len(documents)}")
    
    # 3. Build BM25 index
    print("\n3. Building BM25 index...")
    tokenized_docs = [doc.lower().split() for doc in tqdm(documents)]
    bm25 = BM25Okapi(tokenized_docs)
    
    bm25_path = OUTPUT_DIR / f"{INDEX_PREFIX}_bm25.pkl"
    with open(bm25_path, 'wb') as f:
        pickle.dump(bm25, f)
    print(f"   Saved: {bm25_path}")
    
    # 4. Build FAISS index
    print("\n4. Building FAISS vector index...")
    print("   Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    
    print("   Encoding documents (this may take a while)...")
    batch_size = 64
    all_embeddings = []
    
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i:i+batch_size]
        embeddings = embed_model.encode(batch, show_progress_bar=False)
        all_embeddings.append(embeddings)
    
    embeddings = np.vstack(all_embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    index.add(embeddings)
    
    faiss_path = OUTPUT_DIR / f"{INDEX_PREFIX}_faiss.index"
    faiss.write_index(index, str(faiss_path))
    print(f"   Saved: {faiss_path}")
    
    # 5. Save metadata
    print("\n5. Saving metadata...")
    meta_path = OUTPUT_DIR / f"{INDEX_PREFIX}_metadata.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump({
            'documents': documents,
            'metadata': metadata
        }, f)
    print(f"   Saved: {meta_path}")
    
    # 6. Summary
    print("\n" + "=" * 70)
    print("RAG INDEX REBUILT SUCCESSFULLY")
    print("=" * 70)
    print(f"Total documents indexed: {len(documents)}")
    print(f"Index prefix: {INDEX_PREFIX}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nTo use the new index, update rag_retriever.py:")
    print(f'  INDEX_PREFIX = "{INDEX_PREFIX}"')


if __name__ == "__main__":
    main()

