#!/usr/bin/env python3
import re
"""
RAG RETRIEVER for VFinal Pipeline
==================================

Uses hybrid BM25 + FAISS vector search to find similar cases
and extract CPT/ICD candidates + few-shot examples.

Features:
- Uses `Parsed_Compact` for efficient retrieval (69% token savings)
- Returns `indication` and `impression` for high-quality few-shot examples
- Supports both 36k (old) and 60k (new enriched) indices

Build new indices:
    python scripts/build_index_from_aws.py
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

import numpy as np

# Try to import the heavy dependencies
try:
    import faiss
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False
    print("Warning: RAG dependencies not installed. Run: pip install faiss-cpu rank_bm25 sentence-transformers")


# Paths - check multiple locations for indices
PROJECT_DIR = Path(__file__).parent.parent.parent
VFINAL_INDICES = Path(__file__).parent.parent / "indices"
LEGACY_INDICES = PROJECT_DIR / "versions" / "v7.5_hybrid_rag_deployment"

# Index prefixes in order of preference (newest first)
INDEX_PREFIXES = ["nyxmed_71k", "nyxmed_60k", "nyxmed_53k", "nyxmed_36k"]

# Model names
EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def find_index_files() -> Tuple[Optional[Path], Optional[str]]:
    """
    Find available index files.
    
    Returns:
        Tuple of (index_directory, index_prefix) or (None, None) if not found
    """
    search_paths = [VFINAL_INDICES, LEGACY_INDICES]
    
    for search_dir in search_paths:
        if not search_dir.exists():
            continue
        
        for prefix in INDEX_PREFIXES:
            bm25_path = search_dir / f"{prefix}_bm25.pkl"
            faiss_path = search_dir / f"{prefix}_faiss.index"
            meta_path = search_dir / f"{prefix}_metadata.pkl"
            
            if all(p.exists() for p in [bm25_path, faiss_path, meta_path]):
                return search_dir, prefix
    
    return None, None


class RAGRetriever:
    """
    Retrieves similar cases using hybrid BM25 + vector search.
    
    Features:
    - Hybrid search (BM25 + dense vectors)
    - Optional cross-encoder reranking
    - Returns CPT and ICD candidates from similar cases
    - Formats few-shot examples with indication/impression
    """
    
    def __init__(
        self, 
        index_dir: str = None, 
        index_prefix: str = None,
        use_reranker: bool = True  # Cross-encoder reranking is REQUIRED in production
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            index_dir: Path to RAG index files (auto-detected if None)
            index_prefix: Index file prefix (auto-detected if None)
            use_reranker: Whether to use cross-encoder reranking (slower but better)
        """
        self.use_reranker = use_reranker
        self.loaded = False
        
        if not HAS_RAG_DEPS:
            print("RAG dependencies not available")
            return
        
        # Auto-detect index location
        if index_dir is None or index_prefix is None:
            detected_dir, detected_prefix = find_index_files()
            if detected_dir is None:
                print(f"No index files found. Run: python scripts/build_index_from_aws.py")
                return
            self.index_dir = detected_dir
            self.index_prefix = detected_prefix
        else:
            self.index_dir = Path(index_dir)
            self.index_prefix = index_prefix
        
        self._load_indices()
    
    def _load_indices(self):
        """Load all RAG indices and models."""
        print(f"Loading RAG indices from {self.index_dir}...")
        print(f"  Index prefix: {self.index_prefix}")
        
        bm25_path = self.index_dir / f"{self.index_prefix}_bm25.pkl"
        faiss_path = self.index_dir / f"{self.index_prefix}_faiss.index"
        meta_path = self.index_dir / f"{self.index_prefix}_metadata.pkl"
        
        if not all(p.exists() for p in [bm25_path, faiss_path, meta_path]):
            print(f"Index files not found")
            return
        
        # Load BM25
        with open(bm25_path, "rb") as f:
            self.bm25 = pickle.load(f)
        
        # Load FAISS
        self.faiss_index = faiss.read_index(str(faiss_path))
        
        # Load Metadata
        with open(meta_path, "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.index_version = data.get("version", "unknown")
        
        # Load embedding model
        print("  Loading embedding model...")
        self.embed_model = SentenceTransformer(EMBED_MODEL)
        
        # Load reranker (optional)
        if self.use_reranker:
            print("  Loading reranker model...")
            self.reranker = CrossEncoder(RERANK_MODEL, max_length=512)
        else:
            self.reranker = None
        
        self.loaded = True
        print(f"  ✓ RAG loaded: {len(self.documents)} documents (version: {self.index_version})")
    
    def search(
        self, 
        query_text: str, 
        top_k: int = 5, 
        exclude_text: str = None
    ) -> List[Dict]:
        """
        Search for similar cases.
        
        Args:
            query_text: The report text to search for
            top_k: Number of results to return
            exclude_text: If provided, exclude documents that match this text
        
        Returns:
            List of dicts with 'score', 'content', 'metadata' keys
        """
        if not self.loaded:
            return []
        
        # A. BM25 sparse search
        tokenized_query = query_text.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_indices = np.argsort(bm25_scores)[::-1][:100]
        
        # B. Vector dense search
        query_vec = self.embed_model.encode([query_text])
        faiss.normalize_L2(query_vec)
        distances, vector_indices = self.faiss_index.search(query_vec, 100)
        vector_indices = vector_indices[0]
        
        # C. Combine candidates
        all_indices = set(bm25_indices) | set(vector_indices)
        candidates = list(all_indices)
        
        # D. Filter out exact matches
        if exclude_text:
            exclude_normalized = exclude_text[:500].lower().strip()
            filtered_candidates = []
            for idx in candidates:
                doc_normalized = self.documents[idx][:500].lower().strip()
                if doc_normalized != exclude_normalized:
                    filtered_candidates.append(idx)
            candidates = filtered_candidates
        
        # E. Rerank or use BM25 scores
        if self.reranker and len(candidates) > 0:
            pairs = [[query_text, self.documents[idx]] for idx in candidates]
            rerank_scores = self.reranker.predict(pairs)
            sorted_indices = np.argsort(rerank_scores)[::-1][:top_k]
            
            results = []
            for rank_idx in sorted_indices:
                original_idx = candidates[rank_idx]
                results.append({
                    "score": float(rerank_scores[rank_idx]),
                    "content": self.documents[original_idx],
                    "metadata": self.metadata[original_idx]
                })
        else:
            # Fallback: use BM25 scores
            scored = [(idx, bm25_scores[idx]) for idx in candidates]
            scored.sort(key=lambda x: -x[1])
            
            results = []
            for idx, score in scored[:top_k]:
                results.append({
                    "score": float(score),
                    "content": self.documents[idx],
                    "metadata": self.metadata[idx]
                })
        
        return results
    
    def get_candidates(
        self, 
        report_text: str, 
        top_k: int = 5
    ) -> Tuple[List[str], List[str]]:
        """
        Get CPT and ICD candidates from similar cases.
        
        Args:
            report_text: The report to find candidates for
            top_k: Number of similar cases to retrieve
        
        Returns:
            Tuple of (cpt_candidates, icd_candidates)
        """
        if not self.loaded:
            return [], []
        
        hits = self.search(report_text, top_k=top_k)
        
        cpt_codes = []
        icd_codes = []
        
        for hit in hits:
            meta = hit['metadata']
            
            # Extract CPT
            cpt = meta.get('cpt_code') or meta.get('gt_cpt') or meta.get('Procedure')
            if cpt and str(cpt) not in ['nan', 'None', '']:
                cpt_codes.append(str(cpt))
            
            # Extract ICD codes
            icd_str = meta.get('icd_codes') or meta.get('gt_codes') or meta.get('ICD10 - Diagnosis') or ''
            if icd_str and str(icd_str) not in ['nan', 'None', '']:
                for code in str(icd_str).split(','):
                    code = code.strip()
                    if code:
                        icd_codes.append(code)
        
        # Count and sort by frequency
        cpt_counter = Counter(cpt_codes)
        icd_counter = Counter(icd_codes)
        
        unique_cpts = [c for c, _ in cpt_counter.most_common()]
        unique_icds = [c for c, _ in icd_counter.most_common()]
        
        return unique_cpts, unique_icds
    
    def format_examples(
        self, 
        hits: List[Dict], 
        n: int = 3,
        use_parsed_sections: bool = True
    ) -> str:
        """
        Format similar cases as few-shot examples.
        
        Uses the enriched metadata (indication, impression) when available
        to create production-quality few-shot examples.
        
        Args:
            hits: Search results from self.search()
            n: Number of examples to include
            use_parsed_sections: Whether to use parsed indication/impression
        
        Returns:
            Formatted string of examples
        """
        examples = []
        
        for hit in hits:
            if len(examples) >= n:
                break
                
            meta = hit['metadata']
            
            # Get codes - use gt_cpt/gt_icd which have 100% coverage
            cpt = str(meta.get('gt_cpt') or meta.get('cpt_code') or '')
            icd = str(meta.get('gt_icd') or meta.get('icd_codes') or '')
            modifier = str(meta.get('modifier') or '')
            
            # SKIP invalid CPT codes (5 digits or G-codes)
            if not cpt or cpt in ['nan', 'None', '']:
                continue
            if not (cpt.isdigit() and len(cpt) == 5) and not cpt.upper().startswith('G'):
                continue
            
            # Build report content - USE parsed_compact which has ALL sections
            # (INDICATION + LATERALITY + IMPRESSION + FINDINGS)
            if use_parsed_sections:
                # PRIORITY 1: Use parsed_compact - it has INDICATION, IMPRESSION, FINDINGS, LATERALITY
                parsed_compact = meta.get('parsed_compact')
                if parsed_compact and str(parsed_compact) not in ['nan', 'None', '']:
                    report_content = re.sub(r'-{3,}', '', str(parsed_compact))[:700]
                else:
                    # FALLBACK: Build from individual fields
                    parts = []
                    
                    # Indication
                    indication = meta.get('indication')
                    if indication and str(indication) not in ['nan', 'None', '']:
                        parts.append(f"INDICATION: {indication}")
                    
                    # Laterality
                    laterality = meta.get('laterality')
                    if laterality and str(laterality) not in ['nan', 'None', '', 'none']:
                        parts.append(f"LATERALITY: {laterality.upper()}")
                    
                    # Impression
                    impression = meta.get('impression')
                    if impression and str(impression) not in ['nan', 'None', '']:
                        parts.append(f"IMPRESSION: {impression}")
                    
                    if parts:
                        report_content = '\n'.join(parts)
                    elif meta.get('compact'):
                        report_content = meta['compact'][:600]
                    else:
                        report_content = meta.get('report_snippet', hit['content'][:400])
            else:
                report_content = hit['content'][:400]
            
            # Format output line - MATCH TRAINING DATA FORMAT
            # Training format: CPT, MODIFIER, ICD1, ICD2, ...
            # Regular CPT codes should have modifier 26 (professional)
            mod = modifier if modifier and modifier not in ['nan', 'None', ''] else ''
            is_g_code = cpt.upper().startswith('G')
            
            if mod:
                output = f"{cpt}, {mod}, {icd}"
            elif not is_g_code:
                # Default to 26 for regular CPT codes
                output = f"{cpt}, 26, {icd}"
            else:
                output = f"{cpt}, {icd}"
            
            examples.append(f"Report: {report_content}\nOutput: {output}")
        
        return '\n---\n'.join(examples)
    
    def get_similar_with_examples(
        self, 
        report_text: str, 
        top_k: int = 5,
        return_scores: bool = False
    ) -> Tuple[List[str], List[str], str, float]:
        """
        Get CPT candidates, ICD candidates, formatted few-shot examples, and match score.
        
        Convenience method that combines get_candidates and format_examples.
        
        Args:
            report_text: The report to process
            top_k: Number of similar cases to use
            return_scores: If True, returns 4 values including top match score
        
        Returns:
            Tuple of (cpt_candidates, icd_candidates, examples_string, top_match_score)
            top_match_score is the highest RAG similarity score (0-100 scale)
        """
        if not self.loaded:
            return [], [], "", 0.0
        
        # Get more hits to filter out invalid CPTs
        # NOTE: Don't use exclude_text with structured queries - it's too aggressive
        # and filters out valid matches. The query is already a compact representation.
        hits = self.search(report_text, top_k=top_k * 3)
        
        # Extract top match score (convert to 0-100 scale)
        # Reranker scores are typically 0-1, multiply by 100 for percentage
        top_match_score = 0.0
        if hits:
            raw_score = hits[0].get('score', 0)
            # Convert to 0-100 scale
            top_match_score = min(100.0, max(0.0, raw_score * 100))
        
        # Get candidates - only valid 5-digit CPT codes
        cpt_codes = []
        icd_codes = []
        
        for hit in hits:
            meta = hit['metadata']
            
            # Use gt_cpt which has 100% coverage
            cpt = str(meta.get('gt_cpt') or meta.get('cpt_code') or '')
            # Only include valid 5-digit CPT codes
            if cpt.isdigit() and len(cpt) == 5:
                cpt_codes.append(cpt)
            
            # Use gt_icd which has 100% coverage
            icd_str = str(meta.get('gt_icd') or meta.get('icd_codes') or '')
            if icd_str and icd_str not in ['nan', 'None', '']:
                for code in icd_str.split(','):
                    code = code.strip()
                    if code:
                        icd_codes.append(code)
        
        cpt_counter = Counter(cpt_codes)
        icd_counter = Counter(icd_codes)
        
        unique_cpts = [c for c, _ in cpt_counter.most_common()]
        unique_icds = [c for c, _ in icd_counter.most_common()]
        
        # Format examples - pass more hits so we can skip invalid ones
        examples = self.format_examples(hits, n=3, use_parsed_sections=True)
        
        return unique_cpts, unique_icds, examples, top_match_score


# Singleton instance for reuse
_retriever_instance = None


def get_rag_retriever(use_reranker: bool = False) -> Optional[RAGRetriever]:
    """
    Get or create a RAG retriever instance.
    
    Args:
        use_reranker: Whether to use cross-encoder reranking (slower)
    
    Returns:
        RAGRetriever instance or None if not available
    """
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever(use_reranker=True)
    else:
        # If an instance was created without reranking, rebuild it with reranking.
        # (This prevents silently running without the CrossEncoder.)
        if getattr(_retriever_instance, "use_reranker", False) is not True:
            _retriever_instance = RAGRetriever(use_reranker=True)
    
    if _retriever_instance.loaded:
        return _retriever_instance
    
    return None


if __name__ == "__main__":
    # Test the retriever
    print("Testing RAG Retriever")
    print("=" * 60)
    
    retriever = get_rag_retriever(use_reranker=False)
    
    if retriever:
        test_report = """
        EXAM: CT CHEST WITHOUT IV CONTRAST
        CLINICAL HISTORY: 65 year old male with cough and shortness of breath.
        FINDINGS: No acute pulmonary abnormality. Normal heart size. 
                  Mild emphysematous changes in upper lobes.
        IMPRESSION: 1. No acute findings.
                    2. Mild emphysema.
        """
        
        print("\nSearching for similar cases...")
        cpts, icds, examples = retriever.get_similar_with_examples(test_report, top_k=5)
        
        print(f"\nCPT Candidates: {cpts}")
        print(f"ICD Candidates (top 10): {icds[:10]}")
        
        print("\n" + "-" * 60)
        print("FEW-SHOT EXAMPLES:")
        print("-" * 60)
        print(examples)
    else:
        print("\n✗ RAG retriever not available")
        print("  Run: python scripts/build_index_from_aws.py")
