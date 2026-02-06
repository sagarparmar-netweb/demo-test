#!/usr/bin/env python3
"""
V13 PRODUCTION PIPELINE

Main orchestrator that combines:
1. Pre-processing (extract features, candidates, build prompt)
2. LLM prediction (single call for CPT + ICD)
3. Post-processing (modifiers, validation, formatting)

Usage:
    from src.pipeline import ProductionPipeline
    
    pipeline = ProductionPipeline()
    result = pipeline.process(report_text)
    
    print(result.to_dict())
    # {'Procedure': '71046', 'Modifier': '26', 'ICD10 - Diagnosis': 'R91.8, J18.9'}
"""

import os
import json
from pathlib import Path
from typing import Optional
import pandas as pd

from .preprocessor import Preprocessor, PreprocessedData
from .llm_predictor import LLMPredictor, LLMPrediction
from .postprocessor import Postprocessor, FinalResult
from .confidence_scorer import ConfidenceScorer, ConfidenceResult, score_prediction


from dataclasses import dataclass, field
from typing import Dict, List


# Project paths
# __file__ = VFinal/src/pipeline.py
# .parent = VFinal/src/
# .parent.parent = VFinal/
# .parent.parent.parent = NyxMed/ (project root)
PROJECT_DIR = Path(__file__).parent.parent.parent  # NyxMed/
VFINAL_DIR = Path(__file__).parent.parent  # VFinal/
DATA_DIR = PROJECT_DIR / "Training_Data_True_Source_20251231"


@dataclass
class PipelineResult:
    """
    Complete pipeline result with coding AND confidence assessment.
    
    This is what gets sent to the front-end for display and review.
    """
    
    # Coding output
    procedure: str
    modifier: str
    icd10_diagnosis: str
    
    # Confidence assessment
    overall_confidence: float
    needs_review: bool
    auto_submit_ok: bool
    review_flags: List[Dict] = field(default_factory=list)
    
    # Detailed scores
    cpt_confidence: float = 0.0
    icd_confidence: float = 0.0
    modifier_confidence: float = 0.0
    
    # Medical necessity
    has_medical_necessity: bool = True
    medical_necessity_warning: str = ""
    
    def to_dict(self) -> Dict:
        """Return complete result for API/front-end."""
        return {
            # Coding
            'coding': {
                'Procedure': self.procedure,
                'Modifier': self.modifier,
                'ICD10 - Diagnosis': self.icd10_diagnosis
            },
            # Confidence
            'confidence': {
                'overall': round(self.overall_confidence, 2),
                'cpt': round(self.cpt_confidence, 2),
                'icd': round(self.icd_confidence, 2),
                'modifier': round(self.modifier_confidence, 2)
            },
            # Review status
            'review': {
                'needs_review': self.needs_review,
                'auto_submit_ok': self.auto_submit_ok,
                'flags': self.review_flags,
                'medical_necessity': self.has_medical_necessity,
                'warning': self.medical_necessity_warning
            }
        }
    
    def to_simple_dict(self) -> Dict:
        """Return just the coding for backward compatibility."""
        return {
            'Procedure': self.procedure,
            'Modifier': self.modifier,
            'ICD10 - Diagnosis': self.icd10_diagnosis
        }


class ProductionPipeline:
    """
    V13 Production Pipeline for radiology coding.
    
    Architecture:
    1. Pre-processing → Extract features, generate candidates, build prompt
    2. LLM Call → Single call predicts CPT (if needed) + ICD codes
    3. Post-processing → Add modifiers, validate, format output
    """
    
    def __init__(
        self,
        data_path: str = None,
        model: str = "huggingface",  # "huggingface" for fine-tuned model, or "gpt-4o" for OpenAI
        bill_type: str = 'P',
        verbose: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to training data CSV (for candidate generation)
            model: LLM model to use (will be fine-tuned model in production)
            bill_type: 'P' professional, 'T' technical, 'G' global
            verbose: Print progress messages
        """
        self.verbose = verbose
        self.bill_type = bill_type
        
        # Load training data
        if data_path is None:
            data_path = self._find_data_path()
        
        if self.verbose:
            print(f"Loading data: {data_path}")
        
        self.df = pd.read_csv(data_path, low_memory=False)
        
        # Load ICD descriptions and rules
        icd_descriptions = self._load_icd_descriptions()
        rules = self._load_rules()
        
        # Initialize components
        self.preprocessor = Preprocessor(
            df=self.df,
            icd_descriptions=icd_descriptions,
            rules=rules
        )
        
        # Use HuggingFace backend by default (fine-tuned model)
        if model == "huggingface":
            self.llm_predictor = LLMPredictor(backend="huggingface")
        else:
            # Fallback to OpenAI
            self.llm_predictor = LLMPredictor(backend="openai", model=model)
        
        self.postprocessor = Postprocessor(bill_type=bill_type)
        
        self.confidence_scorer = ConfidenceScorer()
        
        if self.verbose:
            print(f"Pipeline ready. CPT mappings: {len(self.preprocessor.cpt_to_icd)}")
    
    def _find_data_path(self) -> str:
        """Find the training data file."""
        candidates = [
            PROJECT_DIR / "data" / "consolidated_cleaned.csv",  # Primary: 71K with parsed fields
            DATA_DIR / "Raw_Data_Consolidated.csv",
            PROJECT_DIR / "Training_Data_True_Source_20251229" / "Raw_Data_Consolidated.csv",
        ]
        
        for path in candidates:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(f"Training data not found. Checked: {[str(p) for p in candidates]}")
    
    def _load_icd_descriptions(self) -> dict:
        """Load ICD code descriptions."""
        desc_path = PROJECT_DIR / "data" / "icd10_official_descriptions.json"
        
        if desc_path.exists():
            with open(desc_path) as f:
                return json.load(f)
        
        return {}
    
    def _load_rules(self) -> dict:
        """Load pattern rules for candidate ranking."""
        rules_path = PROJECT_DIR / "data" / "comprehensive_rules.json"
        
        if rules_path.exists():
            if self.verbose:
                print(f"Loading rules from: {rules_path}")
            with open(rules_path) as f:
                return json.load(f)
        
        return {}
    
    def process(self, report_text: str, cpt_code: str = None) -> PipelineResult:
        """
        Process a radiology report and return coding result with confidence.
        
        Args:
            report_text: Raw radiology report text
            cpt_code: Optional CPT code (if known, skips CPT prediction)
        
        Returns:
            PipelineResult with coding, confidence, and review flags
        """
        # =====================================================================
        # STEP 1: PRE-PROCESSING
        # =====================================================================
        if self.verbose:
            print("Step 1: Pre-processing...")
        
        preprocessed = self.preprocessor.preprocess(report_text)
        
        # Override CPT if provided
        if cpt_code:
            preprocessed.extracted_cpt = cpt_code
            # Re-generate ICD candidates for this CPT
            preprocessed.icd_candidates = self.preprocessor._get_icd_candidates(cpt_code, report_text)
            preprocessed.examples = self.preprocessor._get_examples(cpt_code)
            preprocessed.prompt = self.preprocessor._build_prompt(preprocessed)
        
        if self.verbose:
            print(f"  - CPT Extracted: {preprocessed.extracted_cpt or 'No'}")
            print(f"  - Modality: {preprocessed.modality}")
            print(f"  - Body Part: {preprocessed.body_part}")
            print(f"  - Laterality: {preprocessed.laterality}")
            print(f"  - ICD Candidates: {len(preprocessed.icd_candidates)}")
        
        # =====================================================================
        # STEP 2: LLM PREDICTION
        # =====================================================================
        if self.verbose:
            print("Step 2: LLM prediction...")
        
        # Get valid ICD codes for filtering
        valid_icd_codes = set(code for code, _ in preprocessed.icd_candidates)
        
        # Extract exam description for disambiguation
        import re
        exam_match = re.search(r'^Exam:\s*(.+)$', report_text, re.IGNORECASE | re.MULTILINE)
        exam_desc = exam_match.group(1) if exam_match else ""
        
        prediction = self.llm_predictor.predict(
            prompt=preprocessed.prompt,
            extracted_cpt=preprocessed.extracted_cpt,
            valid_icd_codes=valid_icd_codes,
            report_text=report_text,
            exam_desc=exam_desc
        )
        
        if self.verbose:
            print(f"  - CPT: {prediction.cpt_code}")
            print(f"  - ICD: {prediction.icd_codes}")
        
        # =====================================================================
        # STEP 3: POST-PROCESSING
        # =====================================================================
        if self.verbose:
            print("Step 3: Post-processing...")
        
        coding_result = self.postprocessor.postprocess(
            cpt_code=prediction.cpt_code,
            icd_codes=prediction.icd_codes,
            laterality=preprocessed.laterality,
            cpt_was_extracted=(preprocessed.extracted_cpt is not None),
            valid_icd_codes=valid_icd_codes,
            llm_suggested_modifiers=prediction.modifiers
        )
        
        if self.verbose:
            print(f"  - Coding: {coding_result.to_dict()}")
        
        # =====================================================================
        # STEP 4: CONFIDENCE SCORING
        # =====================================================================
        if self.verbose:
            print("Step 4: Confidence scoring...")
        
        # Get RAG ICD codes for comparison (if available)
        rag_icd_codes = []
        if hasattr(preprocessed, 'rag_candidates') and preprocessed.rag_candidates:
            rag_icd_codes = [c for c, _ in preprocessed.rag_candidates]
        
        confidence = self.confidence_scorer.calculate_confidence(
            cpt_code=coding_result.procedure,
            cpt_was_extracted=(preprocessed.extracted_cpt is not None),
            icd_codes=coding_result.icd10_diagnosis.split(', ') if coding_result.icd10_diagnosis else [],
            modifiers=coding_result.modifier.split() if coding_result.modifier else [],
            cpt_in_rag_candidates=hasattr(preprocessed, 'rag_cpt') and preprocessed.rag_cpt,
            icd_candidates=valid_icd_codes,
            rag_icd_codes=rag_icd_codes,
            laterality_detected=preprocessed.laterality,
            has_medical_necessity=coding_result.has_medical_necessity,
            medical_necessity_warning=coding_result.medical_necessity_warning
        )
        
        if self.verbose:
            print(f"  - Confidence: {confidence.overall_confidence:.0%}")
            print(f"  - Needs Review: {confidence.needs_review}")
            print(f"  - Auto-submit OK: {confidence.auto_submit_ok}")
            if confidence.review_flags:
                print(f"  - Flags: {[f.reason.value for f in confidence.review_flags]}")
        
        # =====================================================================
        # BUILD FINAL RESULT
        # =====================================================================
        return PipelineResult(
            # Coding
            procedure=coding_result.procedure,
            modifier=coding_result.modifier,
            icd10_diagnosis=coding_result.icd10_diagnosis,
            
            # Confidence
            overall_confidence=confidence.overall_confidence,
            cpt_confidence=confidence.cpt_confidence,
            icd_confidence=confidence.icd_confidence,
            modifier_confidence=confidence.modifier_confidence,
            
            # Review status
            needs_review=confidence.needs_review,
            auto_submit_ok=confidence.auto_submit_ok,
            review_flags=[
                {
                    'reason': f.reason.value,
                    'severity': f.severity,
                    'message': f.message,
                    'field': f.field,
                    'details': f.details
                }
                for f in confidence.review_flags
            ],
            
            # Medical necessity
            has_medical_necessity=coding_result.has_medical_necessity,
            medical_necessity_warning=coding_result.medical_necessity_warning
        )
    
    def process_batch(self, reports: list) -> List[PipelineResult]:
        """
        Process multiple reports.
        
        Args:
            reports: List of report texts
        
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        for i, report in enumerate(reports):
            if self.verbose:
                print(f"\n[{i+1}/{len(reports)}]")
            
            try:
                result = self.process(report)
                results.append(result)
            except Exception as e:
                print(f"Error processing report {i+1}: {e}")
                results.append(None)
        
        return results
    
    def get_review_queue(self, results: List[PipelineResult]) -> List[PipelineResult]:
        """
        Filter results that need human review.
        
        Args:
            results: List of pipeline results
        
        Returns:
            List of results that need review (sorted by confidence, lowest first)
        """
        needs_review = [r for r in results if r and r.needs_review]
        return sorted(needs_review, key=lambda r: r.overall_confidence)
    
    def get_auto_submit_queue(self, results: List[PipelineResult]) -> List[PipelineResult]:
        """
        Filter results that can be auto-submitted.
        
        Args:
            results: List of pipeline results
        
        Returns:
            List of results that can be auto-submitted
        """
        return [r for r in results if r and r.auto_submit_ok]
    
    def generate_training_data(
        self,
        output_path: str,
        n_samples: int = None
    ):
        """
        Generate training data for fine-tuning.
        
        Args:
            output_path: Path to save JSONL training data
            n_samples: Number of samples to generate (None = all)
        """
        import random
        
        # Filter valid records
        valid_df = self.df[
            self.df['Report'].notna() &
            self.df['Procedure'].notna() &
            self.df['ICD10 - Diagnosis'].notna()
        ]
        
        if n_samples and n_samples < len(valid_df):
            valid_df = valid_df.sample(n=n_samples, random_state=42)
        
        training_data = []
        
        for idx, row in valid_df.iterrows():
            report = str(row['Report'])
            cpt = str(row['Procedure'])
            icd = str(row['ICD10 - Diagnosis'])
            
            if cpt == 'nan' or icd == 'nan':
                continue
            
            # Pre-process to build prompt
            preprocessed = self.preprocessor.preprocess(report)
            
            # Generate training example
            icd_codes = [c.strip() for c in icd.split(',')]
            example = self.llm_predictor.generate_training_data(
                report=report,
                cpt=cpt,
                icd_codes=icd_codes,
                prompt=preprocessed.prompt
            )
            
            training_data.append(example)
        
        # Save to JSONL
        with open(output_path, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Generated {len(training_data)} training examples → {output_path}")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_pipeline(
    model: str = "huggingface",  # "huggingface" or "gpt-4o"
    bill_type: str = 'P',
    verbose: bool = True
) -> ProductionPipeline:
    """Create a production pipeline with default settings."""
    return ProductionPipeline(
        model=model,
        bill_type=bill_type,
        verbose=verbose
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import json
    
    # Create pipeline
    pipeline = create_pipeline(verbose=True)
    
    # Get a test sample
    sample = pipeline.df[
        pipeline.df['Modifier'].notna() & 
        (pipeline.df['Modifier'] != '')
    ].iloc[0]
    
    print("\n" + "=" * 70)
    print("TEST CASE")
    print("=" * 70)
    print(f"Ground Truth:")
    print(f"  Procedure: {sample['Procedure']}")
    print(f"  Modifier: {sample['Modifier']}")
    print(f"  ICD10: {sample['ICD10 - Diagnosis']}")
    
    print("\n" + "=" * 70)
    print("RUNNING PIPELINE")
    print("=" * 70)
    
    result = pipeline.process(str(sample['Report']))
    
    print("\n" + "=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"\nCoding:")
    print(f"  Ground Truth: CPT={sample['Procedure']}, Mod={sample['Modifier']}, ICD={sample['ICD10 - Diagnosis']}")
    print(f"  Prediction:   CPT={result.procedure}, Mod={result.modifier}, ICD={result.icd10_diagnosis}")
    
    print(f"\nConfidence:")
    print(f"  Overall:  {result.overall_confidence:.0%}")
    print(f"  CPT:      {result.cpt_confidence:.0%}")
    print(f"  ICD:      {result.icd_confidence:.0%}")
    print(f"  Modifier: {result.modifier_confidence:.0%}")
    
    print(f"\nReview Status:")
    print(f"  Needs Review:   {result.needs_review}")
    print(f"  Auto-submit OK: {result.auto_submit_ok}")
    print(f"  Medical Necessity: {result.has_medical_necessity}")
    
    if result.review_flags:
        print(f"\nReview Flags:")
        for flag in result.review_flags:
            print(f"  [{flag['severity'].upper()}] {flag['message']}")
    
    print("\n" + "=" * 70)
    print("FULL API RESPONSE:")
    print("=" * 70)
    print(json.dumps(result.to_dict(), indent=2))

