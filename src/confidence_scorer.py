#!/usr/bin/env python3
"""
CONFIDENCE SCORING & REVIEW FLAGGING

Calculates confidence scores for each prediction and determines
which cases need human review before submission.

Designed for front-end integration - returns structured flags.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum


class ReviewReason(Enum):
    """Reasons why a case might need human review."""
    
    # Critical - should not auto-submit
    NO_MEDICAL_NECESSITY = "no_medical_necessity"
    LOW_CONFIDENCE = "low_confidence"
    RARE_CODE_COMBINATION = "rare_code_combination"
    
    # Warning - review recommended
    CPT_PREDICTED_NOT_EXTRACTED = "cpt_predicted"
    AMBIGUOUS_LATERALITY = "ambiguous_laterality"
    HIGH_CODE_COUNT = "high_code_count"
    RAG_DISAGREEMENT = "rag_disagreement"
    
    # Info - FYI for reviewer
    UNUSUAL_CPT_ICD_PAIRING = "unusual_pairing"
    SCREENING_WITHOUT_Z_CODE = "screening_no_z"


@dataclass
class ReviewFlag:
    """A single flag indicating something to review."""
    reason: ReviewReason
    severity: str  # 'critical', 'warning', 'info'
    message: str
    field: str  # 'cpt', 'icd', 'modifier', 'general'
    details: Dict = field(default_factory=dict)


@dataclass
class ConfidenceResult:
    """Complete confidence assessment for a prediction."""
    
    # Overall scores (0.0 - 1.0)
    overall_confidence: float
    cpt_confidence: float
    icd_confidence: float
    modifier_confidence: float
    
    # Review status
    needs_review: bool
    auto_submit_ok: bool
    review_flags: List[ReviewFlag] = field(default_factory=list)
    
    # Thresholds used
    confidence_threshold: float = 0.75
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON/API response."""
        return {
            'overall_confidence': round(self.overall_confidence, 2),
            'cpt_confidence': round(self.cpt_confidence, 2),
            'icd_confidence': round(self.icd_confidence, 2),
            'modifier_confidence': round(self.modifier_confidence, 2),
            'needs_review': self.needs_review,
            'auto_submit_ok': self.auto_submit_ok,
            'review_flags': [
                {
                    'reason': flag.reason.value,
                    'severity': flag.severity,
                    'message': flag.message,
                    'field': flag.field,
                    'details': flag.details
                }
                for flag in self.review_flags
            ]
        }
    
    def get_critical_flags(self) -> List[ReviewFlag]:
        """Get only critical flags that block auto-submission."""
        return [f for f in self.review_flags if f.severity == 'critical']
    
    def get_warning_flags(self) -> List[ReviewFlag]:
        """Get warning flags that recommend review."""
        return [f for f in self.review_flags if f.severity == 'warning']


class ConfidenceScorer:
    """
    Calculates confidence scores and determines review requirements.
    
    Thresholds:
    - overall_confidence < 0.75 → needs_review = True
    - Any critical flag → auto_submit_ok = False
    - Any warning flag → needs_review = True (but can auto-submit if above threshold)
    """
    
    # Thresholds
    AUTO_SUBMIT_THRESHOLD = 0.85  # Above this = can auto-submit (if no critical flags)
    REVIEW_THRESHOLD = 0.75       # Below this = must review
    LOW_CONFIDENCE_THRESHOLD = 0.50  # Below this = critical
    
    # Code count thresholds
    MAX_NORMAL_CODE_COUNT = 5     # More than this = flag for review
    
    # Common CPT-ICD pairings (simplified - would be data-driven in production)
    COMMON_CPT_ICD_PREFIXES = {
        '71045': ['R', 'J', 'Z'],  # Chest X-ray
        '71046': ['R', 'J', 'Z'],  # Chest X-ray 2 views
        '70450': ['R', 'G', 'S'],  # CT Head
        '74176': ['R', 'K', 'C'],  # CT Abd/Pel
        '77067': ['Z12', 'Z80', 'Z85'],  # Screening mammo
        '77065': ['N', 'R92', 'C50'],  # Diagnostic mammo
    }
    
    def __init__(
        self, 
        auto_submit_threshold: float = 0.85,
        review_threshold: float = 0.75
    ):
        self.auto_submit_threshold = auto_submit_threshold
        self.review_threshold = review_threshold
    
    def calculate_confidence(
        self,
        # Required inputs (no defaults)
        cpt_code: str,
        cpt_was_extracted: bool,
        icd_codes: List[str],
        modifiers: List[str],
        
        # Optional inputs (with defaults)
        cpt_in_rag_candidates: bool = False,
        icd_candidates: Set[str] = None,
        rag_icd_codes: List[str] = None,
        laterality_detected: str = 'none',
        has_medical_necessity: bool = True,
        medical_necessity_warning: str = "",
        report_length: int = 0,
    ) -> ConfidenceResult:
        """
        Calculate comprehensive confidence scores.
        
        Returns ConfidenceResult with scores and review flags.
        """
        flags: List[ReviewFlag] = []
        
        # =====================================================================
        # 1. CPT CONFIDENCE
        # =====================================================================
        cpt_confidence = self._calculate_cpt_confidence(
            cpt_code, cpt_was_extracted, cpt_in_rag_candidates, flags
        )
        
        # =====================================================================
        # 2. ICD CONFIDENCE
        # =====================================================================
        icd_confidence = self._calculate_icd_confidence(
            icd_codes, icd_candidates, rag_icd_codes, cpt_code, flags
        )
        
        # =====================================================================
        # 3. MODIFIER CONFIDENCE
        # =====================================================================
        modifier_confidence = self._calculate_modifier_confidence(
            modifiers, laterality_detected, cpt_code, flags
        )
        
        # =====================================================================
        # 4. MEDICAL NECESSITY CHECK
        # =====================================================================
        if not has_medical_necessity:
            flags.append(ReviewFlag(
                reason=ReviewReason.NO_MEDICAL_NECESSITY,
                severity='critical',
                message='No valid medical necessity justification found',
                field='icd',
                details={'warning': medical_necessity_warning}
            ))
        
        # =====================================================================
        # 5. OVERALL CONFIDENCE
        # =====================================================================
        # Weighted average: ICD is most important, then CPT, then modifier
        overall_confidence = (
            icd_confidence * 0.50 +
            cpt_confidence * 0.35 +
            modifier_confidence * 0.15
        )
        
        # Penalty for missing medical necessity
        if not has_medical_necessity:
            overall_confidence *= 0.5
        
        # Low confidence flag
        if overall_confidence < self.LOW_CONFIDENCE_THRESHOLD:
            flags.append(ReviewFlag(
                reason=ReviewReason.LOW_CONFIDENCE,
                severity='critical',
                message=f'Overall confidence too low: {overall_confidence:.0%}',
                field='general',
                details={'threshold': self.LOW_CONFIDENCE_THRESHOLD}
            ))
        
        # =====================================================================
        # 6. DETERMINE REVIEW STATUS
        # =====================================================================
        has_critical = any(f.severity == 'critical' for f in flags)
        has_warning = any(f.severity == 'warning' for f in flags)
        
        # Auto-submit OK only if:
        # - No critical flags
        # - Overall confidence above auto-submit threshold
        auto_submit_ok = (
            not has_critical and 
            overall_confidence >= self.auto_submit_threshold
        )
        
        # Needs review if:
        # - Any flags (critical or warning)
        # - Overall confidence below review threshold
        needs_review = (
            has_critical or 
            has_warning or 
            overall_confidence < self.review_threshold
        )
        
        return ConfidenceResult(
            overall_confidence=overall_confidence,
            cpt_confidence=cpt_confidence,
            icd_confidence=icd_confidence,
            modifier_confidence=modifier_confidence,
            needs_review=needs_review,
            auto_submit_ok=auto_submit_ok,
            review_flags=flags,
            confidence_threshold=self.review_threshold
        )
    
    def _calculate_cpt_confidence(
        self,
        cpt_code: str,
        cpt_was_extracted: bool,
        cpt_in_rag_candidates: bool,
        flags: List[ReviewFlag]
    ) -> float:
        """Calculate CPT code confidence."""
        
        if not cpt_code:
            return 0.0
        
        confidence = 0.6  # Base - valid CPT format is good start
        
        # Extracted from report = highest confidence
        if cpt_was_extracted:
            confidence += 0.3
        else:
            # Predicted by LLM - still good if valid format
            confidence += 0.2
            # Only flag as INFO, not a penalty
            flags.append(ReviewFlag(
                reason=ReviewReason.CPT_PREDICTED_NOT_EXTRACTED,
                severity='info',
                message='CPT was predicted by AI, not extracted from report',
                field='cpt',
                details={'predicted_cpt': cpt_code}
            ))
        
        # RAG agreement boosts confidence
        if cpt_in_rag_candidates:
            confidence += 0.1
        
        # Valid 5-digit CPT format = confidence boost
        if cpt_code and len(cpt_code) == 5 and cpt_code.isdigit():
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_icd_confidence(
        self,
        icd_codes: List[str],
        icd_candidates: Set[str],
        rag_icd_codes: List[str],
        cpt_code: str,
        flags: List[ReviewFlag]
    ) -> float:
        """Calculate ICD code confidence."""
        
        if not icd_codes:
            return 0.0
        
        confidence = 0.6  # Base - having ICD codes is good
        
        # All codes were in candidates = good
        if icd_candidates:
            codes_in_candidates = sum(1 for c in icd_codes if c in icd_candidates)
            candidate_ratio = codes_in_candidates / len(icd_codes)
            confidence += candidate_ratio * 0.2
        else:
            # No candidates available - but if codes are well-formed, still good
            # Check if all codes match ICD-10 format (letter + digits + optional decimal)
            import re
            well_formed = all(re.match(r'^[A-Z]\d{2}(\.\d{1,4})?$', c) for c in icd_codes)
            if well_formed:
                confidence += 0.15  # Boost for valid format even without candidates
        
        # RAG agreement
        if rag_icd_codes:
            rag_set = set(rag_icd_codes)
            overlap = len(set(icd_codes) & rag_set)
            if overlap > 0:
                confidence += 0.15
            elif len(icd_codes) > 0:
                # No overlap with RAG = potential issue (but only flag, don't penalize hard)
                flags.append(ReviewFlag(
                    reason=ReviewReason.RAG_DISAGREEMENT,
                    severity='info',  # Changed from 'warning' - disagreement isn't always bad
                    message='Predicted codes differ from similar cases',
                    field='icd',
                    details={
                        'predicted': icd_codes,
                        'rag_suggested': list(rag_set)[:5]
                    }
                ))
        
        # Code count check
        if len(icd_codes) > self.MAX_NORMAL_CODE_COUNT:
            flags.append(ReviewFlag(
                reason=ReviewReason.HIGH_CODE_COUNT,
                severity='warning',
                message=f'High number of codes ({len(icd_codes)})',
                field='icd',
                details={'count': len(icd_codes), 'threshold': self.MAX_NORMAL_CODE_COUNT}
            ))
            confidence -= 0.1
        
        # CPT-ICD pairing check (only for common CPTs we know about)
        if cpt_code in self.COMMON_CPT_ICD_PREFIXES:
            expected_prefixes = self.COMMON_CPT_ICD_PREFIXES[cpt_code]
            has_expected = any(
                any(c.startswith(p) for p in expected_prefixes)
                for c in icd_codes
            )
            if not has_expected:
                flags.append(ReviewFlag(
                    reason=ReviewReason.UNUSUAL_CPT_ICD_PAIRING,
                    severity='info',
                    message=f'Unusual ICD codes for CPT {cpt_code}',
                    field='icd',
                    details={'expected_prefixes': expected_prefixes}
                ))
        
        # Reasonable code count boosts confidence
        if 1 <= len(icd_codes) <= 4:
            confidence += 0.1
        
        # Medical codes (C for cancer, etc.) are high-confidence indicators
        has_definitive_diagnosis = any(
            c.startswith(('C', 'D0', 'D1', 'D2', 'D3', 'D4', 'J', 'I', 'M', 'S', 'T'))
            for c in icd_codes
        )
        if has_definitive_diagnosis:
            confidence += 0.05  # Definitive diagnoses are more reliable
        
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_modifier_confidence(
        self,
        modifiers: List[str],
        laterality_detected: str,
        cpt_code: str,
        flags: List[ReviewFlag]
    ) -> float:
        """Calculate modifier confidence."""
        
        confidence = 0.8  # Base - modifiers are rules-based
        
        # Ambiguous laterality
        if laterality_detected == 'none':
            # Check if this CPT typically needs laterality
            laterality_cpts = {
                '73030', '73560', '73610', '73630', '76642', '77065'
            }
            if cpt_code in laterality_cpts:
                flags.append(ReviewFlag(
                    reason=ReviewReason.AMBIGUOUS_LATERALITY,
                    severity='warning',
                    message=f'CPT {cpt_code} may need laterality modifier',
                    field='modifier',
                    details={'cpt': cpt_code, 'detected': laterality_detected}
                ))
                confidence -= 0.2
        
        # Bilateral with conflicting laterality
        if laterality_detected == 'bilateral' and ('LT' in modifiers or 'RT' in modifiers):
            # This shouldn't happen - postprocessor should handle
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def score_prediction(
    cpt_code: str,
    icd_codes: List[str],
    modifiers: List[str],
    cpt_was_extracted: bool = False,
    has_medical_necessity: bool = True,
    laterality: str = 'none',
    rag_icd_codes: List[str] = None,
    icd_candidates: Set[str] = None,
) -> ConfidenceResult:
    """
    Convenience function to score a prediction.
    
    Example usage:
        result = score_prediction(
            cpt_code='71046',
            icd_codes=['R91.8', 'J18.9'],
            modifiers=['26'],
            cpt_was_extracted=True,
            has_medical_necessity=True
        )
        
        if result.needs_review:
            send_to_review_queue(result.review_flags)
        elif result.auto_submit_ok:
            submit_claim()
    """
    scorer = ConfidenceScorer()
    return scorer.calculate_confidence(
        cpt_code=cpt_code,
        cpt_was_extracted=cpt_was_extracted,
        icd_codes=icd_codes,
        modifiers=modifiers,
        laterality_detected=laterality,
        has_medical_necessity=has_medical_necessity,
        rag_icd_codes=rag_icd_codes,
        icd_candidates=icd_candidates
    )


# =============================================================================
# EXAMPLE / TEST
# =============================================================================

if __name__ == "__main__":
    # Test the confidence scorer
    
    print("=" * 70)
    print("CONFIDENCE SCORER TEST")
    print("=" * 70)
    
    # Test Case 1: High confidence - should auto-submit
    result1 = score_prediction(
        cpt_code='71046',
        icd_codes=['R91.8', 'J18.9'],
        modifiers=['26'],
        cpt_was_extracted=True,
        has_medical_necessity=True,
        laterality='none'
    )
    print("\nTest 1: High confidence case")
    print(f"  Overall: {result1.overall_confidence:.0%}")
    print(f"  Auto-submit OK: {result1.auto_submit_ok}")
    print(f"  Needs review: {result1.needs_review}")
    print(f"  Flags: {len(result1.review_flags)}")
    
    # Test Case 2: Low confidence - needs review
    result2 = score_prediction(
        cpt_code='77067',
        icd_codes=['M79.3'],  # Unusual for mammography
        modifiers=['26'],
        cpt_was_extracted=False,
        has_medical_necessity=False,
        laterality='none'
    )
    print("\nTest 2: Low confidence case")
    print(f"  Overall: {result2.overall_confidence:.0%}")
    print(f"  Auto-submit OK: {result2.auto_submit_ok}")
    print(f"  Needs review: {result2.needs_review}")
    print(f"  Flags: {[f.reason.value for f in result2.review_flags]}")
    
    # Test Case 3: Medium confidence with warnings
    result3 = score_prediction(
        cpt_code='73030',
        icd_codes=['M25.511', 'M75.101', 'R29.898'],
        modifiers=['26'],
        cpt_was_extracted=False,
        has_medical_necessity=True,
        laterality='none'  # Shoulder XR without laterality
    )
    print("\nTest 3: Medium confidence with warnings")
    print(f"  Overall: {result3.overall_confidence:.0%}")
    print(f"  Auto-submit OK: {result3.auto_submit_ok}")
    print(f"  Needs review: {result3.needs_review}")
    for flag in result3.review_flags:
        print(f"  - [{flag.severity}] {flag.message}")
    
    print("\n" + "=" * 70)
    print("API Response Format:")
    print("=" * 70)
    import json
    print(json.dumps(result3.to_dict(), indent=2))

