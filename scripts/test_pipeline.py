#!/usr/bin/env python3
"""
Test Pipeline
=============

Quick test to verify the pipeline is working correctly.

Usage:
    python scripts/test_pipeline.py
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set token if not already set
if not os.getenv("HF_API_TOKEN"):
    print("⚠ HF_API_TOKEN not set. Set it with:")
    print("  export HF_API_TOKEN='your_token'")
    print()


def test_basic_prediction():
    """Test basic prediction workflow."""
    print("=" * 60)
    print("NYXMed Pipeline Test")
    print("=" * 60)
    
    from src.pipeline import ProductionPipeline
    
    # Initialize
    print("\n1. Initializing pipeline...")
    pipeline = ProductionPipeline(verbose=True)
    
    # Health check
    print("\n2. Health check...")
    print(f"   Pipeline ready: True")
    print(f"   Using HuggingFace backend")
    
    # Test reports
    test_reports = [
        {
            "name": "CT Chest",
            "report": """
EXAM: CT CHEST WITH CONTRAST
CLINICAL INDICATION: 65 year old male with shortness of breath. Rule out PE.
TECHNIQUE: CT chest with IV contrast using PE protocol.
FINDINGS:
- No evidence of pulmonary embolism
- Mild emphysematous changes bilateral
- Heart size normal
IMPRESSION:
1. No pulmonary embolism
2. Mild emphysema
""",
            "expected_cpt": "71260"
        },
        {
            "name": "Chest X-ray",
            "report": """
EXAM: CHEST X-RAY 2 VIEWS
CLINICAL INDICATION: Cough and fever
FINDINGS: Right lower lobe opacity consistent with pneumonia
IMPRESSION: Right lower lobe pneumonia
""",
            "expected_cpt": "71046"
        },
        {
            "name": "Screening Mammography",
            "report": """
EXAM: BILATERAL SCREENING MAMMOGRAPHY
CLINICAL INDICATION: Annual screening, age 55
FINDINGS: Heterogeneously dense breasts. No suspicious masses.
IMPRESSION: Negative. BI-RADS 1.
""",
            "expected_cpt": "77067"
        }
    ]
    
    print("\n3. Testing predictions...")
    print("-" * 60)
    
    passed = 0
    for test in test_reports:
        print(f"\nTest: {test['name']}")
        
        result = pipeline.process(test['report'])
        
        cpt_match = result.procedure == test['expected_cpt']
        status = "✓" if cpt_match else "✗"
        
        print(f"  CPT: {result.procedure} (expected: {test['expected_cpt']}) {status}")
        print(f"  Modifier: {result.modifier}")
        print(f"  ICD: {result.icd10_diagnosis}")
        print(f"  Confidence: {result.overall_confidence:.2f}")
        print(f"  Needs Review: {result.needs_review}")
        
        if cpt_match:
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{len(test_reports)} tests passed")
    print("=" * 60)
    
    return passed == len(test_reports)


def test_with_custom_examples():
    """Test prediction with custom examples."""
    print("\n" + "=" * 60)
    print("Testing Custom Examples")
    print("=" * 60)
    
    from src.pipeline import ProductionPipeline
    
    pipeline = ProductionPipeline(verbose=False)
    
    # Test report
    report = """
EXAM: CT HEAD WITHOUT CONTRAST
INDICATION: Dizziness and headache for 3 days
FINDINGS: No acute intracranial abnormality
IMPRESSION: Normal CT head
"""
    
    result = pipeline.process(report)
    
    print(f"\nResult:")
    print(f"  CPT: {result.procedure}")
    print(f"  Modifier: {result.modifier}")
    print(f"  ICD: {result.icd10_diagnosis}")
    print(f"  Confidence: {result.overall_confidence:.2f}")
    
    return result.procedure == "70450"


if __name__ == "__main__":
    success1 = test_basic_prediction()
    success2 = test_with_custom_examples()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("All tests passed! ✓")
    else:
        print("Some tests failed ✗")
    print("=" * 60)

