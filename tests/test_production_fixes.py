#!/usr/bin/env python3
"""
VFinal4 - Production Fixes Validation Test

Tests the fixes identified from production comparison analysis:
1. OB Ultrasound codes
2. Shoulder X-ray view count
3. TA+TV Complete pelvic
4. CTA detection
5. Hip bilateral vs unilateral
6. Venous vs arterial duplex
7. Tomosynthesis detection

Usage:
    export HF_API_TOKEN="your_token_here"
    python tests/test_production_fixes.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add VFinal4 to path
VFINAL4_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(VFINAL4_DIR))
sys.path.insert(0, str(VFINAL4_DIR / 'src'))

# Also add project root for data access
PROJECT_DIR = VFINAL4_DIR.parent

import pandas as pd
from src.preprocessor import Preprocessor, apply_cpt_disambiguation
from src.llm_predictor import LLMPredictor
from src.postprocessor import Postprocessor


# Test cases from production comparison
TEST_CASES = [
    # OB Ultrasound cases
    {
        "name": "OB US Dating Scan",
        "exam_desc": "US OB DATING SCAN",
        "report": """EXAM: US OB DATING SCAN

INDICATION: First trimester pregnancy, dating.

FINDINGS:
Single live intrauterine pregnancy. Crown-rump length measures 45mm, 
consistent with 11 weeks 2 days gestational age. Fetal heart rate 158 bpm.
Normal yolk sac. No adnexal masses.

IMPRESSION:
1. Single live intrauterine pregnancy at 11 weeks 2 days.
2. Normal early pregnancy.""",
        "expected_cpt": "76801",
        "category": "OB Ultrasound"
    },
    
    # Shoulder X-ray
    {
        "name": "Shoulder XR 2 Views",
        "exam_desc": "XR SHOULDER LEFT 2 VIEWS",
        "report": """EXAM: XR SHOULDER LEFT 2 VIEWS

INDICATION: Left shoulder pain, rule out fracture.

FINDINGS:
Two views of the left shoulder obtained. No acute fracture or dislocation.
Mild degenerative changes of the acromioclavicular joint. 
Soft tissues unremarkable.

IMPRESSION:
1. No acute fracture.
2. Mild AC joint DJD.""",
        "expected_cpt": "73030",
        "category": "Shoulder X-ray"
    },
    
    # Pelvic US TA+TV
    {
        "name": "Pelvic US Complete TA/TV",
        "exam_desc": "US PELVIS TRANSABDOMINAL AND TRANSVAGINAL",
        "report": """EXAM: US PELVIS TRANSABDOMINAL AND TRANSVAGINAL

INDICATION: Pelvic pain, abnormal bleeding.

TECHNIQUE: Transabdominal and transvaginal ultrasound performed.

FINDINGS:
Uterus is anteverted, measuring 8.5 x 4.2 x 5.1 cm. Endometrial stripe 
measures 6mm. No focal myometrial masses. Ovaries are normal in size and 
appearance bilaterally.

IMPRESSION:
1. Normal pelvic ultrasound.
2. No adnexal pathology.""",
        "expected_cpt": "76856",
        "category": "Pelvic US"
    },
    
    # CTA Abdomen Pelvis
    {
        "name": "CTA Abdomen Pelvis",
        "exam_desc": "CT ANGIOGRAPHY ABDOMEN AND PELVIS WITH CONTRAST",
        "report": """EXAM: CT ANGIOGRAPHY ABDOMEN AND PELVIS WITH CONTRAST

INDICATION: Abdominal pain, evaluate for mesenteric ischemia.

TECHNIQUE: CT angiography of the abdomen and pelvis performed with 
IV contrast. Arterial and venous phase imaging obtained.

FINDINGS:
Celiac, SMA, and IMA are patent without significant stenosis.
No aneurysm. Liver, spleen, kidneys unremarkable.

IMPRESSION:
1. Patent mesenteric vasculature, no evidence of ischemia.
2. Unremarkable abdominal CT.""",
        "expected_cpt": "74174",
        "category": "CTA"
    },
    
    # Hip Bilateral
    {
        "name": "Hip XR Bilateral",
        "exam_desc": "XR HIPS BILATERAL 2 VIEWS",
        "report": """EXAM: XR HIPS BILATERAL 2 VIEWS

INDICATION: Bilateral hip pain.

FINDINGS:
Bilateral hip radiographs, 2 views each. Mild bilateral hip osteoarthritis
with joint space narrowing. No acute fracture. Femoral heads are spherical.

IMPRESSION:
1. Mild bilateral hip osteoarthritis.""",
        "expected_cpt": "73521",
        "category": "Hip X-ray"
    },
    
    # Venous Duplex
    {
        "name": "Venous Duplex Lower Extremity",
        "exam_desc": "DUPLEX LOWER EXTREMITY VENOUS LEFT",
        "report": """EXAM: DUPLEX LOWER EXTREMITY VENOUS LEFT

INDICATION: Left leg swelling, rule out DVT.

FINDINGS:
Duplex ultrasound of the left lower extremity venous system.
Common femoral, femoral, and popliteal veins are patent and compressible.
Normal flow and augmentation. No evidence of deep vein thrombosis.

IMPRESSION:
1. No DVT in the left lower extremity.""",
        "expected_cpt": "93971",
        "category": "Duplex"
    },
    
    # Mammography with Tomo
    {
        "name": "Screening Mammo with Tomo",
        "exam_desc": "MAMMOGRAPHY SCREENING BILATERAL WITH TOMOSYNTHESIS",
        "report": """EXAM: MAMMOGRAPHY SCREENING BILATERAL WITH TOMOSYNTHESIS

INDICATION: Annual screening mammography.

TECHNIQUE: Bilateral digital mammography with tomosynthesis performed.

FINDINGS:
Breast composition: Scattered fibroglandular densities.
No suspicious masses, calcifications, or architectural distortion.

IMPRESSION:
1. Negative screening mammogram. BI-RADS 1.""",
        "expected_cpt": "77063",
        "category": "Mammography"
    },
]


def main():
    hf_token = os.environ.get('HF_API_TOKEN')
    if not hf_token:
        print("ERROR: HF_API_TOKEN not set")
        print("Set it with: export HF_API_TOKEN='your_token_here'")
        sys.exit(1)
    
    print("=" * 80)
    print("VFINAL4 - PRODUCTION FIXES VALIDATION TEST")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Load training data for preprocessor
    data_path = Path('/workspace/data/consolidated_cleaned.csv')
    if not data_path.exists():
        data_path = PROJECT_DIR / 'data' / 'consolidated_cleaned.csv'
    
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Loaded {len(df):,} records")
    
    # Initialize components
    print("\nInitializing pipeline components...")
    preprocessor = Preprocessor(df=df, use_rag=True, rag_use_reranker=False)
    llm = LLMPredictor(backend="huggingface")
    postprocessor = Postprocessor(bill_type='P')
    print("✅ Pipeline initialized")
    
    # Run tests
    passed = 0
    failed = 0
    results = []
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(TEST_CASES)}] {test['name']}")
        print(f"Category: {test['category']}")
        print(f"Expected CPT: {test['expected_cpt']}")
        print(f"{'='*60}")
        
        try:
            # Step 1: Preprocess
            preprocessed = preprocessor.preprocess(test['report'])
            print(f"  Preprocessor CPT candidates: {preprocessed.rag_cpt_candidates[:3]}")
            
            # Step 2: LLM Prediction
            valid_icd_codes = set(code for code, _ in preprocessed.icd_candidates)
            prediction = llm.predict(
                prompt=preprocessed.prompt,
                extracted_cpt=preprocessed.extracted_cpt,
                valid_icd_codes=valid_icd_codes,
                report_text=test['report'],
                exam_desc=test['exam_desc']
            )
            print(f"  LLM predicted: CPT={prediction.cpt_code}, ICD={prediction.icd_codes[:2]}")
            
            # Step 3: Postprocess
            final = postprocessor.postprocess(
                cpt_code=prediction.cpt_code,
                icd_codes=prediction.icd_codes,
                laterality=preprocessed.laterality,
                cpt_was_extracted=(preprocessed.extracted_cpt is not None),
                valid_icd_codes=valid_icd_codes,
                llm_suggested_modifiers=prediction.modifiers
            )
            
            # Check CPT match
            cpt_match = final.procedure == test['expected_cpt']
            
            if cpt_match:
                passed += 1
                status = "✅ PASS"
            else:
                failed += 1
                status = "❌ FAIL"
            
            print(f"\n{status}")
            print(f"  Expected CPT: {test['expected_cpt']}")
            print(f"  Final CPT:    {final.procedure}")
            print(f"  Modifier:     {final.modifier}")
            print(f"  ICD-10:       {final.icd10_diagnosis[:50]}...")
            
            results.append({
                'name': test['name'],
                'category': test['category'],
                'expected_cpt': test['expected_cpt'],
                'actual_cpt': final.procedure,
                'match': cpt_match,
                'modifier': final.modifier,
                'icd': final.icd10_diagnosis
            })
            
        except Exception as e:
            failed += 1
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': test['name'],
                'category': test['category'],
                'expected_cpt': test['expected_cpt'],
                'actual_cpt': 'ERROR',
                'match': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal:  {len(TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Rate:   {passed/len(TEST_CASES)*100:.1f}%")
    
    # Results by category
    print("\nBy Category:")
    categories = {}
    for r in results:
        cat = r['category']
        if cat not in categories:
            categories[cat] = {'pass': 0, 'fail': 0}
        if r['match']:
            categories[cat]['pass'] += 1
        else:
            categories[cat]['fail'] += 1
    
    for cat, counts in categories.items():
        total = counts['pass'] + counts['fail']
        status = "✅" if counts['fail'] == 0 else "❌"
        print(f"  {status} {cat}: {counts['pass']}/{total}")
    
    # Failed cases detail
    if failed > 0:
        print("\nFailed Cases:")
        for r in results:
            if not r['match']:
                print(f"  - {r['name']}: expected {r['expected_cpt']}, got {r['actual_cpt']}")
    
    print("\n" + "=" * 80)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
