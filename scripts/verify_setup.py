#!/usr/bin/env python3
"""
VFinal2 - Pre-Production Verification Script

Run this BEFORE any production run to ensure everything is configured correctly.

Usage:
    cd /Users/vineetdaniels/Desktop/Coding\ Projects/NyxMed/VFinal2
    python3 scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Setup path
SCRIPT_DIR = Path(__file__).parent
VFINAL2_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(VFINAL2_DIR / 'src'))


def main():
    print("=" * 80)
    print("VFINAL2 PRE-PRODUCTION VERIFICATION")
    print("=" * 80)
    
    all_pass = True
    
    # 1. Check disambiguation rules exist
    print("\n1. CPT DISAMBIGUATION RULES")
    try:
        from preprocessor import apply_cpt_disambiguation
        import inspect
        source = inspect.getsource(apply_cpt_disambiguation)
        
        rules = [
            ('Fix 1', '77061', 'Mammography tomosynthesis'),
            ('Fix 5', '74176', 'CT Abd/Pelvis contrast'),
            ('Fix 14', '72131', 'Spine CT vs MR'),
            ('Fix 18', '76641', 'Breast US vs Mammography'),
            ('Fix 20', '74174', 'CTA chest vs abdomen'),
            ('Fix 21', '37244', 'Vascular embolization'),
        ]
        
        for rule, code, desc in rules:
            if code in source:
                print(f"   ✓ {rule}: {desc}")
            else:
                print(f"   ✗ {rule}: {desc} - MISSING!")
                all_pass = False
    except Exception as e:
        print(f"   ✗ Error loading preprocessor: {e}")
        all_pass = False
    
    # 2. Check postprocessor
    print("\n2. POSTPROCESSOR")
    try:
        from postprocessor import Postprocessor
        pp = Postprocessor(bill_type='P')
        if hasattr(pp, 'INTERVENTIONAL_CPT_CODES'):
            print(f"   ✓ INTERVENTIONAL_CPT_CODES ({len(pp.INTERVENTIONAL_CPT_CODES)} codes)")
        else:
            print("   ✗ INTERVENTIONAL_CPT_CODES missing!")
            all_pass = False
    except Exception as e:
        print(f"   ✗ Error loading postprocessor: {e}")
        all_pass = False
    
    # 3. Check LLM predictor
    print("\n3. LLM PREDICTOR")
    llm_file = VFINAL2_DIR / 'src' / 'llm_predictor.py'
    if llm_file.exists():
        with open(llm_file, 'r') as f:
            if 'apply_cpt_disambiguation' in f.read():
                print("   ✓ Disambiguation integrated")
            else:
                print("   ✗ Disambiguation NOT integrated!")
                all_pass = False
    else:
        print("   ✗ llm_predictor.py missing!")
        all_pass = False
    
    # 4. Check RAG indices
    print("\n4. RAG INDICES")
    indices_dir = VFINAL2_DIR / 'indices'
    files = [
        ('nyxmed_71k_bm25.pkl', 'BM25'),
        ('nyxmed_71k_faiss.index', 'FAISS'),
        ('nyxmed_71k_metadata.pkl', 'Metadata'),
    ]
    
    for fname, desc in files:
        f = indices_dir / fname
        if f.exists():
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"   ✓ {desc}: {size_mb:.1f} MB")
        else:
            print(f"   ✗ {desc} MISSING!")
            all_pass = False
    
    # 5. Test disambiguation
    print("\n5. DISAMBIGUATION TESTS")
    tests = [
        ('76641', 'MAMMO BREAST DIAGNOSTIC TOMOSYNTHESIS UNILATERAL', '77061'),
        ('76641', 'MAMMO BREAST SCREENING TOMOSYNTHESIS BILATERAL', '77063'),
        ('74178', 'CT ABDOMEN PELVIS WITHOUT IV CONTRAST', '74176'),
        ('74175', 'CT CHEST ANGIOGRAPHY WITH IV CONTRAST', '71275'),
        ('76641', 'US BREAST COMPLETE BILATERAL', '76642'),
    ]
    
    try:
        for pred, exam, expected in tests:
            result = apply_cpt_disambiguation(pred, '', exam)
            if result == expected:
                print(f"   ✓ {pred}→{result}")
            else:
                print(f"   ✗ {pred}→{result} (expected {expected})")
                all_pass = False
    except Exception as e:
        print(f"   ✗ Error running tests: {e}")
        all_pass = False
    
    # 6. Check HF token
    print("\n6. HUGGINGFACE TOKEN")
    import os
    if os.environ.get('HF_API_TOKEN'):
        print("   ✓ HF_API_TOKEN is set")
    else:
        print("   ⚠ HF_API_TOKEN not set (run: export HF_API_TOKEN='...')")
    
    # Final result
    print("\n" + "=" * 80)
    if all_pass:
        print("✅ ALL CHECKS PASSED - Ready for production!")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues above before running")
    print("=" * 80)
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
