#!/usr/bin/env python3
"""
STEP 1: PRE-PROCESSING

Extracts features from report and prepares data for LLM call.
No LLM calls in this step - all rules-based.

Outputs:
- Extracted CPT (if embedded in report)
- Laterality (for modifiers)
- ICD candidates (from CPT→ICD lookup)
- Few-shot examples
- Built prompt for LLM
"""

import re
import json
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from collections import Counter

# Try to import RAG retriever
try:
    from .rag_retriever import get_rag_retriever, RAGRetriever
    HAS_RAG = True
except ImportError:
    HAS_RAG = False
    RAGRetriever = None

# Import smart section extractor
try:
    # Add src to path for import
    # VFinal/src/preprocessor.py → VFinal/src → VFinal → NyxMed → NyxMed/src
    src_path = Path(__file__).parent.parent.parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from .report_section_extractor import ReportSectionExtractor
    HAS_EXTRACTOR = True
except ImportError:
    HAS_EXTRACTOR = False
    ReportSectionExtractor = None

# =============================================================================
# CPT CODE REFERENCE (for prediction)
# =============================================================================

CPT_CODES = {
    # Chest X-rays
    '71045': 'Chest X-ray, single view',
    '71046': 'Chest X-ray, 2 views',
    '71047': 'Chest X-ray, 3 views',
    '71048': 'Chest X-ray, 4+ views',
    # Chest CT
    '71250': 'CT Chest without contrast',
    '71260': 'CT Chest with contrast',
    '71270': 'CT Chest without and with contrast',
    '71275': 'CT Angiography Chest',
    # Head/Brain
    '70450': 'CT Head without contrast',
    '70460': 'CT Head with contrast',
    '70470': 'CT Head without and with contrast',
    '70496': 'CT Angiography Head',
    '70498': 'CT Angiography Neck',
    '70551': 'MRI Brain without contrast',
    '70553': 'MRI Brain without and with contrast',
    # Abdomen/Pelvis
    '74018': 'Abdomen X-ray, 1 view',
    '74019': 'Abdomen X-ray, 2 views',
    '74021': 'Abdomen X-ray, 3+ views',
    '74150': 'CT Abdomen without contrast',
    '74160': 'CT Abdomen with contrast',
    '74170': 'CT Abdomen without and with contrast',
    '74176': 'CT Abdomen and Pelvis without contrast',
    '74177': 'CT Abdomen and Pelvis with contrast',
    '74178': 'CT Abdomen and Pelvis without and with contrast',
    '74183': 'MRI Abdomen without and with contrast',
    # Ultrasound
    '76700': 'Ultrasound Abdomen, complete',
    '76705': 'Ultrasound Abdomen, limited',
    '76770': 'Ultrasound Retroperitoneal, complete',
    '76856': 'Ultrasound Pelvis, complete',
    '76857': 'Ultrasound Pelvis, limited',
    '76830': 'Ultrasound Transvaginal',
    '76536': 'Ultrasound Thyroid/Soft tissue neck',
    # OB Ultrasound (FIX 1 - Production comparison)
    '76801': 'OB US <14 weeks single gestation',
    '76802': 'OB US <14 weeks each additional',
    '76805': 'OB US >=14 weeks single gestation',
    '76810': 'OB US >=14 weeks each additional',
    '76811': 'OB US <14 weeks detailed single',
    '76812': 'OB US <14 weeks detailed each additional',
    '76815': 'OB US limited (viability)',
    '76816': 'OB US follow-up',
    '76817': 'OB US transvaginal',
    '76818': 'Fetal biophysical profile with NST',
    '76819': 'Fetal biophysical profile without NST',
    # Mammography
    '77063': 'Screening digital breast tomosynthesis, bilateral',
    '77065': 'Diagnostic mammography, unilateral',
    '77066': 'Diagnostic mammography, bilateral',
    '77067': 'Screening mammography, bilateral',
    # Breast Ultrasound
    '76641': 'Ultrasound breast, unilateral, complete',
    '76642': 'Ultrasound breast, unilateral, limited',
    # DEXA
    '77080': 'DEXA bone density, axial',
    '77085': 'DEXA bone density, axial with VFA',
    # Spine
    '72125': 'CT Cervical spine without contrast',
    '72128': 'CT Thoracic spine without contrast',
    '72131': 'CT Lumbar spine without contrast',
    '72141': 'MRI Cervical spine without contrast',
    '72148': 'MRI Lumbar spine without contrast',
    # Extremities
    '73030': 'Shoulder X-ray, minimum 2 views',
    '73110': 'Wrist X-ray, minimum 3 views',
    '73130': 'Hand X-ray, minimum 3 views',
    '73502': 'Hip X-ray, unilateral, minimum 2 views',
    '73560': 'Knee X-ray, 1 or 2 views',
    '73562': 'Knee X-ray, 3 views',
    '73590': 'Tibia/Fibula X-ray, 2 views',
    '73610': 'Ankle X-ray, minimum 3 views',
    '73630': 'Foot X-ray, minimum 3 views',
    '73080': 'Elbow X-ray, minimum 3 views',
    '73200': 'CT Upper extremity without contrast',
    '73201': 'CT Upper extremity with contrast',
    '73202': 'CT Upper extremity without and with contrast',
    '73701': 'CT Lower extremity with contrast',
    '73702': 'CT Lower extremity without and with contrast',
    '93970': 'Duplex scan extremity veins, complete bilateral',
    '73700': 'CT Lower extremity without contrast',
    '73221': 'MRI Upper extremity joint without contrast',
    '73721': 'MRI Lower extremity joint without contrast',
    # Vascular
    '93970': 'Duplex scan extremity veins, complete bilateral',
    '93971': 'Duplex scan extremity veins, unilateral',
    '93925': 'Duplex scan lower extremity arteries, complete bilateral',
    '93926': 'Duplex scan lower extremity arteries, unilateral',
    '93930': 'Duplex scan upper extremity arteries, complete bilateral',
    '93931': 'Duplex scan upper extremity arteries, unilateral',
    '93976': 'Duplex scan pelvic arteries/veins',
    '75574': 'CT Angiography Coronary',
    # Hip X-ray (FIX 5 - Production comparison)
    '73501': 'Hip X-ray, 1 view unilateral',
    '73503': 'Hip X-ray, 3+ views unilateral',
    '73521': 'Hips X-ray, bilateral 2 views',
    '73522': 'Hips X-ray, bilateral 3-4 views',
    '73523': 'Hips X-ray, bilateral 5+ views',
    # Shoulder X-ray (FIX 2 - Production comparison)
    '73020': 'Shoulder X-ray, 1 view',
}


def apply_cpt_disambiguation(predicted_cpt: str, report: str, exam_desc: str = "") -> str:
    """
    Apply targeted fixes for known CPT confusion patterns.
    
    Handles:
    1. 77067 vs 77063 (screening mammo vs tomosynthesis)
    2. 76856 vs 76857 (complete vs limited pelvic US)
    3. 74018 vs 74019 (1 view vs 2 view abdomen XR)
    4. 70496 vs 70498 (head vs neck CTA)
    5. 71250/71260 vs 74177 (chest CT vs abdomen/pelvis CT)
    6. 71045 vs 71046 (1-view vs 2-view chest X-ray) - NEW
    7. 73562 vs 73564 (3-view vs 4-view knee X-ray) - NEW
    8. 77066 vs 76642 (diagnostic mammo vs breast US) - NEW
    9. 73060 vs 73600 (humerus vs ankle X-ray) - NEW
    """
    report_lower = report.lower()
    exam_lower = exam_desc.lower() if exam_desc else ''
    combined = report_lower + ' ' + exam_lower
    
    # =========================================================================
    # FIX 31: DUPLEX vs REGULAR ULTRASOUND (Comprehensive)
    # =========================================================================
    # Duplex = vascular study with Doppler (93xxx codes)
    # Regular US = anatomic imaging (76xxx codes)
    # 
    # Key differentiators:
    # - "duplex", "doppler", "venous", "arterial", "vascular" → Duplex codes
    # - "doppler" alone in pelvic → still pelvic US (76856/76857)
    # =========================================================================
    
    # US codes that could be confused with duplex (ENHANCED for arterial)
    us_duplex_confusion = {
        # Extremity: 76882 (limited extremity US) vs 93971 (venous duplex unilat)
        '76882', '93971', '93970',
        # Pelvic: 76856/76857 (pelvic US) vs 93976 (pelvic duplex)
        '76856', '76857', '93976',
        # Scrotum: 76870 (scrotal US) vs duplex
        '76870',
        # Abdominal: 76705/76700 vs vascular
        '76705', '76700',
        # Arterial duplex codes - FIX 6 Production comparison
        '93925', '93926', '93930', '93931',
    }
    
    if predicted_cpt in us_duplex_confusion:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        
        # Check for duplex/vascular keywords
        # IMPORTANT: "WO DUPLEX" or "WITHOUT DUPLEX" means NOT a duplex study
        has_wo_duplex = 'wo duplex' in exam_title or 'without duplex' in exam_title or 'w/o duplex' in exam_title
        is_duplex = any(kw in exam_title for kw in ['duplex', 'venous doppler', 'arterial doppler', 'vascular']) and not has_wo_duplex
        is_vein_study = 'vein' in exam_title and ('lower' in exam_title or 'upper' in exam_title or 'extrem' in exam_title)
        
        # Check for regular US keywords
        is_scrotal = any(kw in exam_title for kw in ['scrot', 'testic'])
        is_pelvic = any(kw in exam_title for kw in ['pelvi', 'uterus', 'ovary', 'ovaries'])
        is_abdominal = any(kw in exam_title for kw in ['abdomen', 'abdominal', 'abd '])
        is_extremity_us = 'ultrasound' in exam_title and 'extrem' in exam_title and not is_duplex
        
        if is_duplex or is_vein_study:
            # It's a duplex/vascular study - ENHANCED FIX 6 Production comparison
            # Check for venous vs arterial
            is_venous = any(kw in exam_title for kw in ['venous', 'vein', 'veins', 'dvt', 'deep vein'])
            is_arterial = any(kw in exam_title for kw in ['arterial', 'artery', 'arteries', 'pad', 'peripheral arterial'])
            
            if 'pelvi' in exam_title:
                return '93976'  # Duplex pelvic
            # Lower extremity
            elif 'lower' in exam_title or 'leg' in exam_title or 'le ' in exam_title:
                if is_arterial and not is_venous:
                    if 'bilateral' in exam_title or 'complete' in exam_title:
                        return '93925'  # Arterial duplex bilateral lower
                    else:
                        return '93926'  # Arterial duplex unilateral lower
                else:  # Default to venous
                    if 'bilateral' in exam_title or 'complete' in exam_title:
                        return '93970'  # Venous duplex bilateral
                    else:
                        return '93971'  # Venous duplex unilateral
            # Upper extremity
            elif 'upper' in exam_title or 'arm' in exam_title or 'ue ' in exam_title:
                if is_arterial and not is_venous:
                    if 'bilateral' in exam_title:
                        return '93930'  # Arterial duplex bilateral upper
                    else:
                        return '93931'  # Arterial duplex unilateral upper
                else:  # Default to venous
                    if 'bilateral' in exam_title:
                        return '93970'  # Venous duplex bilateral
                    else:
                        return '93971'  # Venous duplex unilateral
            elif 'bilateral' in exam_title or 'complete' in exam_title:
                return '93970'  # Duplex extremity veins bilateral
            else:
                return '93971'  # Duplex extremity veins unilateral
        elif is_scrotal:
            return '76870'  # Scrotal US
        elif is_pelvic:
            if 'limited' in exam_title or 'f/u' in exam_title or 'follow' in exam_title:
                return '76857'  # Limited pelvic
            else:
                return '76856'  # Complete pelvic
        elif is_extremity_us:
            return '76882'  # Limited extremity US
    
    # =========================================================================
    # FIX 32: VIEW COUNT EXTRACTION (Regex-based)
    # =========================================================================
    # Extract view count from exam description using regex
    # Handles: "2 views", "2-view", "2VIEW", "two view", "minimum of 3", etc.
    # =========================================================================
    
    def extract_view_count(text):
        """Extract view count from text. Returns int or None."""
        # Pattern 1: "N view(s)" or "N-view"
        match = re.search(r'(\d+)\s*[-]?\s*views?', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Pattern 2: "minimum of N"
        match = re.search(r'minimum\s+of\s+(\d+)', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        # Pattern 3: Word numbers
        word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'single': 1}
        for word, num in word_to_num.items():
            if f'{word} view' in text.lower():
                return num
        
        # Pattern 4: "N or more"
        match = re.search(r'(\d+)\s+or\s+more', text, re.IGNORECASE)
        if match:
            return int(match.group(1))
        
        return None
    
    # Apply view count fixes for X-ray codes
    xray_view_codes = {
        # Chest: 71045 (1v), 71046 (2v), 71047 (3v), 71048 (4+v)
        ('71045', '71046', '71047', '71048'): {1: '71045', 2: '71046', 3: '71047', 4: '71048'},
        # Shoulder: 73020 (1v), 73030 (2+v) - FIX 2 Production comparison
        ('73020', '73030'): {1: '73020', 2: '73030', 3: '73030', 4: '73030'},
        # Elbow: 73070 (2v), 73080 (3+v)
        ('73070', '73080'): {1: '73070', 2: '73070', 3: '73080', 4: '73080'},
        # Knee: 73560 (1-2v), 73562 (3v), 73564 (4+v)
        ('73560', '73562', '73564'): {1: '73560', 2: '73560', 3: '73562', 4: '73564'},
        # Hip: 73501 (1v), 73502 (2+v), 73503 (3+v)
        ('73501', '73502', '73503'): {1: '73501', 2: '73502', 3: '73503', 4: '73503'},
        # Lumbar spine: 72100 (2-3v), 72110 (4+v)
        ('72100', '72110'): {1: '72100', 2: '72100', 3: '72100', 4: '72110'},
        # Cervical spine: 72040 (2-3v), 72052 (4+v)
        ('72040', '72052'): {1: '72040', 2: '72040', 3: '72040', 4: '72052'},
    }
    
    for code_group, view_map in xray_view_codes.items():
        if predicted_cpt in code_group:
            exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
            view_count = extract_view_count(exam_title)
            # Fallback: search full report if not found in exam title
            if not view_count and report_lower:
                view_count = extract_view_count(report_lower)
            if view_count:
                # Find the appropriate code for this view count
                if view_count >= 4:
                    return view_map.get(4, predicted_cpt)
                elif view_count == 3:
                    return view_map.get(3, predicted_cpt)
                elif view_count == 2:
                    return view_map.get(2, predicted_cpt)
                elif view_count == 1:
                    return view_map.get(1, predicted_cpt)
            break  # Only check one code group
    
    # =========================================================================
    # FIX 33: CONTRAST LEVEL DETECTION (Smart parsing)
    # =========================================================================
    # Distinguish between:
    # - "without contrast" only → w/o code
    # - "with contrast" only → w/ code  
    # - "without and with" or "w/wo" → w/wo code
    # =========================================================================
    
    def detect_contrast_level(text):
        """
        Detect contrast level from text.
        Returns: 'without', 'with', 'both', or None
        """
        text_lower = text.lower()
        
        # Check for "without and with" or "with and without" patterns first
        if any(p in text_lower for p in ['without and with', 'with and without', 'w/wo', 'wo/w', 
                                          'without then with', 'with then without']):
            return 'both'
        
        # Check for "without" patterns
        has_without = any(p in text_lower for p in ['without contrast', 'w/o contrast', 'wo contrast',
                                                     'without iv contrast', 'w/o iv', 'non-contrast',
                                                     'noncontrast', 'no contrast'])
        
        # Check for "with" patterns - but NOT if it's part of "without"
        # Use regex to find "with" not preceded by "out"
        has_with = bool(re.search(r'(?<!out)\bwith\s+(iv\s+)?contrast', text_lower))
        if not has_with:
            has_with = 'w/ contrast' in text_lower or 'w/contrast' in text_lower
        
        if has_without and has_with:
            return 'both'
        elif has_without:
            return 'without'
        elif has_with:
            return 'with'
        
        return None
    
    # MRI Brain contrast codes: 70551 (w/o), 70552 (w/), 70553 (w/wo)
    if predicted_cpt in ['70551', '70552', '70553']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        contrast = detect_contrast_level(exam_title)
        if contrast == 'without':
            return '70551'
        elif contrast == 'with':
            return '70552'
        elif contrast == 'both':
            return '70553'
    
    # MRI Spine contrast codes
    mri_spine_contrast = {
        # Cervical: 72141 (w/o), 72142 (w/), 72156 (w/wo)
        ('72141', '72142', '72156'): {'without': '72141', 'with': '72142', 'both': '72156'},
        # Thoracic: 72146 (w/o), 72147 (w/), 72157 (w/wo)
        ('72146', '72147', '72157'): {'without': '72146', 'with': '72147', 'both': '72157'},
        # Lumbar: 72148 (w/o), 72149 (w/), 72158 (w/wo)
        ('72148', '72149', '72158'): {'without': '72148', 'with': '72149', 'both': '72158'},
    }
    
    for code_group, contrast_map in mri_spine_contrast.items():
        if predicted_cpt in code_group:
            exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
            contrast = detect_contrast_level(exam_title)
            if contrast and contrast in contrast_map:
                return contrast_map[contrast]
            break
    
    # MRI Lower Extremity: 73718 (w/o), 73719 (w/), 73720 (w/wo)
    if predicted_cpt in ['73718', '73719', '73720']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        contrast = detect_contrast_level(exam_title)
        if contrast == 'without':
            return '73718'
        elif contrast == 'with':
            return '73719'
        elif contrast == 'both':
            return '73720'
    
    # =========================================================================
    # END OF NEW FIXES 31-33
    # =========================================================================
    
    # Fix 1: 77063 vs 77067 (tomosynthesis vs regular screening)
    # For combined studies (MAMMO + US), check the PRIMARY procedure only
    if predicted_cpt in ['77062', '77063', '77067', '77061', '77065', '77066']:
        # Get primary exam (first procedure before comma/semicolon)
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        primary_exam = exam_title.split(',')[0].split(';')[0].strip()
        
        # If primary exam says MAMMO, keep it as mammography
        is_primary_mammo = 'mammo' in primary_exam or 'mam ' in primary_exam
        
        # Only switch to US if primary exam is NOT mammography
        if not is_primary_mammo:
            is_us = ('ultrasound' in primary_exam or 'sonograph' in primary_exam or
                     primary_exam.startswith('us ') or ' us ' in primary_exam)
            if is_us:
                if 'unilateral' in exam_lower or 'left' in exam_lower or 'right' in exam_lower:
                    return '76641'
                else:
                    return '76642'
        
        # Check for tomosynthesis in primary exam - ENHANCED FIX 7 Production comparison
        # IMPORTANT: Do NOT blindly trust LLM tomo predictions - check exam description
        # Fix for Sample 29: "Screening mammography" was being coded as 77063 (tomo) instead of 77067
        tomo_keywords = [
            'tomosynthesis', 'tomo ', 'tomo,', 'tomo)', 
            'dbt', '3d mammo', '3-d mammo', '3d breast', 
            '3d ', '3-d ', 'three-dimensional',
            'w/cad', 'ffd mam', 'digital breast tomo',
            'synthesized', 'c-view'
        ]
        has_tomo = any(kw in primary_exam for kw in tomo_keywords)
        
        # Also check full exam_lower for tomo keywords (not just primary exam)
        if not has_tomo:
            has_tomo = any(kw in exam_lower for kw in tomo_keywords)
        
        if has_tomo:
            is_unilateral = 'unilateral' in primary_exam or 'unilat' in primary_exam
            if 'diagnostic' in primary_exam:
                return '77061' if is_unilateral else '77062'
            else:
                return '77061' if is_unilateral else '77063'
        else:
            if 'diagnostic' in primary_exam:
                return '77066'
            else:
                return '77067'
    
    # Fix 2: Pelvic US - Complete vs Limited vs Transvaginal (ENHANCED - FIX 3 Production comparison)
    # 76856 = Complete pelvic (includes TA+TV when both performed)
    # 76857 = Limited pelvic
    # 76830 = Transvaginal only (non-OB)
    if predicted_cpt in ['76856', '76857', '76830']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        
        # FIRST: Check if this is actually an OB exam (route to OB codes instead)
        ob_keywords = ['ob ', 'ob,', 'obstetric', 'pregnancy', 'fetal', 'fetus', 'gestational', 
                       'dating scan', 'viability', 'trimester', 'biophysical']
        is_ob = any(kw in exam_title for kw in ob_keywords) or any(kw in combined for kw in ob_keywords)
        
        if is_ob:
            # Route to OB ultrasound codes
            has_tv = 'transvaginal' in combined or 'endovaginal' in combined or 'tv ' in exam_title
            has_limited = any(kw in combined for kw in ['limited', 'f/u', 'follow-up', 'followup', 'interval'])
            has_first_tri = any(kw in combined for kw in ['dating', 'first trimester', '1st trimester', 'crl', 'crown rump', 'early pregnancy'])
            has_bpp = any(kw in combined for kw in ['biophysical', 'bpp', 'nst'])
            
            if has_bpp:
                return '76818'
            elif has_limited:
                return '76817' if has_tv else '76815'
            elif has_tv:
                return '76817'
            elif has_first_tri:
                return '76801'
            else:
                return '76805'  # Default to 2nd/3rd trimester
        
        # Check for both TA and TV components
        has_ta = any(kw in exam_title for kw in ['transabdominal', 'trans-abdominal', 'ta ', 'ta,', 'ta/'])
        has_tv = any(kw in exam_title for kw in ['transvaginal', 'trans-vaginal', 'tv ', 'tv,', 'tv/', 'endovaginal'])
        # Also check if "transvaginal" appears anywhere without TA
        if 'transvaginal' in exam_title and not has_ta:
            has_tv = True
        has_both = has_ta and has_tv
        
        # Check for "and" between TA and TV
        if 'transabdominal and transvaginal' in exam_title or 'ta and tv' in exam_title or 'ta/tv' in exam_title:
            has_both = True
        
        # Check keywords
        limited_keywords = ['limited', 'focused', 'f/u', 'follow-up']
        complete_keywords = ['complete', 'comprehensive', 'full']
        tv_only_keywords = ['tv only', 'transvaginal only', 'endovaginal only']
        
        has_limited = any(kw in combined for kw in limited_keywords)
        has_complete = any(kw in combined for kw in complete_keywords)
        has_tv_only = any(kw in combined for kw in tv_only_keywords)
        
        # Decision logic
        if has_both or has_complete:
            return '76856'  # Complete pelvic (TA+TV)
        elif has_tv_only or (has_tv and not has_ta):
            return '76830'  # Transvaginal only
        elif has_limited:
            return '76857'  # Limited pelvic
        else:
            return '76856'  # Default to complete
    
    # Fix 34: OB Ultrasound - FIX 1B Production comparison
    # 76801 = First trimester dating (<14 weeks) with biometry
    # 76805 = Second/third trimester (>=14 weeks)
    # 76815 = Limited (viability check, follow-up)
    # 76817 = Transvaginal OB component
    if predicted_cpt in ['76801', '76802', '76805', '76810', '76811', '76815', '76816', '76817', '76818', '76819']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        
        # Check for first trimester dating indicators
        first_trimester_keywords = [
            'dating', 'viability', 'first trimester', '1st trimester',
            'crl', 'crown rump', 'nuchal', 'nt measurement',
            'early pregnancy', 'gestational age'
        ]
        has_first_trimester = any(kw in combined for kw in first_trimester_keywords)
        
        # Check for transvaginal
        has_tv = any(kw in combined for kw in ['transvaginal', 'endovaginal', 'tv ', 'endo-vaginal'])
        
        # Check for limited/follow-up
        has_limited = any(kw in combined for kw in ['limited', 'f/u', 'follow-up', 'followup', 'interval'])
        
        # Check for biophysical profile
        has_bpp = any(kw in combined for kw in ['biophysical', 'bpp', 'nst', 'non-stress'])
        
        # Decision logic
        if has_bpp:
            return '76818'  # Biophysical profile
        elif has_first_trimester and not has_limited:
            return '76801'  # First trimester dating
        elif has_limited:
            if has_tv:
                return '76817'  # TV limited OB
            return '76815'  # Limited OB
        elif has_tv:
            return '76817'  # Transvaginal OB
        
        # Check weeks if mentioned to determine trimester
        week_match = re.search(r'(\d+)\s*weeks?', combined)
        if week_match:
            weeks = int(week_match.group(1))
            if weeks < 14:
                return '76801'  # First trimester
            else:
                return '76805'  # Second/third trimester
    
    # Fix 3: 74018 vs 74019 (1 view vs 2 view abdomen)
    if predicted_cpt in ['74018', '74019', '74021']:
        if '2 view' in combined or 'two view' in combined or 'supine and upright' in combined:
            return '74019'
        elif '3 view' in combined or 'acute series' in combined:
            return '74021'
        elif '1 view' in combined or 'single view' in combined or 'supine only' in combined:
            return '74018'
    
    # Fix 4: 70496 vs 70498 (head vs neck CTA)
    if predicted_cpt in ['70496', '70498']:
        # Check exam_lower primarily for body part
        head_keywords = ['brain', 'intracranial', 'circle of willis', 'head', 'cta head']
        neck_keywords = ['carotid', 'neck', 'cta neck', 'cervical artery']
        # Prioritize exam description
        has_head = any(kw in exam_lower for kw in head_keywords)
        has_neck = any(kw in exam_lower for kw in neck_keywords)
        if 'head and neck' in exam_lower or 'neck and head' in exam_lower:
            return '70498'  # Combined study - typically billed as neck
        elif has_head and not has_neck:
            return '70496'  # Head CTA
        elif has_neck and not has_head:
            return '70498'  # Neck CTA
        elif ', head' in exam_lower or 'head,' in exam_lower:
            return '70496'  # CTA head
    
    # Fix 5: Chest CT vs Abdomen CT (expanded to include 71250)
    # IMPORTANT: Only check exam_lower for body part - report may mention other body parts in findings
    if predicted_cpt in ['71250', '71260', '74176', '74177', '74178']:
        # Check primary exam focus from exam description ONLY (not report)
        is_chest_exam = any(kw in exam_lower for kw in ['chest', 'lung', 'thorax', 'pulmonary'])
        is_abd_exam = any(kw in exam_lower for kw in ['abdomen', 'pelvis', 'abdominal'])
        
        if is_chest_exam and not is_abd_exam:
            # Chest CT - determine contrast status
            # FIX: Check for contrast phrases, not just word presence
            has_without = any(p in exam_lower for p in ['without contrast', 'w/o contrast', 'wo contrast'])
            has_with = any(p in exam_lower for p in ['with contrast', 'w/ contrast', 'with iv contrast'])
            if has_with and not has_without:
                return '71260'  # Chest with contrast
            else:
                return '71250'  # Chest without contrast
        if is_abd_exam and not is_chest_exam:
            # Abdomen/Pelvis CT - determine contrast status
            # FIX: Check for "without contrast" as a PHRASE, not just both words anywhere
            # This prevents "Abd pain without fever" + "WITH IV CONTRAST" from matching as "without"
            has_without = any(p in exam_lower for p in ['without contrast', 'w/o contrast', 'wo contrast'])
            has_with = any(p in exam_lower for p in ['with contrast', 'w/ contrast', 'with iv contrast'])
            has_both = 'without and with' in exam_lower or 'with and without' in exam_lower
            
            if has_both:
                return '74178'  # Without and with contrast
            elif has_without and not has_with:
                return '74176'  # Without contrast only
            elif has_with and not has_without:
                return '74177'  # With contrast only
            elif has_with:
                return '74177'  # Default to with if both detected (edge case)
            else:
                return '74177'  # Default to with contrast if unspecified
    
    # Fix 6: Chest X-ray view count (71045 vs 71046)
    if predicted_cpt in ['71045', '71046', '71047', '71048']:
        # Check for view count in exam description
        if '1 view' in combined or '1view' in combined or 'single view' in combined or 'one view' in combined:
            return '71045'  # 1 view
        elif '2 view' in combined or '2view' in combined or 'two view' in combined or 'frontal and lateral' in combined:
            return '71046'  # 2 views
        elif '3 view' in combined or '3view' in combined:
            return '71047'  # 3 views
        elif '4 view' in combined or '4view' in combined or '4 or more' in combined:
            return '71048'  # 4+ views
    
    # Fix 7: Knee X-ray view count (73560, 73562, 73564)
    if predicted_cpt in ['73560', '73562', '73564']:
        if '1 view' in combined or '1view' in combined or 'single' in combined:
            return '73560'  # 1-2 views
        elif '3 view' in combined or '3view' in combined or 'three view' in combined:
            return '73562'  # 3 views
        elif '4 view' in combined or '4view' in combined or '4 or more' in combined or 'complete' in combined:
            return '73564'  # 4+ views
    
    # Fix 8: REMOVED - consolidated into Fix 18 for better mammography handling
    
    # Fix 9: Upper extremity body part confusion (73060 humerus vs others)
    if predicted_cpt in ['73060', '73600', '73590', '73610', '73620', '73630']:
        # Check body part keywords in exam_lower first (more reliable)
        if any(kw in exam_lower for kw in ['humerus', 'upper arm']):
            return '73060'  # Humerus
        elif any(kw in exam_lower for kw in ['ankle']):
            # Ankle X-ray - check view count
            if '3 view' in exam_lower or '3 or more' in exam_lower or '3view' in exam_lower:
                return '73610'  # Ankle 3+ views (complete)
            else:
                return '73600'  # Ankle 2 views (minimum)
        elif any(kw in exam_lower for kw in ['tibia', 'fibula', 'tib fib', 'tib/fib']):
            return '73590'  # Tibia/fibula
        elif any(kw in exam_lower for kw in ['foot']):
            # Foot X-ray - check view count
            if '3 view' in exam_lower or '3 or more' in exam_lower:
                return '73630'  # Foot 3+ views (complete)
            else:
                return '73620'  # Foot 2 views (minimum)
    
    # Fix 30: CTA Abd/Pelvis detection (74174 vs 74177) - ENHANCED FIX 4 Production comparison
    # Fix for Sample 8: "Computed tomographic angiography, abdomen and pelvis" should be 74174, not 74177
    # If LLM predicts regular CT but exam says "angiography", change to CTA
    if predicted_cpt in ['74176', '74177', '74178']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        
        # Expanded CTA detection keywords
        cta_keywords = [
            'angiography', 'angiogram', 'arteriography',
            'cta ', 'cta,', 'cta)', 'ct angio', 'ct-angio',
            'angio ', 'angio,', 'angio)', 
            'ct angiography', 'computed tomographic angiography',
            'runoff', 'aortogram'
        ]
        is_cta = any(kw in exam_title for kw in cta_keywords)
        is_abd = 'abdomen' in exam_title or 'abd' in exam_title
        is_pelvis = 'pelvis' in exam_title
        
        if is_cta and (is_abd or is_pelvis):
            return '74174'  # CTA abdomen and pelvis
    
    # Fix 10: CT Abd/Pelvis contrast variants (74176 vs 74177 vs 74178)
    if predicted_cpt in ['74176', '74177', '74178']:
        # Check contrast info from exam TITLE only (first line) to avoid false matches
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        
        # Check for w/o or without
        has_wo = 'w/o' in exam_title or 'without' in exam_title
        # Check for "with" but NOT inside "without" - use word boundary
        has_w = bool(re.search(r'(?<!out)\bwith\b', exam_title)) or 'w/ ' in exam_title
        has_both = has_wo and has_w
        
        if has_both:
            return '74178'  # Without and with contrast
        elif has_wo:
            return '74176'  # Without contrast
        elif has_w:
            return '74177'  # With contrast
    
    # Fix 11: Elbow X-ray views (73070 vs 73080)
    if predicted_cpt in ['73070', '73080']:
        if any(kw in exam_lower for kw in ['3 view', '3view', 'complete', '3 or more']):
            return '73080'  # Elbow complete (3+ views)
        else:
            return '73070'  # Elbow 2 views (minimum)
    
    # Fix 12: Wrist vs Forearm X-ray (73090 vs 73100 vs 73110)
    if predicted_cpt in ['73090', '73100', '73110']:
        # Check body part FIRST - forearm is different from wrist
        if 'forearm' in exam_lower:
            return '73090'  # Forearm 2 views
        elif 'wrist' in exam_lower:
            if any(kw in exam_lower for kw in ['3 view', '3view', 'complete', 'minimum of 3']):
                return '73110'  # Wrist complete (3+ views)
            else:
                return '73100'  # Wrist 2 views
    
    # Fix 13: Pelvic US vs Duplex (76856 vs 93976)
    if predicted_cpt in ['76856', '76830', '93976']:
        # Check if it's a duplex study
        if any(kw in exam_lower for kw in ['duplex', 'doppler', 'venous', 'arterial']):
            return '93976'  # Duplex
        elif any(kw in exam_lower for kw in ['pelvic', 'uterus', 'ovary', 'ovaries']):
            if 'transvaginal' in exam_lower or 'endovaginal' in exam_lower or 'tv' in exam_lower:
                return '76830'  # Transvaginal
            else:
                return '76856'  # Transabdominal pelvic
    
    # Fix 14: Spine CT vs MR (72125-72133 vs 72141-72158)
    # CT spine: 72125 (cervical), 72128 (thoracic), 72131 (lumbar)
    # MR spine: 72141 (cervical), 72146 (thoracic), 72148 (lumbar)
    if predicted_cpt in ['72125', '72128', '72131', '72132', '72133', '72141', '72146', '72148', '72149']:
        is_ct = any(kw in exam_lower for kw in ['ct ', 'ct,', 'computed tomography'])
        is_mr = any(kw in exam_lower for kw in ['mr ', 'mri', 'magnetic resonance'])
        
        if is_ct and not is_mr:
            # CT Spine - determine level and contrast
            is_lumbar = any(kw in exam_lower for kw in ['lumbar', 'l-spine', 'lspine', 'ls spine'])
            is_thoracic = any(kw in exam_lower for kw in ['thoracic', 't-spine', 'tspine', 'ts spine'])
            is_cervical = any(kw in exam_lower for kw in ['cervical', 'c-spine', 'cspine'])
            has_contrast = 'with' in exam_lower and 'contrast' in exam_lower and 'without' not in exam_lower
            
            if is_lumbar:
                if has_contrast:
                    return '72132'  # CT lumbar with contrast
                else:
                    return '72131'  # CT lumbar without contrast
            elif is_thoracic:
                if has_contrast:
                    return '72129'  # CT thoracic with contrast
                else:
                    return '72128'  # CT thoracic without contrast
            elif is_cervical:
                if has_contrast:
                    return '72126'  # CT cervical with contrast
                else:
                    return '72125'  # CT cervical without contrast
        elif is_mr and not is_ct:
            # MRI Spine - determine level and contrast
            # IMPORTANT: 'neck' = cervical spine (Fix for Sample 45)
            is_lumbar = any(kw in exam_lower for kw in ['lumbar', 'l-spine', 'lspine', 'ls spine'])
            is_thoracic = any(kw in exam_lower for kw in ['thoracic', 't-spine', 'tspine', 'ts spine'])
            is_cervical = any(kw in exam_lower for kw in ['cervical', 'c-spine', 'cspine', 'neck spine', 'neck '])
            has_contrast = 'with' in exam_lower and 'contrast' in exam_lower and 'without' not in exam_lower
            
            if is_cervical:
                return '72142' if has_contrast else '72141'  # MRI cervical
            elif is_thoracic:
                return '72147' if has_contrast else '72146'  # MRI thoracic
            elif is_lumbar:
                return '72149' if has_contrast else '72148'  # MRI lumbar
            # NO default - let LLM prediction stand if no spine region detected
    
    # Fix 15: Elbow X-ray - distinguish from humerus (73060 vs 73070)
    if predicted_cpt in ['73060', '73070']:
        if any(kw in exam_lower for kw in ['elbow']):
            return '73070'  # Elbow
        elif any(kw in exam_lower for kw in ['humerus', 'upper arm']):
            return '73060'  # Humerus
    
    # Fix 16: Ribs X-ray view count (71100 vs 71101)
    if predicted_cpt in ['71100', '71101', '71110', '71111']:
        if '2 view' in exam_lower or '2view' in exam_lower or 'two view' in exam_lower:
            if 'unilateral' in exam_lower:
                return '71100'  # Ribs unilateral, 2 views
            else:
                return '71110'  # Ribs bilateral, 2 views
        elif '3 view' in exam_lower or '3view' in exam_lower or '3 or more' in exam_lower or 'including oblique' in exam_lower:
            if 'unilateral' in exam_lower:
                return '71101'  # Ribs unilateral, 3+ views
            else:
                return '71111'  # Ribs bilateral, 3+ views
    
    # Fix 17: Sinus/Maxillofacial CT (70486 vs 70487 vs 70488)
    # BUT first check if it's actually MRI (not CT) - if so, skip this fix
    if predicted_cpt in ['70486', '70487', '70488']:
        is_mri = 'mri' in exam_lower or 'magnetic' in exam_lower or 'mr ' in exam_lower
        if not is_mri:
            # It's CT - apply contrast logic
            if 'without and with' in exam_lower or 'with and without' in exam_lower:
                return '70488'  # Without and with contrast
            elif 'with contrast' in exam_lower and 'without' not in exam_lower:
                return '70487'  # With contrast only
            else:
                return '70486'  # Without contrast
    
    # Fix 18: Breast US vs Mammography (76641/76642 vs 77061/77062/77063)
    if predicted_cpt in ['76641', '76642', '77061', '77062', '77063', '77065', '77066', '77067']:
        is_us = any(kw in exam_lower for kw in ['ultrasound', 'us breast', 'sonograph', 'breast us'])
        is_mammo = any(kw in exam_lower for kw in ['mammogra', 'mammo ', ' mam ', 'mam screen', 'screening mammo', 'w/cad'])
        
        if is_us and not is_mammo:
            # It's breast ultrasound
            # Fix for Sample 6: "ULTRASOUND BREAST COMPLETE" should be 76641, not 76642
            # 76641 = Complete (unilateral OR complete exam)
            # 76642 = Limited (targeted/focused exam)
            if 'complete' in exam_lower or 'unilateral' in exam_lower or 'left' in exam_lower or 'right' in exam_lower:
                return '76641'  # Breast US complete/unilateral
            else:
                return '76642'  # Breast US limited
        elif is_mammo and not is_us:
            # It's mammography - check for tomosynthesis first
            has_tomo = any(kw in exam_lower for kw in ['tomosynthesis', 'tomo '])
            is_diagnostic = 'diagnostic' in exam_lower
            is_unilateral = 'unilateral' in exam_lower or 'single breast' in exam_lower
            
            if has_tomo:
                if is_diagnostic:
                    if is_unilateral:
                        return '77061'  # Diagnostic tomo unilateral
                    else:
                        return '77062'  # Diagnostic tomo bilateral
                else:
                    return '77063'  # Screening tomo
            elif is_diagnostic:
                if is_unilateral:
                    return '77065'  # Diagnostic mammo unilateral
                else:
                    return '77066'  # Diagnostic mammo bilateral
            else:
                return '77067'  # Screening mammo bilateral
    
    # Fix 19: Chest X-ray vs Abdominal US (71045 vs 76705)
    if predicted_cpt in ['71045', '71046', '76705']:
        is_chest_xr = any(kw in exam_lower for kw in ['chest', 'xr chest', 'x-ray chest', 'radiologic examination, chest'])
        is_abd_us = any(kw in exam_lower for kw in ['abdominal ultrasound', 'us abd', 'us abdomen'])
        
        if is_chest_xr and not is_abd_us:
            if '1 view' in exam_lower or 'single' in exam_lower:
                return '71045'
            else:
                return '71046'
        elif is_abd_us and not is_chest_xr:
            return '76705'
    
    # Fix 20: CTA chest vs spine CT vs CTA abdomen (71275 vs 72128 vs 74174/74175)
    if predicted_cpt in ['71275', '72128', '72129', '74174', '74175']:
        is_spine = any(kw in exam_lower for kw in ['spine', 'thoracic spine', 't-spine'])
        is_cta = 'cta' in exam_lower or 'angiography' in exam_lower
        is_chest = any(kw in exam_lower for kw in ['chest', 'aorta', 'pulmonary'])
        is_abd = any(kw in exam_lower for kw in ['abdomen', 'pelvis', 'abdominal', 'renal'])
        
        if is_spine and not is_cta:
            has_contrast = 'with' in exam_lower and 'contrast' in exam_lower and 'without' not in exam_lower
            if has_contrast:
                return '72129'  # CT thoracic with contrast
            else:
                return '72128'  # CT thoracic without contrast
        elif is_cta:
            # CTA - determine body region
            if is_abd and not is_chest:
                return '74174'  # CTA abdomen/pelvis
            elif is_chest and not is_abd:
                return '71275'  # CTA chest
            elif is_abd and is_chest:
                return '74175'  # CTA chest abd pelvis combined
    
    # Fix 21: Vascular embolization (37244) vs arterial catheterization (36245/36247)
    if predicted_cpt in ['36245', '36246', '36247', '37244']:
        is_embolization = any(kw in exam_lower for kw in ['embolization', 'occlusion', 'embolize'])
        is_catheter = any(kw in exam_lower for kw in ['catheter', 'angiogram', 'arteriogram'])
        
        if is_embolization:
            return '37244'  # Vascular embolization
        elif is_catheter:
            return '36247'  # Arterial catheterization, 3rd order
    
    # Fix 22: Lower extremity body part confusion (knee vs ankle vs femur)
    if predicted_cpt in ['73560', '73562', '73564', '73610', '73630', '73552', '73550']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        
        if 'knee' in exam_title:
            if '3' in exam_title or 'three' in exam_title:
                return '73562'  # Knee 3 views
            else:
                return '73560'  # Knee 1-2 views
        elif 'ankle' in exam_title:
            return '73610'  # Ankle
        elif 'foot' in exam_title or 'feet' in exam_title:
            return '73630'  # Foot
        elif 'femur' in exam_title or 'thigh' in exam_title:
            return '73552'  # Femur 2+ views
        elif 'tibia' in exam_title or 'fibula' in exam_title or 'leg' in exam_title:
            return '73590'  # Tibia/Fibula
    
    # Fix 35: Hip X-ray Unilateral vs Bilateral - FIX 5 Production comparison
    # 73501 = Hip 1 view unilateral
    # 73502 = Hip 2+ views unilateral  
    # 73503 = Hip 3+ views unilateral
    # 73521 = Hips bilateral 2 views
    # 73522 = Hips bilateral 3-4 views
    # 73523 = Hips bilateral 5+ views
    if predicted_cpt in ['73501', '73502', '73503', '73521', '73522', '73523']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        
        # Check laterality
        is_unilateral = any(kw in exam_title for kw in ['right', 'left', 'rt ', 'lt ', 'unilateral', 'single'])
        is_bilateral = any(kw in exam_title for kw in ['bilateral', 'both', 'hips', 'b/l'])
        
        # Check view count using the extract_view_count function defined earlier
        view_count = extract_view_count(exam_title)
        
        if is_bilateral and not is_unilateral:
            # Bilateral hip
            if view_count and view_count >= 5:
                return '73523'
            elif view_count and view_count >= 3:
                return '73522'
            else:
                return '73521'
        else:
            # Unilateral hip (default if not explicitly bilateral)
            if view_count and view_count >= 3:
                return '73503'
            elif view_count and view_count >= 2:
                return '73502'
            else:
                return '73501'
    
    # Fix 23: Spine X-ray region confusion (thoracic vs lumbar vs cervical)
    if predicted_cpt in ['72020', '72040', '72052', '72070', '72072', '72074', '72100', '72110', '72114']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        
        # Check which region is mentioned - prioritize explicit mentions
        is_thoracic = 'thoracic' in exam_title or 't spine' in exam_title or 't-spine' in exam_title
        is_lumbar = 'lumbar' in exam_title or 'l spine' in exam_title or 'l-spine' in exam_title or 'lumbosacral' in exam_title
        is_cervical = 'cervical' in exam_title or 'c spine' in exam_title or 'c-spine' in exam_title
        
        # If exam explicitly says thoracic but prediction is lumbar, fix it
        if is_thoracic and not is_lumbar:
            if '2 view' in exam_title or 'two view' in exam_title or '2view' in exam_title:
                return '72070'  # Thoracic spine 2 views
            elif '3' in exam_title or 'three' in exam_title or 'complete' in exam_title:
                return '72072'  # Thoracic spine 3 views
            else:
                return '72070'  # Default thoracic 2 views
        elif is_lumbar and not is_thoracic:
            # Check 4+ views FIRST (before checking for 'minimum' which could be 'minimum of 4')
            if '4 view' in exam_title or 'minimum of 4' in exam_title or '4 or more' in exam_title or 'complete' in exam_title:
                return '72110'  # Lumbar spine 4+ views
            elif '2 view' in exam_title or 'two view' in exam_title or '2-3' in exam_title or 'minimum of 2' in exam_title:
                return '72100'  # Lumbar spine 2-3 views
            else:
                return '72100'  # Default lumbar
        elif is_cervical:
            if '2 view' in exam_title or 'two view' in exam_title:
                return '72040'  # Cervical spine 2-3 views
            else:
                return '72052'  # Cervical spine 4+ views
    
    # Fix 24: Breast US predicted but exam is mammography
    if predicted_cpt in ['76641', '76642']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        is_mammo = 'mammo' in exam_title or 'mam ' in exam_title or 'screening' in exam_title and 'breast' in exam_title
        if is_mammo and 'ultrasound' not in exam_title and ' us ' not in exam_title:
            # It's mammography, not ultrasound
            is_tomo = 'tomo' in exam_title or '3d' in exam_title
            is_diagnostic = 'diagnostic' in exam_title
            is_unilateral = 'unilateral' in exam_title or 'unilat' in exam_title
            
            if is_tomo:
                if is_diagnostic:
                    return '77061' if is_unilateral else '77062'  # Diagnostic tomo (77061=unilat, 77062=bilat)
                else:
                    return '77061' if is_unilateral else '77063'  # Screening tomo (uses same unilat code)
            else:
                if is_diagnostic:
                    return '77065' if is_unilateral else '77066'  # Diagnostic mammo
                else:
                    return '77067'  # Screening mammo bilateral
    
    # Fix 25: MRI predicted as CT (or vice versa)
    if predicted_cpt in ['70450', '70460', '70470', '70486', '70487', '70488']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        is_mri = 'mri' in exam_title or 'magnetic' in exam_title or 'mr ' in exam_title
        if is_mri:
            # It's MRI, not CT
            if 'brain' in exam_title:
                if 'without' in exam_title and 'with' not in exam_title.replace('without', ''):
                    return '70551'  # MRI brain w/o
                elif 'with' in exam_title and 'without' in exam_title:
                    return '70553'  # MRI brain w/wo
                else:
                    return '70553'  # Default MRI brain w/wo
            elif 'tmj' in exam_title or 'temporomandibular' in exam_title:
                return '70336'  # MRI TMJ
    
    # Fix 26: CT Angio chest vs abd/pelvis
    if predicted_cpt in ['74174', '74175']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        is_chest = 'chest' in exam_title
        is_abd = 'abd' in exam_title or 'pelvis' in exam_title
        if is_chest and not is_abd:
            return '71275'  # CT Angio chest
    
    # Fix 27: MRI spine region (cervical vs thoracic vs lumbar)
    # IMPORTANT: 'neck' = cervical spine (Fix for Sample 45)
    if predicted_cpt in ['72141', '72142', '72146', '72147', '72148', '72149', '72156', '72157', '72158']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        is_cervical = 'cervical' in exam_title or 'c-spine' in exam_title or 'c spine' in exam_title or 'neck spine' in exam_title or 'neck ' in exam_title
        is_thoracic = 'thoracic' in exam_title or 't-spine' in exam_title or 't spine' in exam_title
        is_lumbar = 'lumbar' in exam_title or 'l-spine' in exam_title or 'l spine' in exam_title
        # Use word boundary to avoid "with" matching inside "without"
        has_wo = 'without' in exam_title or 'w/o' in exam_title
        has_w = bool(re.search(r'(?<!out)\bwith\b', exam_title)) or 'w/' in exam_title.replace('w/o', '')
        has_both = has_wo and has_w
        has_contrast = has_w and not has_wo
        
        if is_cervical and not is_lumbar:
            if has_both:
                return '72156'  # MRI cervical w/wo
            elif has_contrast:
                return '72142'  # MRI cervical w/
            else:
                return '72141'  # MRI cervical w/o
        elif is_thoracic and not is_lumbar:
            if has_both:
                return '72157'  # MRI thoracic w/wo
            elif has_contrast:
                return '72147'  # MRI thoracic w/
            else:
                return '72146'  # MRI thoracic w/o
        elif is_lumbar:
            if has_both:
                return '72158'  # MRI lumbar w/wo
            elif has_contrast:
                return '72149'  # MRI lumbar w/
            else:
                return '72148'  # MRI lumbar w/o
    
    # Fix 28: Lumbar spine X-ray view count (72100 vs 72110)
    if predicted_cpt in ['72100', '72110', '72114']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        # Check for 4+ views patterns
        if '4 view' in exam_title or '4 or more' in exam_title or 'minimum of 4' in exam_title or 'min 4' in exam_title:
            return '72110'  # 4+ views
        elif '2 view' in exam_title or '3 view' in exam_title or 'minimum of 2' in exam_title or '2-3' in exam_title:
            return '72100'  # 2-3 views
    
    # Fix 29: Forearm X-ray (73090 vs 73100)
    if predicted_cpt in ['73090', '73100']:
        exam_title = exam_lower.split('\\n')[0] if '\\n' in exam_lower else exam_lower.split('\n')[0]
        # Check for 2 views pattern (including "2 views" with 's')
        if '2 view' in exam_title or 'two view' in exam_title or ', 2' in exam_title:
            return '73090'  # 2 views
        elif 'complete' in exam_title or 'minimum of 3' in exam_title:
            return '73100'  # Complete/3+ views
    
    return predicted_cpt


@dataclass
class PreprocessedData:
    """Output from preprocessing step."""
    
    # Original input
    report_text: str
    
    # Extracted CPT (if found in report)
    extracted_cpt: Optional[str] = None
    
    # Extracted features
    modality: str = "unknown"
    body_part: str = "unknown"
    laterality: str = "none"  # 'left', 'right', 'bilateral', 'none'
    has_contrast: bool = False
    
    # Candidates for LLM
    icd_candidates: List[Tuple[str, str]] = field(default_factory=list)  # [(code, description), ...]
    
    # RAG-based CPT candidates (from similar cases)
    rag_cpt_candidates: List[str] = field(default_factory=list)
    
    # RAG match score (0-100, how similar is the closest match in the index)
    rag_match_score: float = 0.0
    
    # Few-shot examples
    examples: List[Dict] = field(default_factory=list)  # [{"report": ..., "cpt": ..., "icd": ...}, ...]
    
    # Built prompt
    prompt: str = ""


class Preprocessor:
    """
    Preprocesses radiology reports for the LLM step.
    
    Uses RAG to find similar cases and extract candidates:
    - RAG-based CPT candidate extraction
    - RAG-based ICD candidate extraction
    - Rules-based candidate ranking
    - ICD descriptions in prompt
    """
    
    def __init__(
        self, 
        df=None, 
        icd_descriptions: Dict[str, str] = None, 
        rules: Dict = None,
        use_rag: bool = True,
        rag_use_reranker: bool = True  # Cross-encoder reranking is REQUIRED in production
    ):
        """
        Args:
            df: Training data DataFrame with Procedure, ICD10 - Diagnosis, Report columns
            icd_descriptions: Dict mapping ICD code to description
            rules: Dict of pattern rules for candidate ranking
            use_rag: Whether to use RAG for candidate generation
            rag_use_reranker: Whether to use cross-encoder reranking (slower)
        """
        self.df = df
        self.icd_descriptions = icd_descriptions or {}
        self.rules = rules or {}
        self.use_rag = use_rag
        
        # Build indexes from training data
        self.cpt_to_icd: Dict[str, Set[str]] = {}
        self.top_500_codes: Set[str] = set()
        self.code_frequency: Dict[str, int] = {}
        
        if df is not None:
            self._build_indexes()
        
        # Initialize smart section extractor
        self.section_extractor = None
        if HAS_EXTRACTOR:
            try:
                self.section_extractor = ReportSectionExtractor()
            except Exception as e:
                print(f"Warning: Could not initialize section extractor: {e}")
        
        # Initialize RAG retriever
        self.rag_retriever = None
        if use_rag and HAS_RAG:
            try:
                # Cross-encoder reranking is not optional for production quality.
                self.rag_retriever = get_rag_retriever(use_reranker=True)
                if self.rag_retriever and self.rag_retriever.loaded:
                    print("RAG retriever loaded successfully")
                else:
                    print("RAG retriever not available, falling back to lookup")
                    self.rag_retriever = None
            except Exception as e:
                print(f"Error loading RAG retriever: {e}")
                self.rag_retriever = None
    
    def _build_indexes(self):
        """Build CPT→ICD mapping and top 500 codes from training data."""
        all_codes = []
        
        for _, row in self.df.iterrows():
            cpt = str(row.get('Procedure', ''))
            icd = str(row.get('ICD10 - Diagnosis', ''))
            
            if cpt and cpt != 'nan' and icd and icd != 'nan':
                if cpt not in self.cpt_to_icd:
                    self.cpt_to_icd[cpt] = set()
                
                for c in icd.split(','):
                    code = c.strip().upper()
                    if code and len(code) >= 3:
                        self.cpt_to_icd[cpt].add(code)
                        all_codes.append(code)
        
        self.code_frequency = Counter(all_codes)
        self.top_500_codes = set(c for c, _ in self.code_frequency.most_common(500))
    
    def _get_rule_scores(self, report: str, cpt: str) -> Dict[str, float]:
        """
        Get rule-based relevance scores for codes.
        Matches BestInference logic.
        """
        if not self.rules:
            return {}
        
        report_lower = report.lower()
        code_scores = {}
        
        for pattern_key, rule_data in self.rules.items():
            parts = pattern_key.split('|')
            pattern = parts[0]
            rule_cpt = parts[1] if len(parts) > 1 else None
            
            if pattern in report_lower:
                weight = 2.0 if rule_cpt == cpt else 1.0
                for code_info in rule_data.get('codes', []):
                    code = code_info.get('code', '')
                    conf = code_info.get('confidence', 0.5)
                    score = conf * weight
                    code_scores[code] = max(code_scores.get(code, 0), score)
        
        return code_scores
    
    def _get_sorted_candidates(
        self, 
        candidates: Set[str], 
        report: str, 
        cpt: str
    ) -> List[str]:
        """
        Sort candidates by rule + frequency score.
        Matches BestInference logic.
        """
        rule_scores = self._get_rule_scores(report, cpt)
        
        # Frequency scores (normalized)
        freq_scores = {c: self.code_frequency.get(c, 0) for c in candidates}
        max_freq = max(freq_scores.values()) if freq_scores else 1
        
        # Combined score
        combined = {}
        for c in candidates:
            freq = freq_scores.get(c, 0) / max_freq
            rule = rule_scores.get(c, 0)
            combined[c] = freq + rule * 2  # Rules weighted 2x
        
        return sorted(candidates, key=lambda c: -combined.get(c, 0))
    
    def preprocess(self, report_text: str) -> PreprocessedData:
        """
        Main preprocessing function.
        
        Uses RAG to find similar cases and extract candidates.
        
        Args:
            report_text: Raw radiology report text
        
        Returns:
            PreprocessedData with all extracted features and built prompt
        """
        data = PreprocessedData(report_text=report_text)
        text_upper = report_text.upper()
        
        # 1. Try to extract CPT from report (explicit CPT codes only)
        data.extracted_cpt = self._extract_cpt(text_upper)
        
        # 2. Extract features
        data.modality = self._extract_modality(text_upper)
        data.body_part = self._extract_body_part(text_upper)
        data.laterality = self._extract_laterality(text_upper)
        data.has_contrast = self._extract_contrast(text_upper)
        
        # 3. Use RAG to get candidates from similar cases
        # Uses enriched metadata (indication, impression) from AWS data
        rag_cpt_candidates = []
        rag_icd_candidates = []
        rag_examples = ""
        rag_match_score = 0.0
        
        if self.rag_retriever:
            # IMPORTANT: Extract sections first to match indexed format
            # Index uses Parsed_Compact, so search should use similar structured text
            search_query = report_text
            if self.section_extractor:
                try:
                    sections = self.section_extractor.extract(report_text)
                    compact = sections.to_compact_text()
                    if compact and len(compact) > 50:
                        search_query = compact  # Use structured text for better matching
                except Exception:
                    pass  # Fall back to full report
            
            # Get candidates AND formatted few-shot examples in one call
            # The new method uses enriched metadata (indication, impression) for better examples
            # Also returns rag_match_score (0-100) for closest match similarity
            rag_cpt_candidates, rag_icd_candidates, rag_examples, rag_match_score = \
                self.rag_retriever.get_similar_with_examples(search_query, top_k=5)
        
        # 4. Get ICD candidates
        if data.extracted_cpt:
            # Use CPT→ICD lookup for candidates
            data.icd_candidates = self._get_icd_candidates(data.extracted_cpt, report_text)
        elif rag_icd_candidates:
            # Use RAG candidates + expand with related codes
            all_candidates = set(rag_icd_candidates)
            
            # Also add codes from RAG CPT candidates
            for cpt in rag_cpt_candidates[:3]:
                all_candidates.update(self.cpt_to_icd.get(cpt, set()))
            
            # Filter to top 500 if possible
            filtered = all_candidates & self.top_500_codes
            if filtered:
                all_candidates = filtered
            
            # Sort by frequency
            sorted_codes = sorted(all_candidates, key=lambda c: -self.code_frequency.get(c, 0))
            data.icd_candidates = self._format_candidates_sorted(sorted_codes)
        else:
            # Fallback: use modality/body part heuristics
            likely_cpts = self._get_likely_cpts(data)
            all_candidates = set()
            for cpt in likely_cpts:
                all_candidates.update(self.cpt_to_icd.get(cpt, set()))
            sorted_codes = sorted(all_candidates, key=lambda c: -self.code_frequency.get(c, 0))
            data.icd_candidates = self._format_candidates_sorted(sorted_codes)
        
        # 5. Store RAG CPT candidates and match score for LLM prompt
        data.rag_cpt_candidates = rag_cpt_candidates
        data.rag_match_score = rag_match_score
        
        # 6. Get few-shot examples
        if rag_examples:
            # Use RAG examples (more relevant)
            data.examples = self._parse_rag_examples(rag_examples)
        elif data.extracted_cpt and self.df is not None:
            data.examples = self._get_examples(data.extracted_cpt)
        
        # 7. Build prompt
        data.prompt = self._build_prompt(data)
        
        return data
    
    def _parse_rag_examples(self, examples_str: str) -> List[Dict]:
        """Parse RAG examples string into list of dicts.
        
        Handles format from rag_retriever.format_examples():
            Report: <content>
            Output: CPT, MODIFIER, ICD1, ICD2, ...
        """
        examples = []
        for block in examples_str.split('---'):
            block = block.strip()
            if not block:
                continue
            
            # Extract report content (everything between "Report:" and "Output:")
            report_match = re.search(r'Report:\s*(.+?)(?=\nOutput:|\Z)', block, re.DOTALL)
            
            # Extract output line: "Output: CPT, MODIFIER, ICD1, ICD2, ..."
            output_match = re.search(r'Output:\s*(.+?)$', block, re.MULTILINE)
            
            if report_match and output_match:
                report_content = report_match.group(1).strip()[:500]
                output_parts = [p.strip() for p in output_match.group(1).split(',')]
                
                # Parse output parts: first is CPT, then optional modifier, then ICDs
                cpt = output_parts[0] if output_parts else ''
                
                # Check if second part is a modifier or ICD
                modifier = ''
                icd_start = 1
                if len(output_parts) > 1:
                    second = output_parts[1]
                    # Modifiers: 26, TC, LT, RT, 50, or combinations like "26 LT"
                    if second in ['26', 'TC', 'LT', 'RT', '50', 'JW', 'PI'] or \
                       '26' in second or 'LT' in second or 'RT' in second or '50' in second:
                        modifier = second
                        icd_start = 2
                
                icd_codes = ', '.join(output_parts[icd_start:]) if len(output_parts) > icd_start else ''
                
                examples.append({
                    'report': report_content,
                    'cpt': cpt,
                    'modifier': modifier,
                    'icd': icd_codes
                })
        
        return examples
    
    def _extract_cpt(self, text: str) -> Optional[str]:
        """
        Extract CPT code if embedded in report.
        
        Priority:
        1. Explicit CPT code in text
        2. Parse Exam Description to determine CPT
        """
        # Pattern 1: "CPT Code: 77066" or "CPT: 77066"
        match = re.search(r'CPT(?:\s*CODE)?[:\s]+([0-9]{5})', text)
        if match:
            return match.group(1)
        
        # Pattern 2: "Exam: 77066 - Description"
        # BUT skip if it's a multi-procedure study (has multiple body parts)
        match = re.search(r'EXAM[:\s]+([0-9]{5})\s*[-\[]([^\n]+)', text)
        if match:
            cpt = match.group(1)
            desc = match.group(2).upper() if match.group(2) else ''
            # Skip extraction for multi-procedure studies
            multi_proc_indicators = ['AND', 'SPINE', 'ABD PELVIS', 'CHEST ABD', 'C/A/P', 'CAP']
            is_multi = any(ind in desc for ind in multi_proc_indicators)
            if not is_multi:
                return cpt
        
        # Pattern 3: "[77066]" or "(77066)"
        match = re.search(r'[\[\(]([0-9]{5})[\]\)]', text)
        if match:
            return match.group(1)
        
        # Pattern 4: "Exam: CR CHEST 71046" at end of line
        match = re.search(r'EXAM[:\s]+[A-Z\s]+([0-9]{5})\s*$', text, re.MULTILINE)
        if match:
            return match.group(1)
        
        # DON'T infer CPT from exam description - let the LLM predict it
        # The extraction from exam description has only 47% accuracy
        # while the LLM achieves 70%+ accuracy
        
        return None
    
    def _infer_cpt_from_exam_description(self, text: str) -> Optional[str]:
        """
        Infer CPT from Exam Description field.
        
        Common patterns:
        - "CT CHEST WITHOUT IV CONTRAST" → 71250
        - "XR CHEST 1 VIEW" → 71045
        - "MA DIGITAL MAMMO SCREENING" → 77067
        """
        # Extract Exam Description line - try multiple formats
        exam_match = re.search(r'EXAM\s*DESCRIPTION[:\s]*([^\n]+)', text)
        if not exam_match:
            exam_match = re.search(r'EXAMINATION[:\s]+([^\n]+)', text)
        if not exam_match:
            exam_match = re.search(r'\bEXAM[:\s]+([^\n]+)', text)
        
        if not exam_match:
            return None
        
        exam_desc = exam_match.group(1).upper().strip()
        
        # PRIORITY: Check for XR first (more specific matches)
        if 'XR ' in exam_desc or 'X-RAY' in exam_desc:
            return self._infer_xray_cpt(exam_desc)
        
        # Check for US/Ultrasound (before CT which might also match "contrast")
        if 'US ' in exam_desc or 'ULTRASOUND' in exam_desc:
            return self._infer_ultrasound_cpt(exam_desc)
        
        # Check for mammography
        if 'MAMMO' in exam_desc or 'TOMO' in exam_desc:
            return self._infer_mammo_cpt(exam_desc)
        
        # Check for CT
        if 'CT ' in exam_desc or 'CT\n' in exam_desc:
            return self._infer_ct_cpt(exam_desc)
        
        # Check for MR/MRI
        if 'MR ' in exam_desc or 'MRI' in exam_desc:
            return self._infer_mri_cpt(exam_desc)
        
        # Check for DEXA
        if 'DEXA' in exam_desc or 'BONE DENSITY' in exam_desc or 'BMD' in exam_desc:
            return '77080'
        
        return None
    
    def _infer_xray_cpt(self, exam_desc: str) -> Optional[str]:
        """Infer CPT for X-ray exams."""
        if 'CHEST' in exam_desc:
            if '1 VIEW' in exam_desc or 'SINGLE' in exam_desc or 'PORTABLE' in exam_desc:
                return '71045'
            elif '2 VIEW' in exam_desc or 'TWO VIEW' in exam_desc or 'PA AND LAT' in exam_desc:
                return '71046'
            return '71046'  # Default to 2 views
        elif 'ABDOMEN' in exam_desc:
            if '1 VIEW' in exam_desc or 'SINGLE' in exam_desc:
                return '74018'
            elif '2 VIEW' in exam_desc:
                return '74019'
            return '74018'
        elif 'SHOULDER' in exam_desc:
            return '73030'
        elif 'KNEE' in exam_desc:
            if '3' in exam_desc:
                return '73562'
            return '73560'
        elif 'ANKLE' in exam_desc:
            return '73610'
        elif 'FOOT' in exam_desc or 'FEET' in exam_desc:
            return '73630'
        elif 'HAND' in exam_desc:
            return '73130'
        elif 'WRIST' in exam_desc:
            return '73110'
        elif 'ELBOW' in exam_desc:
            return '73080'
        elif 'HIP' in exam_desc:
            return '73502'
        elif 'TIBIA' in exam_desc or 'FIBULA' in exam_desc:
            return '73590'
        elif 'FINGER' in exam_desc:
            return '73140'
        elif 'NECK' in exam_desc and 'SOFT' in exam_desc:
            return '70360'
        return None
    
    def _infer_ultrasound_cpt(self, exam_desc: str) -> Optional[str]:
        """Infer CPT for ultrasound exams."""
        if 'BREAST' in exam_desc:
            if 'LIMITED' in exam_desc:
                return '76642'
            return '76641'
        elif 'PELVI' in exam_desc:
            if 'TRANSVAG' in exam_desc or 'ENDOVAG' in exam_desc:
                return '76830'
            elif 'LIMITED' in exam_desc:
                return '76857'
            return '76856'
        elif 'ABDOMEN' in exam_desc:
            if 'LIMITED' in exam_desc:
                return '76705'
            return '76700'
        elif 'THYROID' in exam_desc or ('NECK' in exam_desc and 'SOFT' in exam_desc):
            return '76536'
        elif 'VEIN' in exam_desc or 'VENOUS' in exam_desc or 'LOWER EXTREM' in exam_desc:
            return '93971'
        elif 'RETROPERITON' in exam_desc:
            return '76770'
        return None
    
    def _infer_mammo_cpt(self, exam_desc: str) -> Optional[str]:
        """Infer CPT for mammography exams."""
        # Check for tomosynthesis FIRST (most specific)
        if 'TOMO' in exam_desc or '3D' in exam_desc or 'DBT' in exam_desc:
            return '77063'  # Tomosynthesis
        elif 'SCREEN' in exam_desc:
            return '77067'  # Screening
        elif 'DIAGNOSTIC' in exam_desc:
            if 'BILATERAL' in exam_desc or 'BILAT' in exam_desc:
                return '77066'
            return '77065'
        # Default for screening-like descriptions
        if 'BILATERAL' in exam_desc or 'BILAT' in exam_desc:
            return '77067'
        return '77067'
    
    def _infer_ct_cpt(self, exam_desc: str) -> Optional[str]:
        """Infer CPT for CT exams."""
        if 'CHEST' in exam_desc:
            if 'ANGIO' in exam_desc or 'CTA' in exam_desc:
                return '71275'
            elif 'WITH' in exam_desc and 'WITHOUT' in exam_desc:
                return '71270'
            elif 'WITHOUT' in exam_desc:
                return '71250'
            elif 'WITH' in exam_desc:
                return '71260'
            return '71250'
        elif 'HEAD' in exam_desc or 'BRAIN' in exam_desc:
            if 'ANGIO' in exam_desc or 'CTA' in exam_desc:
                return '70496'
            elif 'WITH' in exam_desc and 'WITHOUT' in exam_desc:
                return '70470'
            elif 'WITHOUT' in exam_desc:
                return '70450'
            elif 'WITH' in exam_desc:
                return '70460'
            return '70450'
        elif 'ABDOMEN' in exam_desc and 'PELVIS' in exam_desc:
            if 'WITH' in exam_desc and 'WITHOUT' in exam_desc:
                return '74178'
            elif 'WITHOUT' in exam_desc:
                return '74176'
            elif 'WITH' in exam_desc:
                return '74177'
            return '74177'
        elif 'ABDOMEN' in exam_desc:
            if 'WITHOUT' in exam_desc:
                return '74150'
            return '74160'
        elif 'NECK' in exam_desc:
            if 'ANGIO' in exam_desc or 'CTA' in exam_desc:
                return '70498'
        elif 'CERVICAL' in exam_desc or 'C-SPINE' in exam_desc or 'C SPINE' in exam_desc:
            return '72125'
        elif 'THORACIC' in exam_desc or 'T-SPINE' in exam_desc or 'T SPINE' in exam_desc:
            return '72128'
        elif 'LUMBAR' in exam_desc or 'L-SPINE' in exam_desc or 'L SPINE' in exam_desc:
            return '72131'
        return None
    
    def _infer_mri_cpt(self, exam_desc: str) -> Optional[str]:
        """Infer CPT for MRI exams."""
        if 'BRAIN' in exam_desc or 'HEAD' in exam_desc:
            if 'WITH' in exam_desc and 'WITHOUT' in exam_desc:
                return '70553'
            return '70551'
        elif 'CERVICAL' in exam_desc:
            return '72141'
        elif 'LUMBAR' in exam_desc:
            return '72148'
        elif 'ABDOMEN' in exam_desc:
            return '74183'
        elif 'BREAST' in exam_desc:
            return '77049'
        elif 'KNEE' in exam_desc or 'LOWER EXTREM' in exam_desc:
            return '73721'
        elif 'SHOULDER' in exam_desc or 'UPPER EXTREM' in exam_desc:
            return '73221'
        return None
        
        # === CHEST ===
        if 'CHEST' in exam_desc:
            if 'CT' in exam_desc or 'COMPUTED' in exam_desc:
                if 'WITH' in exam_desc and 'WITHOUT' in exam_desc:
                    return '71270'  # CT Chest without and with
                elif 'WITHOUT' in exam_desc:
                    return '71250'  # CT Chest without
                elif 'WITH' in exam_desc:
                    return '71260'  # CT Chest with
                elif 'ANGIO' in exam_desc or 'CTA' in exam_desc:
                    return '71275'  # CTA Chest
                return '71250'  # Default to without contrast
            elif 'XR' in exam_desc or 'X-RAY' in exam_desc or 'RADIOGRAPH' in exam_desc:
                if '1 VIEW' in exam_desc or 'SINGLE' in exam_desc or 'PORTABLE' in exam_desc:
                    return '71045'
                elif '2 VIEW' in exam_desc or 'TWO VIEW' in exam_desc:
                    return '71046'
                return '71046'  # Default to 2 views
        
        # === HEAD/BRAIN ===
        if 'HEAD' in exam_desc or 'BRAIN' in exam_desc:
            if 'CT' in exam_desc:
                if 'ANGIO' in exam_desc or 'CTA' in exam_desc:
                    return '70496'  # CTA Head
                elif 'WITH' in exam_desc and 'WITHOUT' in exam_desc:
                    return '70470'
                elif 'WITHOUT' in exam_desc:
                    return '70450'
                elif 'WITH' in exam_desc:
                    return '70460'
                return '70450'  # Default to without
            elif 'MR' in exam_desc:
                if 'WITH' in exam_desc and 'WITHOUT' in exam_desc:
                    return '70553'
                return '70551'  # Default to without
        
        # === ABDOMEN/PELVIS ===
        if 'ABDOMEN' in exam_desc or 'PELVIS' in exam_desc:
            if 'CT' in exam_desc:
                both = 'ABDOMEN' in exam_desc and 'PELVIS' in exam_desc
                if both:
                    if 'WITH' in exam_desc and 'WITHOUT' in exam_desc:
                        return '74178'
                    elif 'WITHOUT' in exam_desc:
                        return '74176'
                    elif 'WITH' in exam_desc:
                        return '74177'
                    return '74177'  # Default to with contrast
                elif 'ABDOMEN' in exam_desc:
                    if 'WITHOUT' in exam_desc:
                        return '74150'
                    return '74160'
            elif 'XR' in exam_desc:
                if '1 VIEW' in exam_desc:
                    return '74018'
                elif '2 VIEW' in exam_desc:
                    return '74019'
                return '74018'
            elif 'US' in exam_desc or 'ULTRASOUND' in exam_desc:
                if 'PELVI' in exam_desc:
                    if 'TRANSVAG' in exam_desc or 'TV' in exam_desc:
                        return '76830'
                    elif 'LIMITED' in exam_desc:
                        return '76857'
                    return '76856'  # Complete
                return '76700'  # Abdomen complete
        
        # === MAMMOGRAPHY ===
        if 'MAMMO' in exam_desc:
            if 'TOMO' in exam_desc or 'DBT' in exam_desc or '3D' in exam_desc:
                return '77063'  # Tomosynthesis
            elif 'SCREEN' in exam_desc:
                return '77067'  # Screening
            elif 'DIAGNOSTIC' in exam_desc:
                if 'BILATERAL' in exam_desc or 'BILAT' in exam_desc:
                    return '77066'
                return '77065'  # Unilateral
            return '77067'  # Default to screening
        
        # === BREAST ULTRASOUND ===
        if 'BREAST' in exam_desc and ('US' in exam_desc or 'ULTRASOUND' in exam_desc):
            if 'LIMITED' in exam_desc:
                return '76642'
            return '76641'
        
        # === SPINE ===
        if 'SPINE' in exam_desc or 'CERVICAL' in exam_desc or 'LUMBAR' in exam_desc or 'THORACIC' in exam_desc:
            if 'CT' in exam_desc:
                if 'CERVICAL' in exam_desc or 'C-SPINE' in exam_desc:
                    return '72125'
                elif 'THORACIC' in exam_desc or 'T-SPINE' in exam_desc:
                    return '72128'
                elif 'LUMBAR' in exam_desc or 'L-SPINE' in exam_desc:
                    return '72131'
            elif 'MR' in exam_desc:
                if 'CERVICAL' in exam_desc:
                    return '72141'
                elif 'LUMBAR' in exam_desc:
                    return '72148'
        
        # === EXTREMITIES ===
        if 'XR' in exam_desc or 'X-RAY' in exam_desc or 'RADIOGRAPH' in exam_desc:
            if 'SHOULDER' in exam_desc:
                return '73030'
            elif 'KNEE' in exam_desc:
                if '3' in exam_desc or 'THREE' in exam_desc:
                    return '73562'
                return '73560'
            elif 'ANKLE' in exam_desc:
                return '73610'
            elif 'FOOT' in exam_desc or 'FEET' in exam_desc:
                return '73630'
            elif 'HAND' in exam_desc:
                return '73130'
            elif 'WRIST' in exam_desc:
                return '73110'
            elif 'ELBOW' in exam_desc:
                return '73080'
            elif 'HIP' in exam_desc:
                return '73502'
            elif 'TIBIA' in exam_desc or 'FIBULA' in exam_desc:
                return '73590'
        
        # === VASCULAR ===
        if 'DUPLEX' in exam_desc or 'VENOUS' in exam_desc:
            if 'LOWER' in exam_desc or 'LEG' in exam_desc:
                return '93971'
        
        # === DEXA ===
        if 'DEXA' in exam_desc or 'BONE DENSITY' in exam_desc or 'BMD' in exam_desc:
            return '77080'
        
        # === THYROID/SOFT TISSUE ===
        if 'THYROID' in exam_desc or ('NECK' in exam_desc and 'SOFT TISSUE' in exam_desc):
            if 'US' in exam_desc or 'ULTRASOUND' in exam_desc:
                return '76536'
        
        # === NECK SOFT TISSUE X-RAY ===
        if 'NECK' in exam_desc and 'SOFT TISSUE' in exam_desc and ('XR' in exam_desc or 'X-RAY' in exam_desc):
            return '70360'
        
        return None
    
    def _extract_modality(self, text: str) -> str:
        """Extract imaging modality."""
        if 'CT ' in text or 'COMPUTED TOMOGRAPHY' in text or ' CT' in text:
            if 'ANGIO' in text or 'CTA ' in text or ' CTA' in text:
                return 'cta'
            return 'ct'
        elif 'MRI' in text or 'MR ' in text or 'MAGNETIC RESONANCE' in text:
            return 'mri'
        elif 'X-RAY' in text or 'RADIOGRAPH' in text or 'XR ' in text or ' XR' in text:
            return 'xray'
        elif 'ULTRASOUND' in text or ' US ' in text or 'SONOGRA' in text:
            if 'TRANSVAGINAL' in text or 'ENDOVAGINAL' in text:
                return 'us_transvaginal'
            return 'ultrasound'
        elif 'MAMMO' in text or 'TOMOSYNTHESIS' in text:
            if 'TOMOSYNTHESIS' in text or 'DBT' in text or '3D' in text:
                return 'tomosynthesis'
            return 'mammography'
        elif 'PET' in text and ('FDG' in text or 'POSITRON' in text):
            return 'pet'
        elif 'NUCLEAR' in text or ' NM ' in text or 'RENAL SCAN' in text:
            return 'nuclear'
        elif 'FLUORO' in text:
            return 'fluoroscopy'
        return 'unknown'
    
    def _extract_body_part(self, text: str) -> str:
        """Extract body part."""
        # Check exam description line first
        exam_match = re.search(r'EXAM(?:INATION)?[:\s]+([^\n]+)', text)
        exam_line = exam_match.group(1) if exam_match else text[:500]
        
        if 'CHEST' in exam_line or 'LUNG' in exam_line or 'THORAX' in exam_line:
            return 'chest'
        elif 'HEAD' in exam_line or 'BRAIN' in exam_line:
            return 'head'
        elif 'NECK' in exam_line and 'SOFT TISSUE' in exam_line:
            return 'neck_soft_tissue'
        elif 'NECK' in exam_line:
            return 'neck'
        elif 'ABDOMEN' in exam_line and 'PELVIS' in exam_line:
            return 'abdomen_pelvis'
        elif 'ABDOMEN' in exam_line:
            return 'abdomen'
        elif 'PELVIS' in exam_line:
            return 'pelvis'
        elif 'KNEE' in exam_line:
            return 'knee'
        elif 'ANKLE' in exam_line:
            return 'ankle'
        elif 'FOOT' in exam_line or 'FEET' in exam_line:
            return 'foot'
        elif 'SHOULDER' in exam_line:
            return 'shoulder'
        elif 'WRIST' in exam_line:
            return 'wrist'
        elif 'HAND' in exam_line or 'FINGER' in exam_line:
            return 'hand'
        elif 'HIP' in exam_line:
            return 'hip'
        elif 'ELBOW' in exam_line:
            return 'elbow'
        elif 'SPINE' in exam_line or 'CERVICAL' in exam_line or 'LUMBAR' in exam_line:
            return 'spine'
        elif 'BREAST' in exam_line:
            return 'breast'
        elif 'RENAL' in exam_line or 'KIDNEY' in exam_line:
            return 'renal'
        elif 'EXTREMITY' in exam_line or 'VEIN' in exam_line:
            return 'extremity'
        
        return 'unknown'
    
    def _extract_laterality(self, text: str) -> str:
        """Extract laterality from report."""
        # Check multiple sections in priority order
        sections = []
        
        # Exam description
        exam_match = re.search(r'EXAM(?:INATION)?[:\s]+([^\n]+)', text)
        if exam_match:
            sections.append(exam_match.group(1))
        
        # Technique
        tech_match = re.search(r'TECHNIQUE[:\s]+([^\n]+)', text)
        if tech_match:
            sections.append(tech_match.group(1))
        
        # Findings (first 500 chars)
        findings_match = re.search(r'FINDINGS[:\s]*(.{0,500})', text, re.DOTALL)
        if findings_match:
            sections.append(findings_match.group(1))
        
        # Impression
        impression_match = re.search(r'IMPRESSION[:\s]*(.{0,300})', text, re.DOTALL)
        if impression_match:
            sections.append(impression_match.group(1))
        
        # Fallback to first 1000 chars
        sections.append(text[:1000])
        
        for section in sections:
            has_bilateral = bool(re.search(r'\bBILATERAL\b|\bBOTH\s+(SIDES|KNEES|HIPS|LEGS|EXTREMIT|BREASTS)\b', section))
            has_left = bool(re.search(r'\bLEFT\b', section))
            has_right = bool(re.search(r'\bRIGHT\b', section))
            
            if has_bilateral or (has_left and has_right):
                return 'bilateral'
            elif has_left and not has_right:
                return 'left'
            elif has_right and not has_left:
                return 'right'
        
        return 'none'
    
    def _extract_contrast(self, text: str) -> bool:
        """Check if contrast was used."""
        has_with = 'WITH CONTRAST' in text or 'W/ CONTRAST' in text or 'W/CONTRAST' in text
        has_without = 'WITHOUT CONTRAST' in text or 'W/O CONTRAST' in text
        return has_with and not has_without
    
    def _get_likely_cpts(self, data: PreprocessedData) -> List[str]:
        """Get likely CPT codes based on modality and body part."""
        candidates = []
        
        modality = data.modality
        body_part = data.body_part
        
        if modality == 'ct':
            if body_part == 'head':
                candidates = ['70450', '70460', '70470']
            elif body_part == 'chest':
                candidates = ['71250', '71260', '71270']
            elif body_part in ('abdomen', 'abdomen_pelvis'):
                candidates = ['74176', '74177', '74178']
        elif modality == 'cta':
            if body_part == 'head':
                candidates = ['70496']
            elif body_part == 'neck':
                candidates = ['70498']
            elif body_part == 'chest':
                candidates = ['71275']
        elif modality == 'xray':
            if body_part == 'chest':
                candidates = ['71045', '71046']
            elif body_part == 'knee':
                candidates = ['73560', '73562', '73564']
            elif body_part == 'ankle':
                candidates = ['73600', '73610']
            elif body_part == 'foot':
                candidates = ['73620', '73630']
            elif body_part == 'shoulder':
                candidates = ['73030']
            elif body_part == 'hip':
                candidates = ['73502']
        elif modality in ('mammography', 'tomosynthesis'):
            candidates = ['77067', '77063', '77065', '77066']
        elif modality == 'ultrasound':
            if body_part == 'pelvis':
                candidates = ['76856', '76857']
            elif body_part == 'breast':
                candidates = ['76641', '76642']
            elif body_part == 'extremity':
                candidates = ['93971', '76881', '76882']
        
        return candidates
    
    def _get_icd_candidates(self, cpt: str, report: str = "") -> List[Tuple[str, str]]:
        """Get ICD candidates for a CPT code, sorted by relevance."""
        candidates = self.cpt_to_icd.get(cpt, set())
        
        # Filter to top 500 if possible
        filtered = candidates & self.top_500_codes
        if filtered:
            candidates = filtered
        
        # Sort by rules + frequency (matches BestInference)
        if report:
            sorted_codes = self._get_sorted_candidates(candidates, report, cpt)
        else:
            # Fallback to frequency-only sorting
            sorted_codes = sorted(candidates, key=lambda c: -self.code_frequency.get(c, 0))
        
        return self._format_candidates_sorted(sorted_codes)
    
    def _format_candidates(self, codes: Set[str]) -> List[Tuple[str, str]]:
        """Format candidates as (code, description) tuples, sorted by frequency."""
        formatted = []
        for code in codes:
            desc = self.icd_descriptions.get(code, code)
            freq = self.code_frequency.get(code, 0)
            formatted.append((code, desc, freq))
        
        # Sort by frequency descending
        formatted.sort(key=lambda x: -x[2])
        
        # Return top 50, without frequency
        return [(code, desc) for code, desc, _ in formatted[:50]]
    
    def _format_candidates_sorted(self, sorted_codes: List[str]) -> List[Tuple[str, str]]:
        """Format pre-sorted candidates as (code, description) tuples."""
        formatted = []
        for code in sorted_codes[:50]:  # Top 50
            desc = self.icd_descriptions.get(code, code)
            formatted.append((code, str(desc)[:65]))  # Limit description length
        return formatted
    
    def _get_examples(self, cpt: str, n: int = 3) -> List[Dict]:
        """Get few-shot examples from training data for same CPT.
        
        Uses smart section extraction (INDICATION + LATERALITY + IMPRESSION)
        instead of simple truncation for better context.
        """
        if self.df is None:
            return []
        
        same_cpt = self.df[self.df['Procedure'] == cpt]
        examples = []
        
        for _, row in same_cpt.head(n * 2).iterrows():
            report = str(row.get('Report', ''))
            icd = str(row.get('ICD10 - Diagnosis', ''))
            modifier = str(row.get('Modifier', ''))
            
            if report and icd and 'nan' not in icd.lower():
                # Use smart extraction if available, otherwise use Parsed_Compact from data
                if 'Parsed_Compact' in row and row['Parsed_Compact'] and str(row['Parsed_Compact']) != 'nan':
                    report_compact = str(row['Parsed_Compact'])[:600]
                elif self.section_extractor:
                    # Extract on the fly
                    sections = self.section_extractor.extract(report)
                    report_compact = sections.to_compact_text()[:600]
                else:
                    # Fallback to truncation
                    report_compact = report[:400]
                
                examples.append({
                    'report': report_compact,
                    'cpt': cpt,
                    'icd': icd,
                    'modifier': modifier if modifier != 'nan' else ''
                })
            
            if len(examples) >= n:
                break
        
        return examples
    
    def _build_prompt(self, data: PreprocessedData) -> str:
        """
        Build the prompt for LLM.
        
        Uses the EXACT format from BestInference that achieved 34-36% exact match.
        """
        # Format examples with smart extraction (INDICATION + LATERALITY + IMPRESSION)
        examples_str = ""
        if data.examples:
            examples_parts = []
            for ex in data.examples:
                # Include modifier in output if present
                if ex.get('modifier'):
                    output = f"{ex['cpt']}, {ex['modifier']}, {ex['icd']}"
                else:
                    output = f"{ex['cpt']}, {ex['icd']}"
                examples_parts.append(f"Report: {ex['report']}\nOutput: {output}")
            examples_str = '\n---\n'.join(examples_parts)
        
        # Format candidates - JUST CODES, no descriptions (matches training data)
        codes_str = '\n'.join([
            code 
            for code, desc in data.icd_candidates[:50]
        ])
        
        # Build prompt - EXACT format from BestInference
        if data.extracted_cpt:
            # CPT known - just predict ICD (matches BestInference exactly)
            prompt = f'''Study these similar cases:

{examples_str}

---

Code this report:
{data.report_text[:3200]}

CPT CODE (from exam): {data.extracted_cpt}

Available ICD codes (from similar cases):
{codes_str}

Rules:
- Determine the CPT code from the procedure/exam description
- Add appropriate modifiers (26=professional, LT=left, RT=right, 50=bilateral)
- Code the clinical indication (R/Z/S code)
- Code confirmed findings from impression
- Use laterality info to select correct modifier (LT/RT/50)

Output format: CPT, MODIFIER, ICD1, ICD2, ...
Example: 71046, 26, R91.8, J18.9
Example: 73030, 26 LT, M25.511, S43.401A'''
        
        else:
            # CPT not known - predict both
            # Use RAG CPT candidates if available, otherwise use full list
            if data.rag_cpt_candidates:
                # RAG found similar cases - use their CPT codes as top candidates
                # JUST CODES - no descriptions (matches training data format)
                rag_cpts = list(dict.fromkeys(data.rag_cpt_candidates))[:10]  # Unique, top 10
                rag_cpt_str = '\n'.join(rag_cpts)
                
                prompt = f'''Study these similar cases:

{examples_str}

---

Code this report:
{data.report_text[:2800]}

SIMILAR CASES CPT CODES (most likely):
{rag_cpt_str}

Available ICD codes (from similar cases):
{codes_str}

Rules:
- Determine the CPT code from the procedure/exam description
- Add appropriate modifiers (26=professional, LT=left, RT=right, 50=bilateral)
- Code the clinical indication (R/Z/S code)
- Code confirmed findings from impression
- Use laterality info to select correct modifier (LT/RT/50)

Output format: CPT, MODIFIER, ICD1, ICD2, ...
Example: 71046, 26, R91.8, J18.9
Example: 73030, 26 LT, M25.511, S43.401A'''
            
            else:
                # Fallback: use full CPT list
                cpt_options_str = '\n'.join([f'{code}: {desc}' for code, desc in CPT_CODES.items()])
                
                prompt = f'''You are an expert radiology coder.

Code this report:
{data.report_text[:2800]}

DETECTED EXAM INFO:
- Modality: {data.modality}
- Body Part: {data.body_part}
- Contrast: {'Yes' if data.has_contrast else 'No'}

CPT CODE OPTIONS:
{cpt_options_str}

Available ICD codes (sorted by relevance):
{codes_str}

CPT SELECTION RULES:
1. Match the imaging modality (X-ray, CT, MRI, Ultrasound, Mammography)
2. Match the body part (head, chest, abdomen, pelvis, extremity)
3. Check for contrast (with, without, or both)
4. For mammography: screening vs diagnostic, tomosynthesis vs regular
5. For X-rays: check view count (1 view, 2 views, 3+ views)

ICD SELECTION RULES:
- Code the clinical indication (R/Z/S code)
- Code confirmed findings from impression
- For normal studies, code only the indication

Output format: CPT, ICD1, ICD2, ...
Example: 71046, R91.8, J18.9'''
        
        return prompt

