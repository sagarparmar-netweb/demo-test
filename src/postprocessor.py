#!/usr/bin/env python3
"""
STEP 3: POST-PROCESSING

Adds modifiers, validates codes, and formats final output.
No LLM calls in this step - all rules-based.

Handles:
- Modifier prediction (26, LT, RT, 50, TC)
- Medical necessity validation (R/Z/S code)
- Laterality consolidation
- Conflict resolution
- Output formatting
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple


@dataclass
class FinalResult:
    """Final output from the pipeline."""
    
    # Core outputs (matching data format)
    procedure: str          # CPT code
    modifier: str           # Space-separated modifiers
    icd10_diagnosis: str    # Comma-separated ICD codes
    
    # Metadata
    cpt_was_extracted: bool = False
    confidence: float = 0.0
    
    # Medical necessity validation
    has_medical_necessity: bool = True
    medical_necessity_warning: str = ""
    
    def to_dict(self) -> dict:
        """Return as dictionary matching data format."""
        return {
            'Procedure': self.procedure,
            'Modifier': self.modifier,
            'ICD10 - Diagnosis': self.icd10_diagnosis
        }
    
    def needs_review(self) -> bool:
        """Check if this result needs human review."""
        return not self.has_medical_necessity or bool(self.medical_necessity_warning)


class Postprocessor:
    """
    Post-processes LLM predictions to produce final output.
    """
    
    # CPT codes that use laterality modifiers (LT/RT)
    # Only codes with >10% laterality usage in training data
    # Excludes chest/abdomen/spine CTs where "left/right" refers to findings, not procedure
    LATERALITY_CPT_CODES = {
        # Ribs (special case - bilateral imaging)
        '71100', '71101',
        # Extremity X-rays - UPPER
        '73000',  # Clavicle
        '73020',  # Shoulder (1 view)
        '73030',  # Shoulder (2+ views)
        '73060',  # Humerus
        '73070', '73080',  # Elbow
        '73090', '73092',  # Forearm
        '73100', '73110',  # Wrist
        '73120', '73130', '73140',  # Hand/Fingers
        '73200', '73201', '73202', '73206',  # CT upper extremity
        '73221', '73222', '73223',  # MRI upper extremity
        # Extremity X-rays - LOWER
        '73501', '73502', '73503',  # Hip
        '73551', '73552',  # Femur
        '73560', '73562', '73564',  # Knee
        '73590', '73592',  # Tibia/Fibula
        '73600', '73610', '73620', '73630',  # Ankle/Foot
        '73650', '73660',  # Calcaneus/Toe
        '73700', '73701', '73702', '73706',  # CT lower extremity
        '73718', '73720', '73721', '73723',  # MRI lower extremity
        # Breast imaging
        '76642',  # Breast ultrasound
        '77061', '77065',  # Mammography
        # Ultrasound
        '76882',  # Extremity ultrasound
        # Vascular
        '93926', '93931', '93971',  # Venous/arterial studies
    }
    
    # CPT codes that use 50 modifier when bilateral
    # 76641 uses 50 in 73% of cases (mammography screening)
    BILATERAL_50_CPT_CODES = {'76641', '76642', '76881', '76882'}
    
    # CPT codes where bilateral is the DEFAULT (e.g., screening mammography)
    # Only use LT/RT if explicitly stated
    BILATERAL_DEFAULT_CPTS = {'76641'}
    
    # CPT codes where bilateral = bill twice (no 50 modifier)
    # These get LT and RT billed separately
    BILATERAL_BILL_TWICE_CPT = {
        '73030', '73630', '73560', '73130', '73502', '73610',
        '73562', '73110', '73590', '73700', '73564', '73090',
    }
    
    # =========================================================================
    # INTERVENTIONAL PROCEDURES - DO NOT AUTO-ADD MODIFIER 26
    # =========================================================================
    # These are procedures, not just interpretations. Modifier 26 (professional
    # component) typically does not apply. Data shows 0% usage of 26 for these.
    # 
    INTERVENTIONAL_CPT_CODES = {
        # Paracentesis / Thoracentesis
        '49083',  # Paracentesis (0% have 26 in training data)
        '49082',  # Paracentesis without imaging
        '32555',  # Thoracentesis with imaging (0% have 26)
        '32554',  # Thoracentesis without imaging
        
        # Biopsies
        '10005', '10006', '10007', '10008', '10009',  # FNA biopsies
        '10010', '10011', '10012',  # FNA biopsies
        '10021',  # FNA without imaging
        '19083',  # Breast biopsy with stereotactic guidance
        '19084',  # Breast biopsy with MR guidance
        
        # Drainage procedures
        '10030',  # Image-guided fluid collection drainage
        '49405', '49406', '49407',  # Abscess drainage
        
        # Vascular access
        '36556', '36557', '36558',  # Central venous catheter
        '36569', '36571',  # PICC line placement
        '77001',  # Fluoroscopic guidance for vascular access
        
        # Joint procedures
        '20610', '20611',  # Arthrocentesis (joint aspiration)
        '20600', '20604', '20605', '20606',  # Joint injection
        
        # Lumbar puncture
        '62270',  # Spinal puncture, lumbar, diagnostic
        
        # Tube placements
        '43752', '43753',  # Gastrostomy tube placement
        '74340',  # Contrast injection for tube check
        
        # Nerve blocks (pain management)
        '64490', '64491', '64492',  # Facet joint injections
        '64493', '64494', '64495',  # Facet joint injections
        '62322', '62323',  # Epidural injections
    }
    
    # =========================================================================
    # SPECIALTY MODIFIERS
    # =========================================================================
    
    # PI = Positron Emission Tomography (PET) to inform initial treatment strategy
    # Used for PET and PET/CT scans when determining treatment plan
    PI_CPT_CODES = {
        '78811', '78812', '78813',  # PET imaging
        '78814', '78815', '78816',  # PET/CT imaging
        'G0235', 'G0252',           # PET for specific conditions
    }
    
    # JZ = Zero Drug Amount Discarded/Not Administered
    # Used when contrast/drug vial is fully used with no waste
    # Requires external flag - cannot determine from report alone
    JZ_CPT_CODES = set()  # Needs billing context
    
    # 76 = Repeat Procedure by Same Physician on Same Day
    # 77 = Repeat Procedure by Another Physician on Same Day
    # These require comparing to prior procedures - cannot determine from single report
    REPEAT_PROCEDURE_MODIFIERS = {'76', '77'}
    
    # XU = Unusual Non-Overlapping Service
    # Payer-specific, used when same-day services don't overlap
    # Requires specific payer rules
    
    # =========================================================================
    # SMART MEDICAL NECESSITY VALIDATION
    # =========================================================================
    # 
    # Medical necessity = justification for why imaging was ordered
    # NOT just R/Z/S codes - diagnosis codes can also justify imaging
    #
    # Categories of codes that justify imaging:
    # - R codes: Symptoms/Signs (pain, abnormal findings)
    # - Z codes: Encounters/Screening (routine exam, surveillance)
    # - S/T codes: Trauma/Injury
    # - Diagnosis codes: The condition itself justifies imaging
    #
    # =========================================================================
    
    # ICD-10 prefixes that inherently justify imaging (no R/Z/S needed)
    # These are DIAGNOSIS codes that explain why imaging was ordered
    DIAGNOSIS_JUSTIFIES_IMAGING = {
        # Cancer/Neoplasms - justify surveillance/staging
        'C': 'malignant neoplasm',
        'D0': 'in situ neoplasm',
        'D1': 'benign neoplasm',
        'D2': 'benign neoplasm',
        'D3': 'benign neoplasm',
        'D4': 'neoplasm uncertain behavior',
        
        # Infections - justify diagnostic imaging
        'A': 'infectious disease',
        'B': 'infectious disease',
        'J1': 'influenza/pneumonia',  # J10-J18
        'J2': 'lower respiratory infection',  # J20-J22
        'J4': 'chronic lower respiratory',  # J40-J47 (COPD, asthma)
        'J8': 'other respiratory disease',  # J80-J84 (pulmonary fibrosis)
        'J9': 'other respiratory disease',  # J90-J99 (pleural effusion)
        'K8': 'gallbladder/biliary/pancreas',  # K80-K87
        'N1': 'kidney disease',  # N10-N19
        'N2': 'kidney stones/urinary',  # N20-N29
        
        # Cardiovascular - justify cardiac/vascular imaging
        'I2': 'ischemic heart disease',  # I20-I25
        'I6': 'cerebrovascular disease',  # I60-I69 (stroke)
        'I7': 'arterial disease',  # I70-I79 (aneurysm, PAD)
        'I8': 'venous disease',  # I80-I89 (DVT, varicose)
        
        # Musculoskeletal - justify skeletal imaging
        'M1': 'arthritis',  # M10-M19
        'M4': 'spine disorders',  # M40-M54
        'M5': 'spine disorders',  # M50-M54
        'M7': 'soft tissue disorders',  # M70-M79
        'M8': 'bone disorders',  # M80-M89 (osteoporosis)
        
        # Trauma - justify imaging
        'S': 'injury',
        'T': 'injury/complications',
        
        # Congenital - justify imaging
        'Q': 'congenital malformation',
    }
    
    # CPT codes that REQUIRE specific indication types
    # If these CPTs are coded, we expect certain indication patterns
    CPT_EXPECTED_INDICATIONS = {
        # Screening mammography - typically Z12.31
        '77067': {'expected': ['Z12.31', 'Z12.39'], 'category': 'screening'},
        '77063': {'expected': ['Z12.31', 'Z12.39'], 'category': 'screening'},
        
        # Diagnostic mammography - needs breast symptom or history
        '77065': {'expected_prefix': ['N', 'R92', 'Z85.3', 'Z80.3'], 'category': 'diagnostic'},
        '77066': {'expected_prefix': ['N', 'R92', 'Z85.3', 'Z80.3'], 'category': 'diagnostic'},
    }
    
    # DO NOT auto-add these - flag for review instead
    # Adding unsupported codes = compliance risk
    NEVER_AUTO_ADD = True
    
    def __init__(self, bill_type: str = 'P'):
        """
        Args:
            bill_type: 'P' for professional, 'T' for technical, 'G' for global
        """
        self.bill_type = bill_type
    
    def postprocess(
        self,
        cpt_code: str,
        icd_codes: List[str],
        laterality: str,
        cpt_was_extracted: bool = False,
        valid_icd_codes: Set[str] = None,
        llm_suggested_modifiers: List[str] = None
    ) -> FinalResult:
        """
        Main post-processing function.
        
        Args:
            cpt_code: Predicted CPT code
            icd_codes: Predicted ICD codes
            laterality: Extracted laterality ('left', 'right', 'bilateral', 'none')
            cpt_was_extracted: Whether CPT was extracted from report
            valid_icd_codes: Set of valid ICD codes to filter against
            llm_suggested_modifiers: Modifiers suggested by LLM (will be validated)
        
        Returns:
            FinalResult with formatted output
        """
        # 1. Predict modifiers using rules
        rules_modifiers = self._predict_modifiers(cpt_code, laterality)
        
        # 2. Merge with LLM suggestions (if valid)
        modifiers = self._merge_modifiers(rules_modifiers, llm_suggested_modifiers or [], cpt_code)
        
        # 3. Validate and filter ICD codes
        if valid_icd_codes:
            icd_codes = [c for c in icd_codes if c in valid_icd_codes]
        
        # 4. Filter External Cause codes (W, X, Y) - per RP policy
        icd_codes = self._filter_external_cause_codes(icd_codes)
        
        # 5. Filter incidental findings based on CPT context
        icd_codes = self._filter_incidental_findings(icd_codes, cpt_code)
        
        # 6. Apply diagnosis hierarchy (definitive diagnosis overrules symptom)
        icd_codes = self._apply_diagnosis_hierarchy(icd_codes)
        
        # 7. Consolidate laterality
        icd_codes = self._consolidate_laterality(icd_codes)
        
        # 8. Remove conflicts
        icd_codes = self._remove_conflicts(icd_codes)
        
        # 9. Prioritize anatomic codes (M/N) over symptoms (R)
        icd_codes = self._prioritize_anatomic_codes(icd_codes)
        
        # 10. Remove duplicates, preserve order
        icd_codes = list(dict.fromkeys(icd_codes))
        
        # 11. Validate medical necessity (NO auto-adding codes!)
        icd_codes, has_necessity, warning = self._ensure_medical_necessity(
            icd_codes, valid_icd_codes, cpt_code
        )
        
        # 12. Calculate confidence
        confidence = self._calculate_confidence(icd_codes, cpt_was_extracted, has_necessity)
        
        return FinalResult(
            procedure=cpt_code,
            modifier=' '.join(modifiers),
            icd10_diagnosis=', '.join(icd_codes),
            cpt_was_extracted=cpt_was_extracted,
            confidence=confidence,
            has_medical_necessity=has_necessity,
            medical_necessity_warning=warning
        )
    
    def predict_modifiers(self, cpt_code: str, laterality: str, auto_add_laterality: bool = False) -> List[str]:
        """
        Predict modifiers based on CPT code and laterality.
        
        Public method for external use.
        
        Modifier logic:
        - 26: Professional component (radiologist interpretation)
        - TC: Technical component (facility billing)
        - LT: Left side (only if auto_add_laterality=True or CPT requires it)
        - RT: Right side (only if auto_add_laterality=True or CPT requires it)
        - 50: Bilateral procedure
        - PI: PET scans for initial treatment strategy
        
        Args:
            cpt_code: The CPT code
            laterality: Extracted laterality ('left', 'right', 'bilateral', 'none')
            auto_add_laterality: If True, automatically add laterality modifiers.
                                 If False (default), only add for bilateral-default CPTs.
                                 LLM suggestions will still be merged in _merge_modifiers.
        """
        modifiers = []
        
        # Professional/Technical component
        # BUT: Do NOT add 26 for interventional procedures (they are procedures, not interpretations)
        if self.bill_type == 'P' and cpt_code not in self.INTERVENTIONAL_CPT_CODES:
            modifiers.append('26')
        elif self.bill_type == 'T':
            modifiers.append('TC')
        # Global billing (G) = no modifier
        # Interventional procedures = no 26 modifier
        
        # PI modifier for PET scans (initial treatment strategy)
        if cpt_code in self.PI_CPT_CODES:
            modifiers.append('PI')
        
        # Laterality modifiers - CONSERVATIVE approach
        # Only auto-add for CPTs where bilateral is the DEFAULT (like screening mammography)
        # For other CPTs, let the LLM suggest laterality (merged in _merge_modifiers)
        if cpt_code in self.BILATERAL_DEFAULT_CPTS:
            # For screening mammography, assume bilateral unless explicitly unilateral
            if laterality == 'left':
                modifiers.append('LT')
            elif laterality == 'right':
                modifiers.append('RT')
            else:
                # Bilateral is the default
                if cpt_code in self.BILATERAL_50_CPT_CODES:
                    modifiers.append('50')
        elif auto_add_laterality and (cpt_code in self.LATERALITY_CPT_CODES):
            # Only auto-add if explicitly requested (e.g., for production use)
            if laterality == 'left':
                modifiers.append('LT')
            elif laterality == 'right':
                modifiers.append('RT')
            elif laterality == 'bilateral':
                if cpt_code in self.BILATERAL_50_CPT_CODES:
                    modifiers.append('50')
        
        return modifiers
    
    def _merge_modifiers(
        self, 
        rules_modifiers: List[str], 
        llm_modifiers: List[str],
        cpt_code: str
    ) -> List[str]:
        """
        Merge rules-based modifiers with LLM suggestions.
        
        Strategy:
        - Rules take precedence for 26/TC (billing type) and PI (PET scans)
        - LLM suggestions can add laterality if rules didn't detect
        - LLM can suggest specialty modifiers (JZ, XU, 76, 77) that rules can't determine
        - Validate LLM suggestions against CPT-specific rules
        """
        result = list(rules_modifiers)
        
        # LLM can suggest laterality if rules didn't detect
        has_laterality = any(m in result for m in ['LT', 'RT', '50'])
        
        if not has_laterality:
            for mod in llm_modifiers:
                if mod in ['LT', 'RT', '50']:
                    # Validate that this CPT code can have laterality
                    if cpt_code in self.LATERALITY_CPT_CODES or cpt_code in self.BILATERAL_DEFAULT_CPTS:
                        result.append(mod)
                        break  # Only add one laterality modifier
        
        # LLM can suggest specialty modifiers that rules can't determine
        # These require context we don't have (same-day procedures, drug waste, etc.)
        SPECIALTY_MODIFIERS_FROM_LLM = {'JZ', 'XU', '76', '77'}
        for mod in llm_modifiers:
            if mod in SPECIALTY_MODIFIERS_FROM_LLM and mod not in result:
                result.append(mod)
        
        # Remove duplicates and sort (26/TC first, then laterality, then specialty)
        def modifier_sort_key(m):
            if m in ['26', 'TC']:
                return (0, m)
            elif m == 'PI':
                return (1, m)
            elif m in ['LT', 'RT', '50']:
                return (2, m)
            else:
                return (3, m)
        
        return sorted(list(set(result)), key=modifier_sort_key)
    
    def _predict_modifiers(self, cpt_code: str, laterality: str) -> List[str]:
        """Internal wrapper for consistency."""
        return self.predict_modifiers(cpt_code, laterality)
    
    def _ensure_medical_necessity(
        self,
        icd_codes: List[str],
        valid_icd_codes: Set[str] = None,
        cpt_code: str = None
    ) -> Tuple[List[str], bool, str]:
        """
        Validate medical necessity - DO NOT auto-add codes.
        
        Returns:
            Tuple of (icd_codes, has_medical_necessity, warning_message)
        """
        if not icd_codes:
            return icd_codes, False, "NO_CODES: No ICD codes predicted"
        
        # Check 1: Traditional indication codes (R/Z/S)
        has_rz_indication = any(
            c.startswith(('R', 'Z')) for c in icd_codes
        )
        
        # Check 2: Trauma codes (S/T)
        has_trauma = any(
            c.startswith(('S', 'T')) for c in icd_codes
        )
        
        # Check 3: Diagnosis codes that inherently justify imaging
        has_diagnosis_justification = False
        justifying_diagnosis = None
        
        for code in icd_codes:
            for prefix, category in self.DIAGNOSIS_JUSTIFIES_IMAGING.items():
                if code.startswith(prefix):
                    has_diagnosis_justification = True
                    justifying_diagnosis = f"{code} ({category})"
                    break
            if has_diagnosis_justification:
                break
        
        # Determine if we have valid medical necessity
        has_medical_necessity = (
            has_rz_indication or 
            has_trauma or 
            has_diagnosis_justification
        )
        
        # Build warning message if missing
        warning = ""
        if not has_medical_necessity:
            warning = f"REVIEW_NEEDED: No clear indication found. Codes: {icd_codes}"
        
        # CPT-specific validation (if provided)
        if cpt_code and cpt_code in self.CPT_EXPECTED_INDICATIONS:
            cpt_rules = self.CPT_EXPECTED_INDICATIONS[cpt_code]
            
            if 'expected' in cpt_rules:
                has_expected = any(c in cpt_rules['expected'] for c in icd_codes)
                if not has_expected and cpt_rules.get('category') == 'screening':
                    # Screening without screening code - might be diagnostic
                    warning = f"INFO: {cpt_code} typically uses {cpt_rules['expected']}"
        
        # IMPORTANT: Never auto-add codes - just flag for review
        return icd_codes, has_medical_necessity, warning
    
    def _consolidate_laterality(self, codes: List[str]) -> List[str]:
        """
        Convert R+L code pairs to bilateral/unspecified codes.
        
        Example: M25.561 (right knee pain) + M25.562 (left) → M25.569 (unspecified)
        """
        result = list(codes)
        
        # Common laterality consolidation patterns
        # Format: (right_code, left_code, bilateral_code)
        pairs = [
            ('M25.511', 'M25.512', 'M25.519'),  # Shoulder pain
            ('M25.521', 'M25.522', 'M25.529'),  # Elbow pain
            ('M25.531', 'M25.532', 'M25.539'),  # Wrist pain
            ('M25.551', 'M25.552', 'M25.559'),  # Hip pain
            ('M25.561', 'M25.562', 'M25.569'),  # Knee pain
            ('M25.571', 'M25.572', 'M25.579'),  # Ankle pain
        ]
        
        for right_code, left_code, bilateral_code in pairs:
            if right_code in result and left_code in result:
                result.remove(right_code)
                result.remove(left_code)
                result.append(bilateral_code)
        
        return result
    
    def _remove_conflicts(self, codes: List[str]) -> List[str]:
        """
        Remove conflicting or redundant codes.
        
        Rules:
        - If both specific and unspecified codes exist, keep specific
        - Remove known conflict pairs
        """
        result = list(codes)
        
        # Known conflict: R91.1 (nodule) + R91.8 (other) → keep R91.1
        if 'R91.1' in result and 'R91.8' in result:
            result.remove('R91.8')
        
        # Remove unspecified if specific exists
        for code in list(result):
            if code.endswith('.9') or code.endswith('.90'):
                base = code.split('.')[0]
                # Check if a more specific code from same category exists
                has_specific = any(
                    c.startswith(base + '.') and c != code
                    for c in result
                )
                if has_specific:
                    result.remove(code)
        
        return result
    
    # =========================================================================
    # ICD FILTERING RULES (Approved 2026-01)
    # =========================================================================
    
    # Incidental findings to suppress (unless exam-specific)
    INCIDENTAL_FINDINGS = {
        # Cardiomegaly - only code if cardiac exam
        'I51.7': {'name': 'cardiomegaly', 'allowed_cpt_prefixes': ['938', '939', '933', '7555']},  # cardiac CPTs
        'R93.1': {'name': 'cardiomegaly on imaging', 'allowed_cpt_prefixes': ['938', '939', '933', '7555']},
        
        # Fatty liver / Hepatic steatosis - suppress as incidental
        'K76.0': {'name': 'fatty liver', 'allowed_cpt_prefixes': []},  # Never code as incidental
        
        # Aortic calcifications - suppress as incidental  
        'I70.0': {'name': 'aortic atherosclerosis', 'allowed_cpt_prefixes': ['938', '939', '7555']},
        
        # Coronary calcifications - suppress as incidental
        'I25.10': {'name': 'coronary atherosclerosis', 'allowed_cpt_prefixes': ['938', '939', '7555']},
        
        # Thyroid nodules - suppress as incidental (unless thyroid exam)
        # 76536 = thyroid/neck ultrasound, 78012-78018 = thyroid nuclear imaging
        'E04.1': {'name': 'thyroid nodule', 'allowed_cpt_prefixes': ['76536', '7801', '7653']},
        'E04.2': {'name': 'multinodular goiter', 'allowed_cpt_prefixes': ['76536', '7801', '7653']},
        
        # Hardware presence - suppress when only noted as finding
        'Z96.6': {'name': 'orthopedic hardware', 'allowed_cpt_prefixes': []},  # Never code as incidental
        'Z96.64': {'name': 'hip hardware', 'allowed_cpt_prefixes': []},
        'Z96.65': {'name': 'knee hardware', 'allowed_cpt_prefixes': []},
        'Z96.66': {'name': 'ankle hardware', 'allowed_cpt_prefixes': []},
    }
    
    # Symptom codes that should be removed if definitive diagnosis exists
    # Map: symptom prefix -> diagnosis prefixes that override it
    SYMPTOM_DIAGNOSIS_HIERARCHY = {
        # Pain codes (M25.5xx) overridden by specific joint diagnoses
        'M25.5': ['M17', 'M19', 'M23', 'M24', 'S8', 'S7', 'S9'],  # Arthritis, meniscus, dislocation, fractures
        
        # Back pain (M54) overridden by specific spine diagnoses
        'M54': ['M51', 'M50', 'M47', 'M48', 'M43', 'S3', 'S2'],  # Disc, spondylosis, stenosis, fractures
        
        # Chest pain (R07) overridden by cardiac/pulmonary diagnoses
        'R07': ['I2', 'I3', 'I4', 'J1', 'J8', 'J9'],  # MI, pneumonia, PE
        
        # Abdominal pain (R10) overridden by GI diagnoses
        'R10': ['K2', 'K3', 'K4', 'K5', 'K8'],  # Appendicitis, hernia, gallbladder
        
        # Headache (R51) overridden by neuro diagnoses
        'R51': ['G43', 'G44', 'I6'],  # Migraine, headache syndromes, stroke
        
        # General R codes overridden by M/N codes for same body part
        'R': ['M', 'S', 'C', 'D', 'I', 'J', 'K', 'N'],
    }
    
    def _filter_external_cause_codes(self, codes: List[str]) -> List[str]:
        """
        Remove External Cause codes (W, X, Y prefixes).
        
        Per RP policy: Do not report External Cause codes like "Fall on ice" (W00).
        """
        return [c for c in codes if not c.startswith(('W', 'X', 'Y'))]
    
    def _filter_incidental_findings(self, codes: List[str], cpt_code: str) -> List[str]:
        """
        Remove incidental findings that are unrelated to the exam indication.
        
        Logic: Suppress cardiomegaly, fatty liver, calcifications, thyroid nodules,
        hardware presence UNLESS the CPT code indicates the exam was specifically
        for that condition.
        """
        result = []
        
        for code in codes:
            # Check if this is an incidental finding
            is_incidental = False
            for incidental_code, info in self.INCIDENTAL_FINDINGS.items():
                if code.startswith(incidental_code.split('.')[0]) and code == incidental_code:
                    # Check if CPT allows this code
                    allowed_prefixes = info['allowed_cpt_prefixes']
                    if allowed_prefixes:
                        # Only allow if CPT matches
                        if not any(cpt_code.startswith(prefix) for prefix in allowed_prefixes):
                            is_incidental = True
                    else:
                        # Never allowed as incidental
                        is_incidental = True
                    break
            
            if not is_incidental:
                result.append(code)
        
        return result
    
    def _apply_diagnosis_hierarchy(self, codes: List[str]) -> List[str]:
        """
        Remove symptom codes when a definitive diagnosis exists.
        
        Logic: If a definitive diagnosis is identified (e.g., Subluxation M24),
        do not code the presenting symptom (e.g., Pain M25.5).
        """
        result = list(codes)
        codes_to_remove = set()
        
        # Check each code against hierarchy rules
        for symptom_prefix, diagnosis_prefixes in self.SYMPTOM_DIAGNOSIS_HIERARCHY.items():
            # Find symptom codes
            symptom_codes = [c for c in result if c.startswith(symptom_prefix)]
            
            if symptom_codes:
                # Check if any diagnosis code exists that overrides this symptom
                has_diagnosis = any(
                    any(c.startswith(diag_prefix) for diag_prefix in diagnosis_prefixes)
                    for c in result if c not in symptom_codes
                )
                
                if has_diagnosis:
                    codes_to_remove.update(symptom_codes)
        
        return [c for c in result if c not in codes_to_remove]
    
    def _prioritize_anatomic_codes(self, codes: List[str]) -> List[str]:
        """
        Prioritize anatomically specific codes (M/N) over general symptom codes (R).
        
        When coding symptoms is necessary, prefer:
        - M codes (musculoskeletal) over R codes
        - N codes (genitourinary) over R codes
        
        This reorders codes to put specific codes first, but keeps R codes
        if they provide unique information.
        """
        # Separate codes by priority
        high_priority = []  # M, N, S, C, D, I, J, K codes
        medium_priority = []  # Other specific codes
        low_priority = []  # R, Z codes (symptoms, encounters)
        
        for code in codes:
            if code.startswith(('M', 'N', 'S', 'C', 'D', 'I', 'J', 'K')):
                high_priority.append(code)
            elif code.startswith(('R', 'Z')):
                low_priority.append(code)
            else:
                medium_priority.append(code)
        
        # Return prioritized order
        return high_priority + medium_priority + low_priority
    
    def _calculate_confidence(
        self,
        icd_codes: List[str],
        cpt_was_extracted: bool,
        has_medical_necessity: bool = True
    ) -> float:
        """Calculate confidence score for the prediction."""
        score = 0.5  # Base score
        
        # Boost if CPT was extracted (more reliable)
        if cpt_was_extracted:
            score += 0.2
        
        # Boost if we have valid medical necessity
        if has_medical_necessity:
            score += 0.2
        else:
            # Penalty if no medical necessity
            score -= 0.2
        
        # Boost if we have a reasonable number of codes
        if 1 <= len(icd_codes) <= 5:
            score += 0.1
        
        return max(0.1, min(score, 1.0))

