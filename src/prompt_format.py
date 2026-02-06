"""
PROMPT FORMAT - Matches training data exactly (v13_train_messages.jsonl)

DO NOT MODIFY without verifying against training data.
"""

import re
from typing import List, Dict, Optional


def format_example_output(cpt: str, modifier: str, icd: str) -> str:
    """Format Output line: CPT, MODIFIER, ICD1, ICD2, ..."""
    mod = str(modifier).strip() if modifier else ''
    if mod in ['nan', 'None', '', 'NaN']:
        mod = ''
    
    icd_clean = str(icd).strip() if icd else ''
    if icd_clean in ['nan', 'None', '']:
        icd_clean = ''
    
    is_g_code = str(cpt).upper().startswith('G')
    
    if mod:
        return f"{cpt}, {mod}, {icd_clean}"
    elif not is_g_code and cpt.isdigit() and len(cpt) == 5:
        # Default to modifier 26 for regular CPT codes
        return f"{cpt}, 26, {icd_clean}"
    else:
        return f"{cpt}, {icd_clean}"


def format_example_report(indication: str, laterality: str, impression: str, findings: str) -> str:
    """Format Report content: INDICATION, LATERALITY, IMPRESSION, FINDINGS"""
    parts = []
    
    if indication and str(indication) not in ['nan', 'None', '']:
        ind = str(indication).strip()
        parts.append(f"INDICATION: {ind}" if not ind.upper().startswith('INDICATION:') else ind)
    
    if laterality and str(laterality).lower() not in ['nan', 'none', '', 'missing']:
        parts.append(f"LATERALITY: {str(laterality).upper().strip()}")
    
    if impression and str(impression) not in ['nan', 'None', '']:
        imp = str(impression).strip()
        parts.append(f"IMPRESSION: {imp}" if not imp.upper().startswith('IMPRESSION:') else imp)
    
    if findings and str(findings) not in ['nan', 'None', '']:
        find = str(findings).strip()[:300]
        parts.append(f"FINDINGS: {find}" if not find.upper().startswith('FINDINGS:') else find)
    
    return '\n'.join(parts)


def format_few_shot_example(metadata: Dict) -> Optional[str]:
    """Format a single few-shot example from RAG metadata."""
    cpt = str(metadata.get('gt_cpt') or metadata.get('cpt_code') or '')
    if not cpt or cpt in ['nan', 'None', '']:
        return None
    
    icd = str(metadata.get('gt_icd') or metadata.get('icd_codes') or '')
    modifier = str(metadata.get('modifier') or '')
    indication = metadata.get('indication', '')
    impression = metadata.get('impression', '')
    laterality = metadata.get('laterality', '')
    
    # Extract findings from parsed_compact
    findings = ''
    parsed_compact = str(metadata.get('parsed_compact') or '')
    if 'FINDINGS:' in parsed_compact.upper():
        match = re.search(r'FINDINGS:\s*(.+?)(?=\n[A-Z]+:|$)', parsed_compact, re.IGNORECASE | re.DOTALL)
        if match:
            findings = match.group(1).strip()[:300]
    
    report = format_example_report(indication, laterality, impression, findings)
    if not report or len(report) < 20:
        return None
    
    output = format_example_output(cpt, modifier, icd)
    return f"Report: {report}\nOutput: {output}"


def format_examples_from_hits(hits: List[Dict], n: int = 3) -> str:
    """Format few-shot examples from RAG hits."""
    examples = []
    for hit in hits:
        if len(examples) >= n:
            break
        meta = hit.get('metadata', {})
        example = format_few_shot_example(meta)
        if example:
            examples.append(example)
    return '\n---\n'.join(examples)


def build_prompt(report_text: str, examples_str: str, cpt_candidates: List[str], 
                 icd_candidates: List[str], extracted_cpt: Optional[str] = None) -> str:
    """Build full prompt matching training data format."""
    cpt_str = '\n'.join(cpt_candidates[:10])
    icd_str = '\n'.join(icd_candidates[:50])
    
    if extracted_cpt:
        prompt = f'''Study these similar cases:

{examples_str}

---

Code this report:
{report_text[:3200]}

CPT CODE (from exam): {extracted_cpt}

Available ICD codes (from similar cases):
{icd_str}

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
        prompt = f'''Study these similar cases:

{examples_str}

---

Code this report:
{report_text[:2800]}

SIMILAR CASES CPT CODES (most likely):
{cpt_str}

Available ICD codes (from similar cases):
{icd_str}

Rules:
- Determine the CPT code from the procedure/exam description
- Add appropriate modifiers (26=professional, LT=left, RT=right, 50=bilateral)
- Code the clinical indication (R/Z/S code)
- Code confirmed findings from impression
- Use laterality info to select correct modifier (LT/RT/50)

Output format: CPT, MODIFIER, ICD1, ICD2, ...
Example: 71046, 26, R91.8, J18.9
Example: 73030, 26 LT, M25.511, S43.401A'''
    
    return prompt
