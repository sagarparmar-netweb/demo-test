"""
Report Section Extractor - Extract structured sections from radiology reports.
Matches the parsed_compact format used in RAG indices.
"""
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExtractedSections:
    indication: str = ""
    impression: str = ""
    findings: str = ""
    laterality: str = ""
    exam_description: str = ""
    
    def to_compact_text(self) -> str:
        """Convert to parsed_compact format for RAG matching."""
        parts = []
        if self.exam_description:
            parts.append(f"EXAM: {self.exam_description}")
        if self.indication:
            parts.append(f"INDICATION: {self.indication}")
        if self.laterality:
            parts.append(f"LATERALITY: {self.laterality}")
        if self.impression:
            parts.append(f"IMPRESSION: {self.impression}")
        if self.findings:
            parts.append(f"FINDINGS: {self.findings[:300]}")
        return "\n".join(parts)


class ReportSectionExtractor:
    """Extract INDICATION, IMPRESSION, LATERALITY, FINDINGS from radiology reports."""
    
    def __init__(self):
        # Patterns for each section
        self.patterns = {
            'indication': [
                r'INDICATION[:\s]*([^\n]+(?:\n(?![A-Z]{3,})[^\n]+)*)',
                r'CLINICAL INDICATION[:\s]*([^\n]+(?:\n(?![A-Z]{3,})[^\n]+)*)',
                r'HISTORY[:\s]*([^\n]+(?:\n(?![A-Z]{3,})[^\n]+)*)',
                r'REASON FOR EXAM[:\s]*([^\n]+)',
                r'Reason for Exam[:\s]*([^\n]+)',
            ],
            'impression': [
                r'IMPRESSION[:\s]*([^\n]+(?:\n(?![A-Z]{3,}:)[^\n]+)*)',
                r'CONCLUSION[:\s]*([^\n]+(?:\n(?![A-Z]{3,}:)[^\n]+)*)',
            ],
            'findings': [
                r'FINDINGS[:\s]*([^\n]+(?:\n(?![A-Z]{3,}:)[^\n]+)*)',
            ],
            'exam_description': [
                r'Exam Description[:\s]*([^\n]+)',
                r'EXAM[:\s]+((?:CT|MR|MRI|XR|US|ULTRASOUND|MAMMO|MA)[^\n]+)',
                r'EXAMINATION[:\s]+([^\n]+)',
            ]
        }
        
        # Laterality patterns
        self.laterality_patterns = [
            (r'\bBILATERAL\b', 'BILATERAL'),
            (r'\bLEFT\b(?!\s+AND\s+RIGHT)', 'LEFT'),
            (r'\bRIGHT\b(?!\s+AND\s+LEFT)', 'RIGHT'),
            (r'\bLT\b', 'LEFT'),
            (r'\bRT\b', 'RIGHT'),
        ]
    
    def extract(self, report_text: str) -> ExtractedSections:
        """Extract sections from report text."""
        sections = ExtractedSections()
        text_upper = report_text.upper()
        
        # Extract each section
        for section_name, patterns in self.patterns.items():
            for pattern in patterns:
                match = re.search(pattern, report_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    value = re.sub(r'\s+', ' ', value)
                    value = value[:500]  # Limit length
                    setattr(sections, section_name, value)
                    break
        
        # Extract laterality
        for pattern, laterality in self.laterality_patterns:
            if re.search(pattern, text_upper):
                sections.laterality = laterality
                break
        
        return sections


# Test if run directly
if __name__ == '__main__':
    test_report = '''
    Exam Description: CT HEAD WITHOUT IV CONTRAST
    INDICATION: Headache, rule out stroke
    FINDINGS: Normal brain parenchyma.
    IMPRESSION: No acute intracranial findings.
    '''
    
    extractor = ReportSectionExtractor()
    sections = extractor.extract(test_report)
    print(f"Extracted: {sections}")
    print(f"Compact: {sections.to_compact_text()}")
