#!/usr/bin/env python3
"""
STEP 2: LLM PREDICTOR

Single LLM call to predict CPT (if needed) and ICD codes.
Supports both HuggingFace Inference Endpoints (fine-tuned model) and OpenAI.

Input: Preprocessed data with prompt
Output: CPT code + ICD codes
"""

import re
import os
import time
import requests
from typing import List, Tuple, Set, Optional
from dataclasses import dataclass

from .preprocessor import CPT_CODES, apply_cpt_disambiguation


@dataclass
class LLMPrediction:
    """Output from LLM prediction."""
    cpt_code: str
    icd_codes: List[str]
    modifiers: List[str]  # LLM-suggested modifiers
    raw_response: str
    latency_ms: float = 0.0


class LLMPredictor:
    """
    Makes a single LLM call to predict CPT and ICD codes.
    
    Supports:
    1. HuggingFace Inference Endpoints (fine-tuned NYXMed model)
    2. OpenAI API (fallback)
    
    Usage:
        # HuggingFace (default - uses fine-tuned model)
        predictor = LLMPredictor(backend="huggingface")
        
        # OpenAI (fallback)
        predictor = LLMPredictor(backend="openai", model="gpt-4o")
    """
    
    # HuggingFace Endpoint Configuration
    HF_ENDPOINT_URL = os.getenv(
        "HF_ENDPOINT_URL",
        "https://fp65wjpuwvscqcao.us-east-2.aws.endpoints.huggingface.cloud/v1/chat/completions"
    )
    HF_MODEL_NAME = "vineetdaniels/NYXMed-Model-67Kset"
    # V14 Update: Added second line to enable instruction-following at inference
    HF_SYSTEM_PROMPT = """You are an expert radiology coder specializing in ICD-10 and CPT coding for radiology reports.

Follow the coding rules provided in each request carefully."""
    
    def __init__(
        self,
        backend: str = "huggingface",  # "huggingface" or "openai"
        api_key: str = None,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 80
    ):
        """
        Args:
            backend: "huggingface" for fine-tuned model, "openai" for GPT
            api_key: API key (HF token or OpenAI key)
            model: Model name (only used for OpenAI backend)
            temperature: Sampling temperature
            max_tokens: Max tokens in response
        """
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if backend == "huggingface":
            self.api_key = api_key or os.getenv("HF_API_TOKEN", "")
            self.model = self.HF_MODEL_NAME
            if not self.api_key:
                print("Warning: HF_API_TOKEN not set. Set via environment variable.")
        else:
            # OpenAI backend
            self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
            self.model = model or "gpt-4o"
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                print("Warning: openai package not installed. Use 'pip install openai'")
                self.client = None
    
    def predict(
        self,
        prompt: str,
        extracted_cpt: Optional[str] = None,
        valid_icd_codes: Set[str] = None,
        report_text: str = "",
        exam_desc: str = ""
    ) -> LLMPrediction:
        """
        Make LLM prediction.
        
        Args:
            prompt: Built prompt from preprocessor
            extracted_cpt: CPT code if already extracted (None if needs prediction)
            valid_icd_codes: Set of valid ICD codes to filter response
            report_text: Original report text (for CPT disambiguation)
            exam_desc: Exam description (for CPT disambiguation)
        
        Returns:
            LLMPrediction with cpt_code and icd_codes
        """
        start_time = time.time()
        
        try:
            if self.backend == "huggingface":
                raw_response = self._call_huggingface(prompt)
            else:
                raw_response = self._call_openai(prompt)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse response
            cpt_code, modifiers, icd_codes = self._parse_response(
                raw_response, 
                extracted_cpt, 
                valid_icd_codes
            )
            
            # Apply CPT disambiguation if CPT was predicted (not extracted)
            if not extracted_cpt and cpt_code:
                cpt_code = apply_cpt_disambiguation(cpt_code, report_text, exam_desc)
            
            return LLMPrediction(
                cpt_code=cpt_code,
                icd_codes=icd_codes,
                modifiers=modifiers,
                raw_response=raw_response,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            # Fallback on error
            return LLMPrediction(
                cpt_code=extracted_cpt or "",
                icd_codes=[],
                modifiers=[],
                raw_response=f"Error: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000
            )
    
    def _call_huggingface(self, prompt: str, max_retries: int = 3) -> str:
        """Call HuggingFace Inference Endpoint."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.HF_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.HF_ENDPOINT_URL,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                elif response.status_code == 503:
                    # Model loading
                    time.sleep(30)
                    continue
                else:
                    if attempt == max_retries - 1:
                        return f"Error: HTTP {response.status_code}"
                    time.sleep(5)
                    
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    return "Error: Request timeout"
                continue
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error: {str(e)}"
                time.sleep(2)
        
        return "Error: Max retries exceeded"
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        if not self.client:
            return "Error: OpenAI client not initialized"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content.strip()
    
    def _parse_response(
        self,
        response: str,
        extracted_cpt: Optional[str],
        valid_icd_codes: Set[str]
    ) -> Tuple[str, List[str], List[str]]:
        """
        Parse LLM response to extract CPT, modifiers, and ICD codes.
        
        Handles formats like:
        - "71046, 26 LT, R91.8, J18.9"
        - "CPT: 71046\nModifier: 26\nICD: R91.8, J18.9"
        - "R91.8, J18.9" (when CPT is already known)
        """
        if response.startswith("Error:"):
            return extracted_cpt or "", [], []
        
        response_upper = response.upper()
        
        # Extract CPT code
        if extracted_cpt:
            cpt_code = extracted_cpt
        else:
            # Look for 5-digit CPT code at the beginning of response (most likely format)
            # Format: "71046, 26, R91.8" - CPT is first
            first_token_match = re.match(r'^([0-9]{5})', response.strip())
            if first_token_match:
                cpt_code = first_token_match.group(1)
            else:
                # Fallback: look for any 5-digit CPT code or G-codes in response
                cpt_match = re.search(r'\b([0-9]{5}|[A-Z][0-9]{4})\b', response)
                if cpt_match:
                    cpt_code = cpt_match.group(1)
                else:
                    cpt_code = ""
        
        # Extract modifiers (26, LT, RT, 50, TC, etc.)
        valid_modifiers = {'26', 'TC', 'LT', 'RT', '50', '76', '77', 'XU', 'XP', 'XE', 'PI', 'JZ'}
        modifiers = []
        for mod in valid_modifiers:
            if re.search(rf'\b{mod}\b', response_upper):
                modifiers.append(mod)
        
        # Extract ICD codes (A00.00 format)
        icd_pattern = r'\b([A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?)\b'
        all_codes = set(re.findall(icd_pattern, response_upper))
        
        # Remove CPT-like matches from ICD codes
        all_codes = {c for c in all_codes if not c[0].isdigit()}
        
        # Filter to valid codes if provided
        if valid_icd_codes:
            icd_codes = list(all_codes & valid_icd_codes)
            # Also include codes not in candidates (model might predict valid codes)
            for code in all_codes:
                if code not in icd_codes and len(code) >= 3:
                    icd_codes.append(code)
        else:
            icd_codes = list(all_codes)
        
        # Sort for consistency
        icd_codes.sort()
        modifiers.sort()
        
        return cpt_code, modifiers, icd_codes
    
    def health_check(self) -> bool:
        """Check if the endpoint is ready."""
        if self.backend == "huggingface":
            try:
                response = requests.post(
                    self.HF_ENDPOINT_URL,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 5
                    },
                    timeout=30
                )
                return response.status_code == 200
            except:
                return False
        else:
            return self.client is not None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_predictor(backend: str = "huggingface", **kwargs) -> LLMPredictor:
    """Factory function to create a predictor."""
    return LLMPredictor(backend=backend, **kwargs)
