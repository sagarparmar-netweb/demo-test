"""
NYXMed VFinal Configuration
===========================
Central configuration for the production pipeline.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
INDICES_DIR = BASE_DIR / "indices"

# Project root (for accessing training data)
PROJECT_ROOT = BASE_DIR.parent

# =============================================================================
# HUGGINGFACE ENDPOINT (Fine-tuned Model)
# =============================================================================

HF_ENDPOINT_URL = os.getenv(
    "HF_ENDPOINT_URL",
    "https://fp65wjpuwvscqcao.us-east-2.aws.endpoints.huggingface.cloud/v1/chat/completions"
)

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

HF_MODEL_NAME = "vineetdaniels/NYXMed-Model-67Kset"

# =============================================================================
# LLM SETTINGS
# =============================================================================

LLM_BACKEND = os.getenv("LLM_BACKEND", "huggingface")  # "huggingface" or "openai"
LLM_MAX_TOKENS = 80
LLM_TEMPERATURE = 0.1

# MUST match training data exactly
# V14 Update: Added second line to enable instruction-following at inference
SYSTEM_PROMPT = """You are an expert radiology coder specializing in ICD-10 and CPT coding for radiology reports.

Follow the coding rules provided in each request carefully."""

# =============================================================================
# RAG SETTINGS
# =============================================================================

RAG_TOP_K = 5
NUM_FEW_SHOT_EXAMPLES = 3
MAX_REPORT_LENGTH = 2500
MAX_EXAMPLE_LENGTH = 500

# =============================================================================
# POSTPROCESSING
# =============================================================================

DEFAULT_BILL_TYPE = 'P'  # 'P' = Professional, 'T' = Technical, 'G' = Global

VALID_NECESSITY_PREFIXES = [
    'R', 'Z', 'S', 'T', 'C', 'D', 'I', 'J', 'K', 'M',
    'N', 'G', 'E', 'O', 'Q', 'L', 'H', 'F', 'A', 'B'
]

# =============================================================================
# CONFIDENCE THRESHOLDS
# =============================================================================

AUTO_SUBMIT_THRESHOLD = 0.85
REVIEW_THRESHOLD = 0.70

# =============================================================================
# AWS S3 SETTINGS (for RAG index building)
# =============================================================================

AWS_BUCKET = "nyxmed-data-backup-20251229"
AWS_REGION = "us-east-1"

# Latest consolidated data with enriched fields
AWS_DATA_KEY = "consolidated_cleaned.csv"

# Data includes (71,657 records):
# - Parsed_Compact: INDICATION + IMPRESSION + LATERALITY (100% coverage)
# - Parsed_Indication: Extracted indication for few-shot examples (86%)
# - Parsed_Impression: Extracted impression for few-shot examples (99%)
# - Parsed_Laterality: Left/Right/Bilateral detection (100%)
# - 100% CPT (Procedure) coverage
# - 100% ICD coverage

# =============================================================================
# TRAINING DATA PATHS
# =============================================================================

TRAINING_DATA_PATHS = [
    PROJECT_ROOT / "data" / "consolidated_cleaned.csv",  # Primary: 71K records with parsed fields
    PROJECT_ROOT / "Training_Data_True_Source_20251231" / "Raw_Data_Consolidated.csv",
    PROJECT_ROOT / "data" / "Raw_Data_Consolidated_LATEST.csv",
]

# =============================================================================
# RAG INDEX SETTINGS
# =============================================================================

# Index prefix - update when rebuilding with new data
RAG_INDEX_PREFIX = "nyxmed_71k"

# Legacy index locations (for fallback)
LEGACY_RAG_DIR = PROJECT_ROOT / "versions" / "v7.5_hybrid_rag_deployment"

# Embedding model for RAG
RAG_EMBED_MODEL = "Snowflake/snowflake-arctic-embed-m"
RAG_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Use parsed compact text for indexing (recommended)
RAG_USE_PARSED_COMPACT = True


def get_training_data_path() -> Path:
    """Find the training data file."""
    for path in TRAINING_DATA_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError("Training data not found. Check TRAINING_DATA_PATHS in config.py")
