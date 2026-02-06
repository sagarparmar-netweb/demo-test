#!/usr/bin/env python3
"""
Pipeline Validation Script
==========================
Run this at the start of every session to verify pipeline integrity.

Usage:
    python scripts/validate_pipeline.py
"""

import json
import hashlib
import sys
from pathlib import Path

def get_file_hash(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    if not filepath.exists():
        return "FILE_NOT_FOUND"
    return hashlib.sha256(filepath.read_bytes()).hexdigest()

def validate_pipeline():
    """Validate pipeline files against lock file."""
    
    # Find project root
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    lock_file = project_dir / "pipeline.lock"
    
    print("=" * 60)
    print("SIMPL AUTOCODE - PIPELINE VALIDATION")
    print("=" * 60)
    print()
    
    # Check lock file exists
    if not lock_file.exists():
        print("❌ ERROR: pipeline.lock not found!")
        print(f"   Expected at: {lock_file}")
        return False
    
    # Load lock file
    with open(lock_file) as f:
        lock = json.load(f)
    
    print(f"Lock file version: {lock['version']}")
    print(f"Last validated: {lock['last_validated']}")
    print(f"Validated by: {lock['validated_by']}")
    print()
    
    # Validate files
    print("Checking file integrity...")
    print("-" * 60)
    
    all_valid = True
    for filepath, file_info in lock['files'].items():
        full_path = project_dir / filepath
        expected_hash = file_info['sha256']
        actual_hash = get_file_hash(full_path)
        
        if actual_hash == "FILE_NOT_FOUND":
            print(f"❌ {filepath}")
            print(f"   FILE NOT FOUND!")
            all_valid = False
        elif actual_hash == expected_hash:
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath}")
            print(f"   MODIFIED! Hash mismatch.")
            print(f"   Expected: {expected_hash[:16]}...")
            print(f"   Actual:   {actual_hash[:16]}...")
            all_valid = False
    
    print("-" * 60)
    print()
    
    # Check indices exist
    print("Checking RAG indices...")
    print("-" * 60)
    
    indices_valid = True
    for index_file in lock['indices']['files']:
        full_path = project_dir / index_file
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✅ {index_file} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {index_file} - NOT FOUND!")
            indices_valid = False
    
    print("-" * 60)
    print()
    
    # Summary
    print("=" * 60)
    if all_valid and indices_valid:
        print("✅ VALIDATION PASSED - Pipeline is ready")
        print("=" * 60)
        return True
    else:
        print("❌ VALIDATION FAILED - Do not proceed!")
        print()
        if not all_valid:
            print("   Some files have been modified.")
            print("   Review changes or restore from backup.")
        if not indices_valid:
            print("   RAG indices are missing.")
            print("   Run: python scripts/rebuild_rag_index.py")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = validate_pipeline()
    sys.exit(0 if success else 1)
