#!/usr/bin/env python3
"""
Local check of the Kaggle import structure.

Kaggle strips the 'src/' prefix from the code dataset, so modules import
directly (pipeline, utils) with no 'src.' prefix.

Usage:
    python3 scripts/test_kaggle_imports.py

Expected output:
    PASS: All imports successful
"""
import sys
from pathlib import Path

print("=" * 70)
print("KAGGLE IMPORT STRUCTURE TEST")
print("=" * 70)
print()

# Kaggle path layout
project_root = Path(__file__).parent.parent
src_dir = project_root / "src"

print(f"Project root: {project_root}")
print(f"Source directory: {src_dir}")
print()

if not src_dir.exists():
    print(f"FAIL: Source directory not found at {src_dir}")
    sys.exit(1)

print(f"Adding to sys.path: {src_dir}")

sys.path.insert(0, str(src_dir))
print()

# Imports without the src. prefix
print("Testing imports...")
print("-" * 70)

try:
    from pipeline.eda import EDA_CONFIG, run_eda_pipeline

    print("PASS: pipeline.eda.EDA_CONFIG imported successfully")
    print("PASS: pipeline.eda.run_eda_pipeline imported successfully")
except ImportError as e:
    print(f"FAIL: Import failed: {e}")
    print()
    print("This import structure would fail on Kaggle.")
    print()
    print("Debug:")
    print(f"  - sys.path[0]: {sys.path[0]}")
    print(f"  - Expected module: pipeline.eda")
    print(f"  - Check that {src_dir}/pipeline/eda.py exists")
    sys.exit(1)

try:
    from utils.environment import detect_environment

    print("PASS: utils.environment.detect_environment imported successfully")
except ImportError as e:
    print(f"FAIL: Import failed: {e}")
    print()
    print("This import structure would fail on Kaggle.")
    print()
    print("Debug:")
    print(f"  - sys.path[0]: {sys.path[0]}")
    print(f"  - Expected module: utils.environment")
    print(f"  - Check that {src_dir}/utils/environment.py exists")
    sys.exit(1)

print("-" * 70)
print()
print("=" * 70)
print("PASS: All imports work with Kaggle structure")
print("=" * 70)
print()
print("Imports used on Kaggle:")
print("  - from pipeline.eda import EDA_CONFIG, run_eda_pipeline")
print("  - from utils.environment import detect_environment")
print()
print("No 'src.' prefix; Kaggle strips it on upload.")
