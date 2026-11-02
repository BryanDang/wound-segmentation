#!/usr/bin/env python3
"""
Build baseline notebook using nbformat for deterministic structure.

This script creates a fresh notebook from cell specifications, ensuring:
- Correct newline handling (no corruption)
- Validated structure (nbformat validation)
- Clean cell metadata
- Reproducible builds

Usage:
    python scripts/build_baseline_notebook.py
"""

import sys
from pathlib import Path

import nbformat as nbf

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baseline_notebook_cells import CELLS


def build_notebook(output_path: Path) -> None:
    """Build notebook from cell specifications using nbformat.

    Args:
        output_path: Path where notebook will be saved

    Returns:
        None (writes notebook to disk)
    """
    print("=" * 80)
    print("BUILDING BASELINE NOTEBOOK WITH NBFORMAT")
    print("=" * 80)
    print(f"\nOutput path: {output_path}")
    print(f"Total cells: {len(CELLS)}")

    # Create new notebook (nbformat v4)
    nb = nbf.v4.new_notebook()

    # Add kernel metadata (needed for Kaggle execution)
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    }

    # Add cells from specifications
    notebook_cells = []

    for idx, cell_spec in enumerate(CELLS):
        cell_type = cell_spec["type"]
        source = cell_spec["source"]

        # Create cell using nbformat (handles all structure automatically)
        if cell_type == "code":
            cell = nbf.v4.new_code_cell(source)
        elif cell_type == "markdown":
            cell = nbf.v4.new_markdown_cell(source)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        notebook_cells.append(cell)

        # Progress update
        if (idx + 1) % 5 == 0:
            print(f"  Added {idx + 1}/{len(CELLS)} cells...")

    # Add all cells to notebook
    nb["cells"] = notebook_cells

    # Normalize notebook (adds missing cell IDs, etc.)
    print(f"\nNormalizing notebook structure...")
    nb = nbf.v4.upgrade(nb)  # Ensure latest format
    nb = nbf.validator.normalize(nb)[1]  # Add missing fields like cell IDs
    print(f"  PASS: Notebook normalized")

    # Validate notebook structure (nbformat built-in validation)
    print(f"\nValidating notebook structure...")
    try:
        nbf.validate(nb)
        print(f"  PASS: Validation passed")
    except nbf.ValidationError as e:
        print(f"  FAIL: Validation failed: {e}")
        raise

    # Write notebook to disk
    print(f"\nWriting notebook to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"  PASS: Notebook written successfully")

    # Verify file was created
    if output_path.exists():
        file_size = output_path.stat().st_size
        print(f"\nPASS: Notebook created successfully")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Cells: {len(CELLS)}")
        print(f"  Markdown cells: {sum(1 for c in CELLS if c['type'] == 'markdown')}")
        print(f"  Code cells: {sum(1 for c in CELLS if c['type'] == 'code')}")
    else:
        raise FileNotFoundError(f"Notebook was not created at {output_path}")

    print("\n" + "=" * 80)
    print("PASS: BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    # Output path
    output_notebook = PROJECT_ROOT / "notebooks" / "02_baseline_model_rebuilt.ipynb"

    # Ensure output directory exists
    output_notebook.parent.mkdir(parents=True, exist_ok=True)

    # Build notebook
    build_notebook(output_notebook)

    print(f"\nNext steps:")
    print(f"  1. Clean notebook: python scripts/clean_notebook.py {output_notebook}")
    print(f"  2. Deploy to Kaggle: kaggle kernels push -p notebooks/02_baseline/")
    print(f"  3. Monitor execution and verify all cells complete")
