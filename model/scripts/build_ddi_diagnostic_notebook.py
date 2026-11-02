#!/usr/bin/env python3
"""
Build DDI Diagnostic Validation Notebook
Uses nbformat to create notebook testing baseline model on dermatological images
"""

from pathlib import Path

import nbformat as nbf
from ddi_diagnostic_cells import CELLS


def build_ddi_diagnostic_notebook():
    """Build DDI diagnostic notebook using nbformat"""

    # Create notebook with metadata
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.13"},
    }

    # Add cells
    notebook_cells = []
    for cell_spec in CELLS:
        if cell_spec["type"] == "code":
            notebook_cells.append(nbf.v4.new_code_cell(cell_spec["source"]))
        elif cell_spec["type"] == "markdown":
            notebook_cells.append(nbf.v4.new_markdown_cell(cell_spec["source"]))

    nb["cells"] = notebook_cells

    # Normalize (adds missing IDs)
    nb = nbf.validator.normalize(nb)[1]

    # Validate
    nbf.validate(nb)

    # Write to file
    output_path = (
        Path(__file__).parent.parent / "notebooks" / "03_ddi_diagnostic" / "03_ddi_diagnostic.ipynb"
    )
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, "w") as f:
        nbf.write(nb, f)

    print(f"Notebook created: {output_path}")
    print(f"   Total cells: {len(notebook_cells)}")
    print(f"   Code cells: {sum(1 for c in CELLS if c['type'] == 'code')}")
    print(f"   Markdown cells: {sum(1 for c in CELLS if c['type'] == 'markdown')}")

    return output_path


if __name__ == "__main__":
    build_ddi_diagnostic_notebook()
