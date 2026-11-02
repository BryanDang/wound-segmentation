#!/usr/bin/env python3
"""
Notebook syntax check for GitHub Actions.
Parses the JSON and compiles each code cell.
"""

import json
import sys
from pathlib import Path


def validate_notebook_syntax(notebook_path):
    """Return True if every code cell compiles."""
    print(f"Validating notebook: {notebook_path}")
    
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in notebook: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Failed to read notebook: {e}")
        return False
    
    if 'cells' not in nb:
        print("ERROR: No cells found in notebook")
        return False
    
    print(f"Found {len(nb['cells'])} cells")
    
    errors = 0
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            
            if code.strip():  # Skip empty cells
                try:
                    compile(code, f'<cell-{i}>', 'exec')
                    print(f"Cell {i}: OK")
                except SyntaxError as e:
                    print(f"Cell {i}: SYNTAX ERROR - {e}")
                    print(f"Code preview: {code[:100]}...")
                    errors += 1
                except Exception as e:
                    print(f"Cell {i}: COMPILATION ERROR - {e}")
                    errors += 1
    
    if errors == 0:
        print("SUCCESS: All cells validated successfully")
        return True
    else:
        print(f"FAILED: {errors} cells have syntax errors")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_notebook.py <notebook_path>")
        sys.exit(1)
    
    notebook_path = Path(sys.argv[1])
    if not notebook_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}")
        sys.exit(1)
    
    success = validate_notebook_syntax(notebook_path)
    sys.exit(0 if success else 1)
