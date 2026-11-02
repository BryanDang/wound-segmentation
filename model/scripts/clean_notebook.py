import json
import sys
from pathlib import Path


def clean_notebook(notebook_path):
    """Strip outputs, execution counts and cell ids. Writes <name>_clean.ipynb."""
    input_path = Path(notebook_path)
    output_path = input_path.parent / f"{input_path.stem}_clean{input_path.suffix}"

    with open(input_path, "r") as f:
        data = json.load(f)

    for cell in data["cells"]:
        if "outputs" in cell:
            cell["outputs"] = []
        if "execution_count" in cell:
            cell["execution_count"] = None
        if "id" in cell:
            del cell["id"]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=1)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_notebook.py <path_to_notebook.ipynb>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    output_path = clean_notebook(notebook_path)
    print(f"Successfully cleaned {notebook_path}")
    print(f"Created: {output_path}")
