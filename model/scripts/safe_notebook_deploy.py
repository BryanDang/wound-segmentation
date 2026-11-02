#!/usr/bin/env python3
"""
Notebook deployment with pre-push validation.

Checks:
1. Kernel ID exists on Kaggle.
2. Notebook file named in the metadata exists.
3. Notebook structure is valid.
4. Only one notebook in the directory.
5. Expected content markers are present.

Usage:
    python scripts/safe_notebook_deploy.py notebooks/02_baseline
    python scripts/safe_notebook_deploy.py notebooks/01_eda --dry-run
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import nbformat as nbf


class NotebookDeploymentValidator:
    """Validates a notebook directory before pushing to Kaggle."""

    def __init__(self, notebook_dir: Path):
        self.notebook_dir = notebook_dir
        self.metadata_path = notebook_dir / "kernel-metadata.json"
        self.validation_results = []

    def validate(self) -> Dict:
        """Run all validation checks."""

        print("=" * 80)
        print("SAFE NOTEBOOK DEPLOYMENT - VALIDATION")
        print("=" * 80)
        print(f"\nNotebook directory: {self.notebook_dir}")

        # Metadata exists
        if not self._check_metadata_exists():
            return self._failure_result("Metadata file not found")

        # Load
        with open(self.metadata_path) as f:
            metadata = json.load(f)

        # Notebook file exists
        if not self._check_notebook_exists(metadata):
            return self._failure_result("Notebook file not found")

        # Single notebook
        if not self._check_no_duplicates():
            return self._failure_result("Multiple notebooks found in directory")

        # Kernel ID on Kaggle
        if not self._check_kernel_id_exists(metadata):
            return self._failure_result("Kernel ID not found on Kaggle")

        # Structure
        notebook_path = self.notebook_dir / metadata["code_file"]
        if not self._check_notebook_structure(notebook_path):
            return self._failure_result("Notebook structure validation failed")

        # Expected content, by notebook type
        if not self._check_expected_content(notebook_path, metadata):
            return self._failure_result("Expected content missing from notebook")

        return self._success_result(metadata)

    def _check_metadata_exists(self) -> bool:
        """Check kernel-metadata.json exists."""
        exists = self.metadata_path.exists()
        self._log_check("Metadata file exists", exists)
        return exists

    def _check_notebook_exists(self, metadata: Dict) -> bool:
        """Check notebook file referenced in metadata exists."""
        code_file = metadata.get("code_file")
        notebook_path = self.notebook_dir / code_file

        exists = notebook_path.exists()
        self._log_check(f"Notebook '{code_file}' exists", exists)

        if not exists:
            # Other notebooks in the directory
            similar = list(self.notebook_dir.glob("*.ipynb"))
            if similar:
                print(f"    Found similar notebooks: {[n.name for n in similar]}")
                print(f"    Set metadata 'code_file' to the right notebook")

        return exists

    def _check_no_duplicates(self) -> bool:
        """Check only one notebook exists in directory."""
        notebooks = list(self.notebook_dir.glob("*.ipynb"))

        is_unique = len(notebooks) == 1
        self._log_check("Single notebook in directory", is_unique)

        if not is_unique:
            print(f"    Found {len(notebooks)} notebooks:")
            for nb in notebooks:
                print(f"      - {nb.name}")
            print(f"    Remove old versions")

        return is_unique

    def _check_kernel_id_exists(self, metadata: Dict) -> bool:
        """Check kernel ID exists on Kaggle."""
        kernel_id = metadata.get("id")

        try:
            result = subprocess.run(
                ["kaggle", "kernels", "list", "--mine"],
                capture_output=True,
                text=True,
                check=True
            )

            exists = kernel_id in result.stdout
            self._log_check(f"Kernel ID '{kernel_id}' exists on Kaggle", exists)

            if not exists:
                # Partial match
                username, slug = kernel_id.split("/")
                partial_matches = [
                    line.split()[0] for line in result.stdout.split('\n')
                    if slug[:15].lower() in line.lower()
                ]

                if partial_matches:
                    print(f"    Found similar kernels: {partial_matches}")
                    print(f"    Set metadata 'id' to the Kaggle kernel ID")

            return exists

        except subprocess.CalledProcessError as e:
            print(f"    ERROR: Failed to query Kaggle API: {e}")
            return False

    def _check_notebook_structure(self, notebook_path: Path) -> bool:
        """Validate notebook structure using nbformat."""
        try:
            with open(notebook_path) as f:
                nb = nbf.read(f, as_version=4)

            # Format
            nbf.validate(nb)

            # Basic structure
            has_cells = len(nb.cells) > 0
            has_metadata = "kernelspec" in nb.metadata

            is_valid = has_cells and has_metadata

            self._log_check("Notebook structure valid", is_valid)

            if is_valid:
                print(f"    Cells: {len(nb.cells)}")
                print(f"    Kernel: {nb.metadata.get('kernelspec', {}).get('display_name', 'Unknown')}")

            return is_valid

        except Exception as e:
            print(f"    ERROR: Notebook validation failed: {e}")
            return False

    def _check_expected_content(self, notebook_path: Path, metadata: Dict) -> bool:
        """Check for expected content based on notebook type."""

        # Markers per notebook type
        expected_markers = {
            "baseline": [
                "YOLO Label Conversion",  # Required for baseline
                "Multi-Seed Training",
                "Bootstrap"
            ],
            "eda": [
                "EDA Pipeline",
                "Visualizer",
                "generate_correlation_heatmap"
            ]
        }

        # Type from title or ID
        title = metadata.get("title", "").lower()
        kernel_id = metadata.get("id", "").lower()

        notebook_type = None
        if "baseline" in title or "baseline" in kernel_id:
            notebook_type = "baseline"
        elif "eda" in title or "eda" in kernel_id:
            notebook_type = "eda"

        if not notebook_type:
            # Unknown type
            self._log_check("Expected content check (skipped, unknown type)", True)
            return True

        # Markers
        with open(notebook_path) as f:
            nb = nbf.read(f, as_version=4)

        notebook_text = "\n".join(
            cell.source for cell in nb.cells
        )

        markers = expected_markers[notebook_type]
        found_markers = [m for m in markers if m in notebook_text]

        all_found = len(found_markers) == len(markers)

        self._log_check(f"Expected {notebook_type} content present", all_found)

        if not all_found:
            missing = set(markers) - set(found_markers)
            print(f"    Missing markers: {missing}")
            print(f"    Found markers: {found_markers}")

        return all_found

    def _log_check(self, check_name: str, passed: bool):
        """Log validation check result."""
        status = "PASS" if passed else "FAIL"
        self.validation_results.append((check_name, passed))
        print(f"\n{status} {check_name}")

    def _success_result(self, metadata: Dict) -> Dict:
        """Return success result."""
        print("\n" + "=" * 80)
        print("PASS: all validation checks passed")
        print("=" * 80)

        return {
            "valid": True,
            "notebook_dir": str(self.notebook_dir),
            "metadata": metadata,
            "checks_passed": len([r for r in self.validation_results if r[1]]),
            "checks_total": len(self.validation_results)
        }

    def _failure_result(self, reason: str) -> Dict:
        """Return failure result."""
        print("\n" + "=" * 80)
        print(f"FAIL: validation failed: {reason}")
        print("=" * 80)

        failed_checks = [r[0] for r in self.validation_results if not r[1]]

        return {
            "valid": False,
            "reason": reason,
            "failed_checks": failed_checks,
            "checks_passed": len([r for r in self.validation_results if r[1]]),
            "checks_total": len(self.validation_results)
        }


def deploy_notebook(notebook_dir: Path, dry_run: bool = False) -> Dict:
    """Deploy notebook to Kaggle after validation."""

    # Validate
    validator = NotebookDeploymentValidator(notebook_dir)
    validation_result = validator.validate()

    if not validation_result["valid"]:
        print("\nFAIL: Deployment aborted, validation failed")
        return validation_result

    if dry_run:
        print("\nDRY RUN - Deployment would proceed with:")
        print(f"   Kernel ID: {validation_result['metadata']['id']}")
        print(f"   Notebook: {validation_result['metadata']['code_file']}")
        return {"dry_run": True, **validation_result}

    # Push
    print("\nDeploying to Kaggle...")

    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(notebook_dir)],
            capture_output=True,
            text=True,
            check=True
        )

        print(result.stdout)

        # Version line
        version_line = [l for l in result.stdout.split('\n') if 'version' in l.lower()]
        version_info = version_line[0] if version_line else "Version unknown"

        print("\n" + "=" * 80)
        print("PASS: DEPLOYMENT SUCCESSFUL")
        print("=" * 80)
        print(f"\n{version_info}")
        print(f"\nKernel: {validation_result['metadata']['id']}")
        print(f"Check progress at: https://www.kaggle.com/code/{validation_result['metadata']['id']}")

        return {
            "deployed": True,
            "version_info": version_info,
            **validation_result
        }

    except subprocess.CalledProcessError as e:
        print(f"\nFAIL: Deployment failed: {e}")
        print(f"Error output: {e.stderr}")

        return {
            "deployed": False,
            "error": str(e),
            **validation_result
        }


def main():
    """CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Deploy a notebook to Kaggle after validation"
    )
    parser.add_argument(
        "notebook_dir",
        type=Path,
        help="Path to notebook directory containing .ipynb and kernel-metadata.json"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without deploying"
    )

    args = parser.parse_args()

    # Directory
    if not args.notebook_dir.exists():
        print(f"FAIL: Error: Directory not found: {args.notebook_dir}")
        sys.exit(1)

    # Deploy
    result = deploy_notebook(args.notebook_dir, dry_run=args.dry_run)

    # Exit code
    if result.get("valid") and (result.get("deployed") or result.get("dry_run")):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
