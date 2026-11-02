"""
Kaggle kernel deployment with kernel ID validation.

Avoids 409 Conflict errors:
1. Query Kaggle for the actual kernel ID.
2. Fix the metadata ID on mismatch.
3. Push with the verified ID.

Usage:
    python scripts/safe_deploy_kernel.py notebooks/02_baseline_smoke
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional


def get_existing_kernel_id(kernel_search: str) -> Optional[str]:
    """
    Find the actual kernel ID on Kaggle.

    Args:
        kernel_search: Search term (e.g., "smoke-test", "baseline")

    Returns:
        Actual kernel ID from Kaggle, or None if not found
    """
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "list", "--mine"],
            capture_output=True,
            text=True,
            check=True
        )

        # Case-insensitive partial match
        search_lower = kernel_search.lower()
        for line in result.stdout.split('\n'):
            if search_lower in line.lower():
                # ID is the first column
                kernel_id = line.split()[0]
                return kernel_id

        return None

    except subprocess.CalledProcessError as e:
        print(f"FAIL: Error querying Kaggle: {e}")
        return None


def validate_and_fix_metadata(metadata_path: Path) -> dict:
    """
    Check the metadata ID against Kaggle and rewrite it on mismatch.

    Args:
        metadata_path: Path to kernel-metadata.json

    Returns:
        {
            "valid": bool,
            "fixed": bool,
            "metadata_id": str,
            "actual_id": str or None,
            "error": str or None
        }
    """
    # Load
    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata_id = metadata.get("id")
    if not metadata_id:
        return {
            "valid": False,
            "fixed": False,
            "metadata_id": None,
            "actual_id": None,
            "error": "No 'id' field in metadata"
        }

    # Slug for searching
    try:
        username, slug = metadata_id.split("/")
    except ValueError:
        return {
            "valid": False,
            "fixed": False,
            "metadata_id": metadata_id,
            "actual_id": None,
            "error": f"Invalid ID format: {metadata_id} (expected 'username/slug')"
        }

    # Kaggle kernel list
    result = subprocess.run(
        ["kaggle", "kernels", "list", "--mine"],
        capture_output=True,
        text=True
    )

    # Exact match
    if metadata_id in result.stdout:
        return {
            "valid": True,
            "fixed": False,
            "metadata_id": metadata_id,
            "actual_id": metadata_id,
            "error": None
        }

    # Partial slug match
    actual_id = None
    for line in result.stdout.split('\n'):
        # First 15 chars of the slug
        if slug[:15].lower() in line.lower():
            actual_id = line.split()[0]
            break

    if not actual_id:
        return {
            "valid": False,
            "fixed": False,
            "metadata_id": metadata_id,
            "actual_id": None,
            "error": f"Kernel matching '{slug}' not found on Kaggle"
        }

    # Mismatch: rewrite metadata
    print(f"\nWARNING: Kernel ID mismatch.")
    print(f"   Metadata has: {metadata_id}")
    print(f"   Kaggle has:   {actual_id}")
    print(f"   Rewriting metadata.")

    metadata["id"] = actual_id

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"PASS: Updated metadata to '{actual_id}'")

    return {
        "valid": True,
        "fixed": True,
        "metadata_id": metadata_id,
        "actual_id": actual_id,
        "error": None
    }


def safe_deploy_kernel(notebook_dir: Path) -> dict:
    """
    Push a kernel after validating and fixing its ID.

    Args:
        notebook_dir: Path to notebook directory with kernel-metadata.json

    Returns:
        {
            "success": bool,
            "deployed_id": str or None,
            "message": str
        }
    """
    metadata_path = notebook_dir / "kernel-metadata.json"

    if not metadata_path.exists():
        return {
            "success": False,
            "deployed_id": None,
            "message": f"kernel-metadata.json not found in {notebook_dir}"
        }

    print(f"=" * 60)
    print(f"SAFE KERNEL DEPLOYMENT")
    print(f"=" * 60)
    print(f"\nNotebook directory: {notebook_dir}")
    print(f"Metadata file: {metadata_path}")

    # Validate and fix metadata
    print(f"\n1. Validating kernel ID...")
    validation = validate_and_fix_metadata(metadata_path)

    if not validation["valid"]:
        return {
            "success": False,
            "deployed_id": None,
            "message": f"Validation failed: {validation['error']}"
        }

    deployed_id = validation["actual_id"]

    if validation["fixed"]:
        print(f"\n2. Metadata fixed")
    else:
        print(f"\n2. Metadata valid")

    # Push
    print(f"\n3. Deploying to Kaggle...")
    print(f"   Kernel ID: {deployed_id}")

    try:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(notebook_dir)],
            capture_output=True,
            text=True,
            check=True
        )

        print(f"\nPASS: Deployment successful.")
        print(f"\n{result.stdout}")

        # Version line
        if "version" in result.stdout.lower():
            version_line = [line for line in result.stdout.split('\n') if 'version' in line.lower()]
            if version_line:
                print(f"\n{version_line[0]}")

        return {
            "success": True,
            "deployed_id": deployed_id,
            "message": "Deployment successful"
        }

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr or e.stdout or str(e)
        print(f"\nFAIL: Deployment failed.")
        print(f"\nError: {error_msg}")

        return {
            "success": False,
            "deployed_id": deployed_id,
            "message": f"Deployment failed: {error_msg}"
        }


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python scripts/safe_deploy_kernel.py <notebook_directory>")
        print("\nExample:")
        print("  python scripts/safe_deploy_kernel.py notebooks/02_baseline_smoke")
        sys.exit(1)

    notebook_dir = Path(sys.argv[1])

    if not notebook_dir.exists():
        print(f"FAIL: Error: Directory not found: {notebook_dir}")
        sys.exit(1)

    if not notebook_dir.is_dir():
        print(f"FAIL: Error: Not a directory: {notebook_dir}")
        sys.exit(1)

    result = safe_deploy_kernel(notebook_dir)

    print(f"\n" + "=" * 60)
    if result["success"]:
        print(f"PASS: DEPLOYMENT COMPLETE")
        print(f"  Kernel ID: {result['deployed_id']}")
        print(f"  Check progress at: https://www.kaggle.com/code/{result['deployed_id']}")
    else:
        print(f"FAIL: DEPLOYMENT FAILED")
        print(f"  Error: {result['message']}")
        sys.exit(1)
    print(f"=" * 60)


if __name__ == "__main__":
    main()
