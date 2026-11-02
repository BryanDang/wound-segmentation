"""
Kaggle kernel metadata check, run before deployment.
"""

import json
import subprocess
import sys
from pathlib import Path


class MetadataValidator:
    """Checks kernel metadata before a Kaggle push."""

    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.metadata = self._load_metadata()
        self.notebook_path = metadata_path.parent / self.metadata['code_file']
        self.errors = []
        self.warnings = []

    def _load_metadata(self):
        """Load kernel metadata JSON."""
        with open(self.metadata_path) as f:
            return json.load(f)

    def validate(self):
        """Run all validation checks."""
        print(f"Validating: {self.metadata_path}")
        print("="*60)

        self.check_kernel_id()
        self.check_internet_requirements()
        self.check_gpu_requirements()
        self.check_dataset_sources()
        self.check_notebook_exists()

        self._print_results()

        return len(self.errors) == 0

    def check_kernel_id(self):
        """Check kernel ID format and existence."""
        kernel_id = self.metadata.get('id', '')

        # Format
        if not kernel_id or '/' not in kernel_id:
            self.errors.append(f"Invalid kernel ID format: '{kernel_id}'")
            return

        username, slug = kernel_id.split('/', 1)

        # Existence on Kaggle
        try:
            result = subprocess.run(
                ['kaggle', 'kernels', 'list', '--mine'],
                capture_output=True,
                text=True,
                check=True
            )

            existing_kernels = [line.split()[0] for line in result.stdout.split('\n') if line.strip()]

            if kernel_id in existing_kernels:
                print(f"Kernel ID exists: {kernel_id}")
            else:
                self.warnings.append(
                    f"Kernel ID not found: {kernel_id}\n"
                    f"   This creates a new kernel"
                )

        except subprocess.CalledProcessError as e:
            self.warnings.append(f"Could not verify kernel existence: {e}")

    def check_internet_requirements(self):
        """Flag pip install in the notebook when internet is disabled."""
        enable_internet = self.metadata.get('enable_internet', 'false')

        # pip install commands
        with open(self.notebook_path) as f:
            notebook_content = f.read()

        needs_internet = any([
            'pip install' in notebook_content,
            'pip3 install' in notebook_content,
            'subprocess.run(["pip"' in notebook_content,
        ])

        if needs_internet and enable_internet == 'false':
            self.errors.append(
                f"CRITICAL: Notebook contains 'pip install' but enable_internet=false\n"
                f"   Set enable_internet='true' in the metadata"
            )
        elif needs_internet:
            print(f"Internet enabled (pip install present)")
        else:
            print(f"No pip install detected, internet: {enable_internet}")

    def check_gpu_requirements(self):
        """Check GPU settings."""
        enable_gpu = self.metadata.get('enable_gpu', 'false')

        # YOLO and other GPU use
        with open(self.notebook_path) as f:
            notebook_content = f.read()

        likely_needs_gpu = any([
            'YOLO(' in notebook_content,
            'model.predict(' in notebook_content,
            'torch.cuda' in notebook_content,
        ])

        if likely_needs_gpu and enable_gpu == 'false':
            self.warnings.append(
                f"Notebook likely needs GPU but enable_gpu=false\n"
                f"   Set enable_gpu='true'"
            )
        elif likely_needs_gpu:
            print(f"GPU enabled (detected YOLO/model usage)")
        else:
            print(f"GPU: {enable_gpu}")

    def check_dataset_sources(self):
        """List dataset sources."""
        dataset_sources = self.metadata.get('dataset_sources', [])

        if not dataset_sources:
            self.warnings.append("No dataset sources specified")
            return

        print(f"\nDataset Sources ({len(dataset_sources)}):")
        for ds in dataset_sources:
            print(f"   - {ds}")

        print(f"{len(dataset_sources)} dataset(s) configured")

    def check_notebook_exists(self):
        """Check notebook file exists."""
        if not self.notebook_path.exists():
            self.errors.append(f"Notebook file not found: {self.notebook_path}")
        else:
            print(f"Notebook exists: {self.notebook_path.name}")

    def _print_results(self):
        """Print validation results."""
        print("\n" + "="*60)
        print("VALIDATION RESULTS")
        print("="*60)

        if self.warnings:
            print(f"\nWARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   {warning}")

        if self.errors:
            print(f"\nFAIL: ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   {error}")
            print(f"\nFAIL: VALIDATION FAILED")
        else:
            print(f"\nVALIDATION PASSED")


def main():
    """Run validation on specified metadata file."""
    if len(sys.argv) < 2:
        print("Usage: python validate_kernel_metadata.py <path/to/kernel-metadata.json>")
        sys.exit(1)

    metadata_path = Path(sys.argv[1])

    if not metadata_path.exists():
        print(f"Error: File not found: {metadata_path}")
        sys.exit(1)

    validator = MetadataValidator(metadata_path)
    success = validator.validate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
