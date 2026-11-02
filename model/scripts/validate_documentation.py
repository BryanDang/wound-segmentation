#!/usr/bin/env python3
"""
Documentation corruption check for docs/ files.

Usage:
    python scripts/validate_documentation.py docs/NEW_FILE.md
    python scripts/validate_documentation.py --check-all
"""

import re
import sys
from pathlib import Path
from typing import Dict, List
from datetime import datetime


def validate_document(filepath: Path) -> Dict:
    """
    Check one documentation file for corruption.

    Returns:
        {
            "valid": bool,
            "filepath": str,
            "issues": List[str],
            "metrics": {
                "char_count": int,
                "space_ratio": float,
                "long_sequences": int,
                "max_sequence_length": int,
                "line_count": int,
                "long_lines": int
            }
        }
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    issues = []

    # Character count
    total_chars = len(content)

    if total_chars == 0:
        issues.append("File is empty")
        return {
            "valid": False,
            "filepath": str(filepath),
            "issues": issues,
            "metrics": {}
        }

    # Space ratio: at least 10%, or 7% for structured docs
    space_count = content.count(' ')
    space_ratio = space_count / total_chars

    # READMEs and deployment guides have tables, commands and paths, so lower ratios
    is_structured_doc = (
        'README' in str(filepath) or
        'DEPLOYMENT_GUIDE' in str(filepath)
    )

    min_ratio = 0.07 if is_structured_doc else 0.10

    if space_ratio < min_ratio:
        expected = "≥ 7%" if is_structured_doc else "≥ 10%"
        issues.append(f"Low space ratio: {space_ratio:.2%} (expected {expected})")

    # Long sequences: 25 or more consecutive letters, outside code
    # Drop fenced code blocks
    content_no_code = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    # Drop inline code
    content_no_code = re.sub(r'`[^`]+`', '', content_no_code)

    long_sequences = re.findall(r'[a-zA-Z]{25,}', content_no_code)
    max_seq_length = max(len(s) for s in long_sequences) if long_sequences else 0

    if long_sequences:
        issues.append(f"Found {len(long_sequences)} suspicious sequences (≥25 letters without spaces)")
        issues.append(f"Example: '{long_sequences[0][:50]}...'")

    # Line lengths
    lines = content.split('\n')
    line_count = len(lines)
    long_lines = [i for i, line in enumerate(lines) if len(line) > 500]

    if long_lines:
        issues.append(f"Found {len(long_lines)} very long lines (>500 chars)")
        issues.append(f"Line numbers: {long_lines[:5]}")

    # Severe corruption: lowercase runs between capitals.
    # "TGuidKdwork" matches, "YOLO" does not.
    severe_corruption = re.findall(r'[A-Z][a-z]{1,2}[A-Z][a-z]{1,2}[A-Z][a-z]{1,2}[A-Z]', content)

    # Flag above 20 instances
    if len(severe_corruption) > 20:
        issues.append(f"Severe corruption detected: {len(severe_corruption)} instances (e.g., '{severe_corruption[0]}')")

    # Lowercase words over 20 chars
    long_words = re.findall(r'\b[a-z]{20,}\b', content)
    if len(long_words) > 5:
        issues.append(f"Unusually long words without spaces: {len(long_words)} instances")

    return {
        "valid": len(issues) == 0,
        "filepath": str(filepath),
        "issues": issues,
        "metrics": {
            "char_count": total_chars,
            "space_ratio": space_ratio,
            "long_sequences": len(long_sequences),
            "max_sequence_length": max_seq_length,
            "line_count": line_count,
            "long_lines": len(long_lines)
        }
    }


def validate_all_docs() -> List[Dict]:
    """Check all markdown files in docs/."""
    docs_dir = Path("docs")

    if not docs_dir.exists():
        print(f"FAIL: docs/ directory not found")
        return []

    results = []
    for md_file in docs_dir.glob("**/*.md"):
        # Skip archive
        if "archive" in str(md_file):
            continue

        result = validate_document(md_file)
        results.append(result)

    return results


def generate_report(results: List[Dict]) -> str:
    """Build the text report."""
    valid_count = sum(1 for r in results if r["valid"])
    total_count = len(results)

    report = f"\n{'='*70}\n"
    report += f"Documentation Validation Report\n"
    report += f"{'='*70}\n"
    report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"Total files: {total_count}\n"
    report += f"Valid: {valid_count}\n"
    report += f"Corrupted: {total_count - valid_count}\n"

    if valid_count < total_count:
        report += f"\n{'='*70}\n"
        report += f"CORRUPTED FILES\n"
        report += f"{'='*70}\n"

        for r in results:
            if not r["valid"]:
                report += f"\n{r['filepath']}\n"
                report += f"   Metrics:\n"
                for key, value in r['metrics'].items():
                    report += f"     - {key}: {value}\n"
                report += f"   Issues:\n"
                for issue in r["issues"]:
                    report += f"     {issue}\n"

    else:
        report += f"\nAll documentation files are valid.\n"

    report += f"\n{'='*70}\n"

    return report


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python validate_documentation.py <filepath>       # Validate single file")
        print("  python validate_documentation.py --check-all      # Validate all docs/")
        print("  python validate_documentation.py --check-all --json  # JSON output")
        sys.exit(1)

    # All docs
    if sys.argv[1] == "--check-all":
        results = validate_all_docs()

        if not results:
            print("FAIL: No documentation files found")
            sys.exit(1)

        # JSON output
        if "--json" in sys.argv:
            import json
            print(json.dumps(results, indent=2))
            sys.exit(0 if all(r["valid"] for r in results) else 1)

        # Text report
        print(generate_report(results))

        # Exit 1 if any file is corrupted
        sys.exit(0 if all(r["valid"] for r in results) else 1)

    # Single file
    else:
        filepath = Path(sys.argv[1])

        if not filepath.exists():
            print(f"FAIL: File not found: {filepath}")
            sys.exit(1)

        result = validate_document(filepath)

        print(f"\n{'='*70}")
        print(f"Validation: {filepath}")
        print(f"{'='*70}")
        print(f"Status: {'VALID' if result['valid'] else 'CORRUPTED'}\n")

        print(f"Metrics:")
        for key, value in result['metrics'].items():
            if key == "space_ratio":
                print(f"  {key}: {value:.2%}")
            else:
                print(f"  {key}: {value:,}")

        if result["issues"]:
            print(f"\nIssues detected:")
            for issue in result["issues"]:
                print(f"  - {issue}")
            sys.exit(1)
        else:
            print(f"\nNo corruption detected.")
            sys.exit(0)


if __name__ == "__main__":
    main()
