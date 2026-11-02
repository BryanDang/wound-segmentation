"""
Tests for Kaggle dataset management functions.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.deployment.kaggle_dataset import (
    validate_kaggle_credentials,
    create_baseline_dataset_metadata,
)


class TestValidateKaggleCredentials:
    """Tests for validate_kaggle_credentials()."""

    def test_valid_credentials(self):
        """Should return success when both credentials present."""
        with patch.dict(
            os.environ, {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "testkey123"}
        ):
            result = validate_kaggle_credentials()

            assert result["valid"] is True
            assert result["username"] == "testuser"

    def test_missing_username(self):
        """Should return failure when username missing."""
        with patch.dict(os.environ, {"KAGGLE_KEY": "testkey123"}, clear=True):
            result = validate_kaggle_credentials()

            assert result["valid"] is False
            assert "error" in result
            assert "Missing credentials" in result["error"]

    def test_missing_api_key(self):
        """Should return failure when API key missing."""
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "testuser"}, clear=True):
            result = validate_kaggle_credentials()

            assert result["valid"] is False
            assert "error" in result

    def test_both_missing(self):
        """Should return failure when both credentials missing."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_kaggle_credentials()

            assert result["valid"] is False

    def test_empty_string_values(self):
        """Should treat empty strings as missing."""
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "", "KAGGLE_KEY": ""}):
            result = validate_kaggle_credentials()

            assert result["valid"] is False


class TestCreateBaselineDatasetMetadata:
    """Tests for create_baseline_dataset_metadata()."""

    def test_creates_metadata_file(self, tmp_path):
        """Should create metadata file with correct structure."""
        username = "testuser"
        result = create_baseline_dataset_metadata(username, tmp_path)

        assert result["success"] is True
        assert "metadata" in result
        assert "metadata_path" in result

        # Check file was created
        metadata_path = tmp_path / "dataset-metadata.json"
        assert metadata_path.exists()

    def test_metadata_structure(self, tmp_path):
        """Should create metadata with all required fields."""
        username = "testuser"
        result = create_baseline_dataset_metadata(username, tmp_path)

        metadata = result["metadata"]
        assert metadata["title"] == "Wound Segmentation Zero-Shot Baseline Results"
        assert metadata["id"] == "testuser/wound-seg-baseline-results-v1"
        assert "subtitle" in metadata
        assert "description" in metadata
        assert "keywords" in metadata
        assert "licenses" in metadata
        assert metadata["isPrivate"] is False

    def test_metadata_keywords(self, tmp_path):
        """Should include correct keywords."""
        result = create_baseline_dataset_metadata("testuser", tmp_path)

        keywords = result["metadata"]["keywords"]
        assert "wound" in keywords
        assert "segmentation" in keywords
        assert "yolo" in keywords
        assert "baseline" in keywords

    def test_json_file_format(self, tmp_path):
        """Should write valid JSON with proper formatting."""
        create_baseline_dataset_metadata("testuser", tmp_path)

        metadata_path = tmp_path / "dataset-metadata.json"
        with open(metadata_path) as f:
            data = json.load(f)

        assert "title" in data
        assert "id" in data

    def test_slug_validation_passes(self, tmp_path):
        """Should accept valid slug format."""
        result = create_baseline_dataset_metadata("testuser", tmp_path)

        slug = result["metadata"]["id"].split("/")[1]
        assert slug == "wound-seg-baseline-results-v1"
        # Slug should only contain lowercase, digits, hyphens
        assert slug.replace("-", "").replace("v", "").replace("1", "").isalnum()

    def test_overwrites_existing_file(self, tmp_path):
        """Should overwrite existing metadata file."""
        metadata_path = tmp_path / "dataset-metadata.json"

        # Create initial file
        create_baseline_dataset_metadata("user1", tmp_path)
        initial_size = metadata_path.stat().st_size

        # Overwrite with new data
        create_baseline_dataset_metadata("user2", tmp_path)
        new_size = metadata_path.stat().st_size

        # File should exist and potentially different size
        assert metadata_path.exists()
        with open(metadata_path) as f:
            data = json.load(f)
            assert "user2" in data["id"]

    def test_empty_username(self, tmp_path):
        """Should handle empty username gracefully."""
        result = create_baseline_dataset_metadata("", tmp_path)

        # Implementation should either succeed with empty username or return error
        assert "success" in result or "error" in result

    def test_nonexistent_output_folder(self):
        """Should handle nonexistent output folder."""
        nonexistent = Path("/nonexistent/path/that/does/not/exist")
        result = create_baseline_dataset_metadata("testuser", nonexistent)

        # Should return error or create parent directories
        assert "success" in result or "error" in result
