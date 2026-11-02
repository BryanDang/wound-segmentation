"""
Unit tests for Kaggle dataset creation functionality.

Tests check the contracts in the function docstrings.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.manage_kaggle_datasets import (
    create_kaggle_dataset,
    create_kaggle_dataset_metadata,
    generate_baseline_dataset_metadata,
    validate_kaggle_credentials,
)


class TestValidateKaggleCredentials:
    """Test Kaggle credentials validation."""

    def test_valid_credentials(self):
        """Should return valid=True when both env vars are set."""
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "test123"}):
            result = validate_kaggle_credentials()
            assert result["valid"] is True
            assert result["username"] == "testuser"
            assert "error" not in result

    def test_missing_username(self):
        """Should return valid=False when KAGGLE_USERNAME missing."""
        with patch.dict(os.environ, {"KAGGLE_KEY": "test123"}, clear=True):
            result = validate_kaggle_credentials()
            assert result["valid"] is False
            assert "KAGGLE_USERNAME" in result["error"]
            assert "username" not in result

    def test_missing_key(self):
        """Should return valid=False when KAGGLE_KEY missing."""
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "testuser"}, clear=True):
            result = validate_kaggle_credentials()
            assert result["valid"] is False
            assert "KAGGLE_KEY" in result["error"]

    def test_missing_both(self):
        """Should return valid=False when both env vars missing."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_kaggle_credentials()
            assert result["valid"] is False
            assert "KAGGLE_USERNAME" in result["error"] or "KAGGLE_KEY" in result["error"]

    def test_does_not_expose_api_key(self):
        """Should never include API key in output."""
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "secret123"}):
            result = validate_kaggle_credentials()
            result_str = str(result)
            assert "secret123" not in result_str
            assert "KAGGLE_KEY" not in result.get("username", "")


class TestCreateKaggleDatasetMetadata:
    """Test dataset metadata generation."""

    def test_minimal_valid_metadata(self, tmp_path):
        """Should create metadata with minimal required fields."""
        result = create_kaggle_dataset_metadata(
            title="Test Dataset",
            username="testuser",
            slug="test-dataset",
            output_folder=tmp_path,
        )

        assert result["success"] is True
        assert result["metadata"]["title"] == "Test Dataset"
        assert result["metadata"]["id"] == "testuser/test-dataset"
        assert result["metadata"]["licenses"] == [{"name": "CC0-1.0"}]
        assert result["metadata_path"] == tmp_path / "dataset-metadata.json"

    def test_metadata_file_created(self, tmp_path):
        """Should write valid JSON file to disk."""
        _ = create_kaggle_dataset_metadata(
            title="Test Dataset",
            username="testuser",
            slug="test-dataset",
            output_folder=tmp_path,
        )

        metadata_path = tmp_path / "dataset-metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["title"] == "Test Dataset"
        assert metadata["id"] == "testuser/test-dataset"

    def test_full_metadata_with_optional_fields(self, tmp_path):
        """Should include all optional fields when provided."""
        result = create_kaggle_dataset_metadata(
            title="Complete Dataset",
            username="testuser",
            slug="complete-dataset",
            output_folder=tmp_path,
            subtitle="A comprehensive test dataset",
            description="Detailed description with **markdown**",
            keywords=["test", "example", "dataset"],
            license_name="MIT",
            is_private=True,
        )

        metadata = result["metadata"]
        assert metadata["subtitle"] == "A comprehensive test dataset"
        assert metadata["description"] == "Detailed description with **markdown**"
        assert metadata["keywords"] == ["test", "example", "dataset"]
        assert metadata["licenses"] == [{"name": "MIT"}]
        assert metadata["isPrivate"] is True

    def test_title_too_short(self, tmp_path):
        """Should raise ValueError for title < 6 characters."""
        with pytest.raises(ValueError, match="title must be 6-50 characters"):
            create_kaggle_dataset_metadata(
                title="Test",  # Only 4 characters
                username="testuser",
                slug="test-dataset",
                output_folder=tmp_path,
            )

    def test_title_too_long(self, tmp_path):
        """Should raise ValueError for title > 50 characters."""
        with pytest.raises(ValueError, match="title must be 6-50 characters"):
            create_kaggle_dataset_metadata(
                title="A" * 51,
                username="testuser",
                slug="test-dataset",
                output_folder=tmp_path,
            )

    def test_slug_too_short(self, tmp_path):
        """Should raise ValueError for slug < 3 characters."""
        with pytest.raises(ValueError, match="slug must be 3-50 characters"):
            create_kaggle_dataset_metadata(
                title="Test Dataset",
                username="testuser",
                slug="ab",  # Only 2 characters
                output_folder=tmp_path,
            )

    def test_slug_invalid_characters(self, tmp_path):
        """Should raise ValueError for invalid slug characters."""
        with pytest.raises(
            ValueError, match="slug must contain only lowercase letters, numbers, and hyphens"
        ):
            create_kaggle_dataset_metadata(
                title="Test Dataset",
                username="testuser",
                slug="Test_Dataset",  # Uppercase and underscore
                output_folder=tmp_path,
            )

    def test_subtitle_too_short(self, tmp_path):
        """Should raise ValueError for subtitle < 20 characters if provided."""
        with pytest.raises(ValueError, match="subtitle must be 20-80 characters"):
            create_kaggle_dataset_metadata(
                title="Test Dataset",
                username="testuser",
                slug="test-dataset",
                output_folder=tmp_path,
                subtitle="Too short",  # Only 9 characters
            )

    def test_subtitle_too_long(self, tmp_path):
        """Should raise ValueError for subtitle > 80 characters."""
        with pytest.raises(ValueError, match="subtitle must be 20-80 characters"):
            create_kaggle_dataset_metadata(
                title="Test Dataset",
                username="testuser",
                slug="test-dataset",
                output_folder=tmp_path,
                subtitle="A" * 81,
            )

    def test_output_folder_does_not_exist(self):
        """Should raise FileNotFoundError if output_folder missing."""
        with pytest.raises(FileNotFoundError):
            create_kaggle_dataset_metadata(
                title="Test Dataset",
                username="testuser",
                slug="test-dataset",
                output_folder=Path("/nonexistent/folder"),
            )

    def test_creates_parent_directories(self, tmp_path):
        """Should create parent directories if they don't exist."""
        nested_path = tmp_path / "level1" / "level2" / "dataset"
        nested_path.mkdir(parents=True)

        result = create_kaggle_dataset_metadata(
            title="Nested Dataset",
            username="testuser",
            slug="nested-dataset",
            output_folder=nested_path,
        )

        assert result["success"] is True
        assert (nested_path / "dataset-metadata.json").exists()


class TestCreateKaggleDataset:
    """Test dataset creation/versioning using Kaggle API."""

    def test_new_dataset_creation_success(self, tmp_path):
        """Should execute kaggle datasets create command."""
        # Create metadata file
        metadata = {
            "title": "Test Dataset",
            "id": "testuser/test-dataset",
            "licenses": [{"name": "CC0-1.0"}],
        }
        metadata_path = tmp_path / "dataset-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        with patch.dict(os.environ, {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "test123"}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="Successfully created dataset", stderr=""
                )

                result = create_kaggle_dataset(metadata_path=metadata_path, new_dataset=True)

                assert result["success"] is True
                assert result["dataset_id"] == "testuser/test-dataset"
                assert "kaggle.com/datasets/testuser/test-dataset" in result["dataset_url"]
                assert result["version"] == 1
                assert "created" in result["message"].lower()

                # Verify correct CLI command
                mock_run.assert_called_once()
                call_args = mock_run.call_args[0][0]
                assert "kaggle" in call_args
                assert "datasets" in call_args
                assert "create" in call_args

    def test_dataset_versioning_success(self, tmp_path):
        """Should execute kaggle datasets version command."""
        metadata = {
            "title": "Test Dataset",
            "id": "testuser/test-dataset",
            "licenses": [{"name": "CC0-1.0"}],
        }
        metadata_path = tmp_path / "dataset-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        with patch.dict(os.environ, {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "test123"}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="Successfully versioned dataset", stderr=""
                )

                result = create_kaggle_dataset(
                    metadata_path=metadata_path,
                    new_dataset=False,
                    version_notes="Updated with new results",
                )

                assert result["success"] is True
                assert "version" in result["message"].lower()

                # Verify correct CLI command with version notes
                call_args = mock_run.call_args[0][0]
                assert "version" in call_args
                assert any("Updated with new results" in str(arg) for arg in call_args)

    def test_missing_metadata_file(self, tmp_path):
        """Should raise FileNotFoundError if metadata file missing."""
        nonexistent = tmp_path / "dataset-metadata.json"

        with pytest.raises(FileNotFoundError):
            create_kaggle_dataset(metadata_path=nonexistent, new_dataset=True)

    def test_missing_credentials(self, tmp_path):
        """Should raise EnvironmentError if credentials not set."""
        metadata_path = tmp_path / "dataset-metadata.json"
        metadata_path.write_text('{"title": "Test", "id": "user/test"}')

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="KAGGLE_USERNAME"):
                create_kaggle_dataset(metadata_path=metadata_path, new_dataset=True)

    def test_cli_command_failure(self, tmp_path):
        """Should raise RuntimeError if kaggle CLI command fails."""
        metadata = {"title": "Test Dataset", "id": "testuser/test-dataset"}
        metadata_path = tmp_path / "dataset-metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)

        with patch.dict(os.environ, {"KAGGLE_USERNAME": "testuser", "KAGGLE_KEY": "test123"}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stdout="", stderr="Error: Dataset already exists"
                )

                with pytest.raises(RuntimeError, match="Dataset already exists"):
                    create_kaggle_dataset(metadata_path=metadata_path, new_dataset=True)


class TestGenerateBaselineDatasetMetadata:
    """Test project-specific baseline dataset metadata generation."""

    def test_baseline_metadata_structure(self, tmp_path):
        """Should generate properly formatted baseline dataset metadata."""
        result = generate_baseline_dataset_metadata(
            username="testuser", output_folder=tmp_path, notebook_version="v1"
        )

        assert result["success"] is True
        metadata = result["metadata"]
        assert "Baseline" in metadata["title"]
        assert "wound-seg-baseline" in metadata["id"]
        assert "v1" in metadata["id"]
        assert len(metadata["keywords"]) > 0
        assert "wound" in metadata["keywords"]
        assert "segmentation" in metadata["keywords"]

    def test_baseline_includes_comprehensive_description(self, tmp_path):
        """Should include detailed description with methodology."""
        result = generate_baseline_dataset_metadata(username="testuser", output_folder=tmp_path)

        description = result["metadata"]["description"]
        assert "YOLOv8n-seg" in description
        assert "Dice" in description
        assert "zero-shot" in description or "baseline" in description
        assert len(description) > 100  # Should be comprehensive

    def test_baseline_version_increments(self, tmp_path):
        """Should handle different version identifiers."""
        result_v1 = generate_baseline_dataset_metadata(
            username="testuser", output_folder=tmp_path, notebook_version="v1"
        )
        result_v2 = generate_baseline_dataset_metadata(
            username="testuser", output_folder=tmp_path, notebook_version="v2"
        )

        assert "v1" in result_v1["metadata"]["id"]
        assert "v2" in result_v2["metadata"]["id"]
        assert result_v1["metadata"]["id"] != result_v2["metadata"]["id"]
