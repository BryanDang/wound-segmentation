import csv
import os
from typing import Any, Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from utils.io import failure, success

# --- Configuration ---
EDA_CONFIG = {
    "heatmap_size": (256, 256),
    "plot_bins": 50,
    "overlay_sample_size": 5,
    "overlay_color": [0, 0, 255],  # BGR for OpenCV
}


class SummaryReporter:
    """
    A class to generate a summary table from the image metadata CSV.
    """

    def __init__(self, metadata_csv_path: str):
        """
        Initializes the SummaryReporter.

        Args:
            metadata_csv_path: Path to the image_metadata.csv file.
        """
        self.metadata_csv_path = metadata_csv_path
        self.metadata_df: pd.DataFrame = pd.DataFrame()

        # Load the CSV during initialization
        try:
            if not os.path.exists(metadata_csv_path):
                raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv_path}")

            self.metadata_df = pd.read_csv(metadata_csv_path)

            if self.metadata_df.empty:
                raise ValueError(f"Metadata CSV is empty: {metadata_csv_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load metadata CSV: {str(e)}")

    def generate_summary_table(self) -> Dict[str, Any]:
        """
        Generates a summary table from the image metadata.

        Returns:
            Dict[str, Any]: A structured response containing the summary DataFrame
                            or an error message.
        """
        try:
            if self.metadata_df.empty:
                return failure("No metadata loaded")

            # Initialize summary statistics dictionary
            summary_stats = {}

            # Total Images Processed
            summary_stats["Total Images"] = len(self.metadata_df)

            # Images per Source
            if "source" in self.metadata_df.columns:
                source_counts = self.metadata_df["source"].value_counts()
                for source, count in source_counts.items():
                    summary_stats[f"Images (Source: {source})"] = count

            # Images per Set
            if "set" in self.metadata_df.columns:
                set_counts = self.metadata_df["set"].value_counts()
                for set_name, count in set_counts.items():
                    summary_stats[f"Images (Set: {set_name})"] = count

            # Images with No Wound (wound_pixel_count == 0)
            if "wound_pixel_count" in self.metadata_df.columns:
                no_wound_count = (self.metadata_df["wound_pixel_count"] == 0).sum()
                summary_stats["Images with No Wound"] = no_wound_count

            # Statistical measures for numerical columns
            numerical_columns = [
                "wound_pixel_count",
                "wound_percentage",
                "bounding_box_aspect_ratio",
            ]

            for col in numerical_columns:
                if col in self.metadata_df.columns:
                    col_data = self.metadata_df[col]

                    # Format column name for display
                    if col == "wound_pixel_count":
                        display_name = "Wound Pixel Count"
                    elif col == "wound_percentage":
                        display_name = "Wound Percentage"
                    elif col == "bounding_box_aspect_ratio":
                        display_name = "Aspect Ratio"
                    else:
                        display_name = col

                    # Calculate statistics
                    summary_stats[f"Avg {display_name}"] = col_data.mean()
                    summary_stats[f"Min {display_name}"] = col_data.min()
                    summary_stats[f"Max {display_name}"] = col_data.max()
                    summary_stats[f"Std {display_name}"] = col_data.std()

            # Create summary DataFrame
            summary_df = pd.DataFrame(list(summary_stats.items()), columns=["Statistic", "Value"])
            summary_df.set_index("Statistic", inplace=True)

            return success(summary_df)

        except Exception as e:
            return failure(f"Error generating summary table: {str(e)}")


class Analyzer:
    def __init__(self, source_map_path: str):
        """Initializes the Analyzer and loads the image source map."""
        self.source_map: Dict[str, str] = {}
        self.init_errors: List[str] = []
        self.init_warnings: List[str] = []

        try:
            if os.path.exists(source_map_path):
                # Load the Excel file using pandas
                df = pd.read_excel(source_map_path, engine="openpyxl")
                # Create mapping from image_id to source
                self.source_map = dict(zip(df["image_id"], df["source"]))
            else:
                warning_msg = f"Source map file not found at {source_map_path}"
                self.init_warnings.append(warning_msg)
        except Exception as e:
            error_msg = f"Error loading source map: {e}"
            self.init_errors.append(error_msg)

    def get_initialization_status(self) -> Dict[str, Any]:
        """Returns the initialization status with any errors or warnings."""
        has_errors = len(self.init_errors) > 0
        return {
            "success": not has_errors,
            "data": {
                "source_map_loaded": len(self.source_map) > 0,
                "source_count": len(self.source_map),
            },
            "errors": self.init_errors,
            "warnings": self.init_warnings,
        }

    def analyze_image(self, image_path: str, mask_path: str) -> Dict[str, Any]:
        """
        Analyzes a single image and returns a dictionary of its metadata, including
        color and positional metrics for the wound region.

        **Contract:**

        Args:
            image_path (str): Path to the image file.
            mask_path (str): Path to the corresponding mask file.

        Returns:
            Dict[str, Any]: A structured dictionary containing the calculated metrics.
                            On success, the `data` key holds a dictionary with fields:
                            - image_id, source, width, height
                            - wound_pixel_count, wound_percentage, bounding_box_aspect_ratio
                            - mean_intensity, mean_r, mean_g, mean_b
                            - centroid_x, centroid_y
        """
        try:
            # Get image filename for source lookup
            image_filename = os.path.basename(image_path)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                return failure(f"Could not read image: {image_path}")

            height, width, _ = image.shape

            # Initialize wound metrics
            wound_pixel_count = 0
            wound_percentage = 0.0
            aspect_ratio = 0.0
            mean_intensity = 0.0
            mean_r = 0.0
            mean_g = 0.0
            mean_b = 0.0
            centroid_x = 0.0
            centroid_y = 0.0

            # Load and analyze mask if it exists
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    wound_pixel_count = np.count_nonzero(mask)
                    wound_percentage = (
                        (wound_pixel_count / (width * height)) * 100 if (width * height) > 0 else 0
                    )

                    # --- Calculations for images with wounds ---
                    if wound_pixel_count > 0:
                        # Bounding Box Aspect Ratio
                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if contours:
                            all_points = np.concatenate(contours)
                            x, y, w, h = cv2.boundingRect(all_points)
                            aspect_ratio = w / h if h > 0 else 0

                        # Color & Brightness Metrics
                        masked_pixels = image[mask > 0]
                        if len(masked_pixels) > 0:
                            # Extract BGR channels from masked pixels (shape: N x 3)
                            mean_b = float(np.mean(masked_pixels[:, 0]))
                            mean_g = float(np.mean(masked_pixels[:, 1]))
                            mean_r = float(np.mean(masked_pixels[:, 2]))
                            mean_intensity = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b

                        # Centroid Metrics
                        M = cv2.moments(mask)
                        if M["m00"] != 0:
                            centroid_x = M["m10"] / M["m00"]
                            centroid_y = M["m01"] / M["m00"]

            # Get source from mapping or default to unknown
            source = self.source_map.get(image_filename, "unknown")

            # Create metadata dictionary
            metadata = {
                "image_id": image_filename,
                "source": source,
                "width": width,
                "height": height,
                "wound_pixel_count": wound_pixel_count,
                "wound_percentage": round(wound_percentage, 4),
                "bounding_box_aspect_ratio": round(aspect_ratio, 4),
                "mean_intensity": round(mean_intensity, 2),
                "mean_r": round(mean_r, 2),
                "mean_g": round(mean_g, 2),
                "mean_b": round(mean_b, 2),
                "centroid_x": round(centroid_x, 2),
                "centroid_y": round(centroid_y, 2),
            }

            return success(metadata)

        except Exception as e:
            return failure(f"Error analyzing image {image_path}: {str(e)}")


class Reporter:
    def __init__(self, output_path: str, fieldnames: List[str]):
        """Initializes the CSV reporter and opens the file for writing."""
        self.output_path = output_path
        self.fieldnames = fieldnames
        self.writer = None
        self.file_handle = None

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Open file and create CSV writer
            self.file_handle = open(output_path, "w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file_handle, fieldnames=fieldnames)
            self.writer.writeheader()

        except Exception as e:
            if self.file_handle:
                self.file_handle.close()
            raise RuntimeError(f"Failed to initialize CSV reporter: {e}")

    def write_row(self, row_data: Dict[str, Any]) -> None:
        """Writes a single row to the CSV file."""
        try:
            if self.writer is None:
                raise RuntimeError("CSV writer not initialized")
            self.writer.writerow(row_data)
            self.file_handle.flush()  # Ensure data is written immediately
        except Exception as e:
            raise RuntimeError(f"Failed to write row: {e}")

    def close_report(self) -> None:
        """Closes the CSV file handle."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            self.writer = None


class Visualizer:
    def __init__(self, config: Dict[str, Any]):
        """Initializes the Visualizer with configuration."""
        self.config = config

    def generate_all_plots(
        self, metadata_csv_path: str, image_data_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """
        Generates all visualizations from the completed metadata CSV.
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Read the metadata CSV
            df = pd.read_csv(metadata_csv_path)

            if df.empty:
                return success(
                    {
                        "message": "No data in metadata CSV, skipped visualizations",
                        "visualizations_generated": 0,
                    }
                )

            generated_files = []

            # --- Add derived columns for visualizations ---
            # Add size_category column for stratification chart
            def categorize_size(wound_pct):
                if wound_pct == 0.0:
                    return "None"
                elif 0.1 <= wound_pct < 2.0:
                    return "Small"
                elif 2.0 <= wound_pct < 10.0:
                    return "Medium"
                else:  # >= 10.0
                    return "Large"

            df["size_category"] = df["wound_percentage"].apply(categorize_size)

            # --- Generate Core Visualizations ---

            for plot_func in [
                self._generate_wound_size_distribution,
                self._generate_aspect_ratio_distribution,
                self._generate_train_test_split,
                self._generate_sample_overlays,
                self._generate_wound_heatmap,
                self._generate_class_imbalance_chart,
                self.generate_color_analysis,
                self.generate_size_stratification_chart,
                self.generate_correlation_heatmap,
                self.generate_no_wound_examples,
                self.generate_positional_heatmap,
                self.generate_extrema_examples,
            ]:
                # Note: some functions have different signatures
                if plot_func in [
                    self._generate_sample_overlays,
                    self._generate_wound_heatmap,
                    self.generate_no_wound_examples,
                    self.generate_extrema_examples,
                ]:
                    result = plot_func(df, image_data_path, output_dir)
                else:
                    result = plot_func(df, output_dir)

                if result["success"] and result.get("data", {}).get("files"):
                    generated_files.extend(result["data"]["files"])

            return success(
                {
                    "message": "All visualizations generated successfully",
                    "visualizations_generated": len(generated_files),
                    "files": generated_files,
                }
            )

        except Exception as e:
            return failure(f"Error generating visualizations: {e}")

    def _generate_class_imbalance_chart(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Generates a pie chart representing class imbalance.

        **Contract:**

        Args:
            df (pd.DataFrame): A DataFrame containing the necessary columns:
                               `width`, `height`, and `wound_pixel_count`.
            output_dir (str): The directory where the output PNG file will be saved.

        Returns:
            Dict[str, Any]: A structured dictionary indicating success or failure.
                            On success, `data['files']` contains the path to the generated chart.

        Edge Cases:
            - Handles cases where the total number of pixels is zero to avoid division errors.
            - Handles cases where the input DataFrame is missing required columns.
        """
        try:
            if "wound_percentage" not in df.columns:
                return failure("wound_percentage column not found in metadata")

            # Calculate overall wound and background percentages
            total_pixels = (df["width"] * df["height"]).sum()
            total_wound_pixels = (df["wound_pixel_count"]).sum()

            if total_pixels == 0:
                return failure("Total pixels is zero, cannot calculate imbalance")

            overall_wound_percentage = (total_wound_pixels / total_pixels) * 100
            background_percentage = 100 - overall_wound_percentage

            # Data for plotting
            labels = ["Background", "Wound"]
            sizes = [background_percentage, overall_wound_percentage]
            colors = ["#4c72b0", "#c44e52"]  # Seaborn default blue and red
            explode = (0, 0.1)  # Explode the 'Wound' slice

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(
                sizes,
                explode=explode,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                shadow=True,
                startangle=90,
                textprops={"fontsize": 12, "fontweight": "bold"},
            )
            ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

            plt.title(
                "Pixel Class Imbalance Across Entire Dataset",
                fontsize=16,
                fontweight="bold",
                pad=20,
            )

            file_path = os.path.join(output_dir, "class_imbalance.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()

            return success({"files": [file_path]})
        except Exception as e:
            return failure(f"Error generating class imbalance chart: {e}")

    def _generate_wound_size_distribution(
        self, df: pd.DataFrame, output_dir: str
    ) -> Dict[str, Any]:
        """Generate wound size distribution histogram."""
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(df["wound_pixel_count"], bins=self.config["plot_bins"], kde=True)
            plt.title("Distribution of Wound Sizes (in pixels)")
            plt.xlabel("Wound Pixel Count")
            plt.ylabel("Frequency")
            file_path = os.path.join(output_dir, "wound_size_distribution.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()
            return success({"files": [file_path]})
        except Exception as e:
            return failure(f"Error generating wound size distribution: {e}")

    def _generate_aspect_ratio_distribution(
        self, df: pd.DataFrame, output_dir: str
    ) -> Dict[str, Any]:
        """Generate aspect ratio distribution histogram."""
        try:
            wound_data = df[df["bounding_box_aspect_ratio"] > 0]
            if not wound_data.empty:
                plt.figure(figsize=(10, 6))
                sns.histplot(
                    wound_data["bounding_box_aspect_ratio"],
                    bins=self.config["plot_bins"],
                    kde=True,
                )
                plt.title("Distribution of Wound Bounding Box Aspect Ratios")
                plt.xlabel("Aspect Ratio (Width / Height)")
                plt.ylabel("Frequency")
                file_path = os.path.join(output_dir, "aspect_ratio_distribution.png")
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()
                return success({"files": [file_path]})
            else:
                return success(
                    {
                        "files": [],
                        "message": "No wound data for aspect ratio distribution",
                    }
                )
        except Exception as e:
            return failure(f"Error generating aspect ratio distribution: {e}")

    def _generate_train_test_split(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """Generate train/test split bar chart."""
        try:
            plt.figure(figsize=(8, 5))
            if "set" in df.columns:
                sns.countplot(x="set", data=df)
            else:
                # Infer set from image paths if 'set' column doesn't exist
                df["inferred_set"] = df["image_id"].apply(
                    lambda x: "train" if "train" in x else "test"
                )
                sns.countplot(x="inferred_set", data=df)
            plt.title("Image Count by Set (Train/Test)")
            plt.xlabel("Set")
            plt.ylabel("Count")
            file_path = os.path.join(output_dir, "train_test_split.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()
            return success({"files": [file_path]})
        except Exception as e:
            return failure(f"Error generating train/test split chart: {e}")

    def _generate_sample_overlays(
        self, df: pd.DataFrame, image_data_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """Generate sample mask overlays."""
        try:
            sample_size = min(self.config["overlay_sample_size"], len(df))
            if sample_size == 0:
                return success(
                    {
                        "files": [],
                        "message": "No images available for overlay generation",
                    }
                )

            sample_df = df.sample(n=sample_size) if len(df) > sample_size else df
            generated_files = []
            errors = []

            for _, row in sample_df.iterrows():
                try:
                    # Find image in train or test directory
                    image_path = None
                    mask_path = None

                    for split in ["train", "test"]:
                        potential_image_path = os.path.join(
                            image_data_path, f"{split}_images", row["image_id"]
                        )
                        potential_mask_path = os.path.join(
                            image_data_path, f"{split}_masks", row["image_id"]
                        )
                        if os.path.exists(potential_image_path):
                            image_path = potential_image_path
                            mask_path = potential_mask_path
                            break

                    if image_path is None:
                        continue

                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if mask is not None:
                            # Create colored mask overlay
                            colored_mask = np.zeros_like(image)
                            colored_mask[mask > 0] = self.config["overlay_color"]
                            overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

                            output_path = os.path.join(output_dir, f"overlay_{row['image_id']}")
                            cv2.imwrite(output_path, overlay)
                            generated_files.append(output_path)

                except Exception as e:
                    errors.append(f"Error creating overlay for {row['image_id']}: {e}")

            return success(
                {
                    "files": generated_files,
                    "overlays_generated": len(generated_files),
                    "errors": errors,
                }
            )

        except Exception as e:
            return failure(f"Error generating sample overlays: {e}")

    def _generate_wound_heatmap(
        self, df: pd.DataFrame, image_data_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """Generate wound location heatmap."""
        try:
            heatmap = np.zeros(self.config["heatmap_size"], dtype=np.float32)

            wound_images = df[df["wound_pixel_count"] > 0]

            if wound_images.empty:
                return success(
                    {
                        "files": [],
                        "message": "No wound images found for heatmap generation",
                    }
                )

            for _, row in wound_images.iterrows():
                # Find mask in train or test directory
                mask_path = None
                for split in ["train", "test"]:
                    potential_mask_path = os.path.join(
                        image_data_path, f"{split}_masks", row["image_id"]
                    )
                    if os.path.exists(potential_mask_path):
                        mask_path = potential_mask_path
                        break

                if mask_path is None:
                    continue

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    resized_mask = cv2.resize(mask, self.config["heatmap_size"])
                    heatmap += resized_mask / 255.0

            if np.max(heatmap) > 0:
                heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
                colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                file_path = os.path.join(output_dir, "wound_location_heatmap.png")
                cv2.imwrite(file_path, colored_heatmap)
                return success({"files": [file_path]})
            else:
                return success({"files": [], "message": "No valid wound data found for heatmap"})

        except Exception as e:
            return failure(f"Error generating wound heatmap: {e}")

    def generate_color_analysis(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Generates histograms for wound color and brightness.

        **Contract:**

        Args:
            df (pd.DataFrame): A DataFrame containing the necessary columns:
                               `mean_intensity`, `mean_r`, `mean_g`, `mean_b`,
                               and `wound_pixel_count`.
            output_dir (str): The directory where the output PNG files will be saved.

        Returns:
            Dict[str, Any]: A structured dictionary indicating success or failure.
                            On success, `data['files']` contains the paths to the
                            two generated charts: `brightness_distribution.png` and
                            `color_channel_distribution.png`.

        Edge Cases:
            - Handles cases where the input DataFrame has no rows with wound pixels.
        """
        try:
            # Check for required columns
            required_columns = ["mean_intensity", "mean_r", "mean_g", "mean_b", "wound_pixel_count"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return failure(f"Missing required columns: {', '.join(missing_columns)}")

            # Filter to only images with wounds
            wound_data = df[df["wound_pixel_count"] > 0]

            if wound_data.empty:
                return success(
                    {"files": [], "message": "No wound data available for color analysis"}
                )

            generated_files = []

            # Generate brightness distribution histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(
                wound_data["mean_intensity"],
                bins=self.config["plot_bins"],
                kde=True,
                color="#4c72b0",
            )
            plt.title(
                "Distribution of Wound Brightness (Mean Intensity)", fontsize=14, fontweight="bold"
            )
            plt.xlabel("Mean Intensity", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            brightness_file = os.path.join(output_dir, "brightness_distribution.png")
            plt.savefig(brightness_file, dpi=300, bbox_inches="tight")
            plt.close()
            generated_files.append(brightness_file)

            # Generate color channel distribution histogram
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot each color channel
            sns.histplot(
                wound_data["mean_r"],
                bins=self.config["plot_bins"],
                kde=True,
                color="red",
                alpha=0.5,
                label="Red Channel",
                ax=ax,
            )
            sns.histplot(
                wound_data["mean_g"],
                bins=self.config["plot_bins"],
                kde=True,
                color="green",
                alpha=0.5,
                label="Green Channel",
                ax=ax,
            )
            sns.histplot(
                wound_data["mean_b"],
                bins=self.config["plot_bins"],
                kde=True,
                color="blue",
                alpha=0.5,
                label="Blue Channel",
                ax=ax,
            )

            ax.set_title("Distribution of Wound RGB Color Channels", fontsize=14, fontweight="bold")
            ax.set_xlabel("Channel Intensity", fontsize=12)
            ax.set_ylabel("Frequency", fontsize=12)
            ax.legend(fontsize=10)

            color_file = os.path.join(output_dir, "color_channel_distribution.png")
            plt.savefig(color_file, dpi=300, bbox_inches="tight")
            plt.close()
            generated_files.append(color_file)

            return success({"files": generated_files})

        except Exception as e:
            return failure(f"Error generating color analysis: {e}")

    def generate_size_stratification_chart(
        self, df: pd.DataFrame, output_dir: str
    ) -> Dict[str, Any]:
        """
        Generates a bar chart showing the distribution of wound size categories.
        This requires the 'size_category' column to be added to the DataFrame first.

        **Contract:**

        Args:
            df (pd.DataFrame): The metadata DataFrame, must include 'size_category'.
            output_dir (str): Directory to save the output PNG.

        Returns:
            Dict[str, Any]: Success/failure dict with the path to the generated file.
        """
        try:
            # Check for required column
            if "size_category" not in df.columns:
                return failure("Missing required column: size_category")

            # Define category order for consistent display
            category_order = ["None", "Small", "Medium", "Large"]

            # Create figure
            plt.figure(figsize=(10, 6))

            # Generate bar chart using countplot
            sns.countplot(x="size_category", data=df, order=category_order, color="#4c72b0")

            # Styling
            plt.title("Distribution of Wound Size Categories", fontsize=14, fontweight="bold")
            plt.xlabel("Size Category", fontsize=12)
            plt.ylabel("Count", fontsize=12)

            # Save figure
            file_path = os.path.join(output_dir, "size_stratification.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()

            return success({"files": [file_path]})

        except Exception as e:
            return failure(f"Error generating size stratification chart: {e}")

    def generate_correlation_heatmap(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Generates a heatmap of the correlation matrix for numerical metrics.

        **Contract:**

        Args:
            df (pd.DataFrame): The metadata DataFrame with all numerical columns.
            output_dir (str): Directory to save the output PNG.

        Returns:
            Dict[str, Any]: Success/failure dict with the path to the generated file.
        """
        try:
            # Define numerical metrics to include in correlation analysis
            numerical_columns = [
                "wound_pixel_count",
                "wound_percentage",
                "bounding_box_aspect_ratio",
                "mean_intensity",
                "mean_r",
                "mean_g",
                "mean_b",
                "centroid_x",
                "centroid_y",
            ]

            # Filter to only columns that exist in the dataframe
            available_columns = [col for col in numerical_columns if col in df.columns]

            # Check for at least 2 numerical columns
            if len(available_columns) < 2:
                return success(
                    {
                        "files": [],
                        "message": "Not enough numerical columns for correlation analysis",
                    }
                )

            # Filter to only wound images (correlations only meaningful for wound data)
            wound_data = df[df["wound_pixel_count"] > 0]

            if wound_data.empty:
                return success(
                    {"files": [], "message": "No wound data available for correlation analysis"}
                )

            # Select numerical columns and calculate correlation matrix
            correlation_data = wound_data[available_columns]
            correlation_matrix = correlation_data.corr()

            # Generate heatmap
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )

            ax.set_title(
                "Correlation Matrix of Wound Metrics", fontsize=16, fontweight="bold", pad=20
            )
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(rotation=0, fontsize=10)

            file_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()

            return success({"files": [file_path]})

        except Exception as e:
            return failure(f"Error generating correlation heatmap: {e}")

    def generate_no_wound_examples(
        self, df: pd.DataFrame, image_data_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """
        Generates a composite image of examples with no wound mask.

        **Contract:**

        Args:
            df (pd.DataFrame): The metadata DataFrame.
            image_data_path (str): Path to the root of the image dataset.
            output_dir (str): Directory to save the output PNG.

        Returns:
            Dict[str, Any]: Success/failure dict with the path to the generated file.
        """
        try:
            # Filter to images with wound_pixel_count == 0
            no_wound_df = df[df["wound_pixel_count"] == 0]

            if no_wound_df.empty:
                return success(
                    {"files": [], "message": "No images with wound_pixel_count == 0 found"}
                )

            # Sample up to overlay_sample_size images
            sample_size = min(self.config["overlay_sample_size"], len(no_wound_df))
            sample_df = (
                no_wound_df.sample(n=sample_size) if len(no_wound_df) > sample_size else no_wound_df
            )

            # Load images
            loaded_images = []
            image_ids = []

            for _, row in sample_df.iterrows():
                # Find image in train or test directory
                image_path = None
                for split in ["train", "test"]:
                    potential_image_path = os.path.join(
                        image_data_path, f"{split}_images", row["image_id"]
                    )
                    if os.path.exists(potential_image_path):
                        image_path = potential_image_path
                        break

                if image_path is None:
                    continue

                # Load the image
                image = cv2.imread(image_path)
                if image is not None:
                    # Convert BGR to RGB for matplotlib display
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    loaded_images.append(image_rgb)
                    image_ids.append(row["image_id"])

            if not loaded_images:
                return success(
                    {"files": [], "message": "No valid image files found for no-wound examples"}
                )

            # Create subplot grid
            n_images = len(loaded_images)
            n_cols = min(3, n_images)
            n_rows = (n_images + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

            # Handle single row/column cases
            if n_images == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            # Display images
            for idx, (image, image_id) in enumerate(zip(loaded_images, image_ids)):
                axes[idx].imshow(image)
                axes[idx].set_title(image_id, fontsize=10, fontweight="bold")
                axes[idx].axis("off")

            # Hide unused subplots
            for idx in range(n_images, len(axes)):
                axes[idx].axis("off")

            # Add main title
            fig.suptitle("Examples of Images with No Wound", fontsize=14, fontweight="bold")
            plt.tight_layout()

            # Save figure
            output_file = os.path.join(output_dir, "no_wound_examples.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            return success({"files": [output_file]})

        except Exception as e:
            return failure(f"Error generating no-wound examples: {e}")

    def generate_positional_heatmap(self, df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Generates a heatmap of wound centroid positions.

        **Contract:**

        Args:
            df (pd.DataFrame): The metadata DataFrame with centroid columns.
            output_dir (str): Directory to save the output PNG.

        Returns:
            Dict[str, Any]: Success/failure dict with the path to the generated file.
        """
        try:
            # Check for required columns
            required_columns = ["centroid_x", "centroid_y", "wound_pixel_count"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return failure(f"Missing required columns: {', '.join(missing_columns)}")

            # Filter to only images with wounds (centroid (0,0) means no wound)
            wound_data = df[df["wound_pixel_count"] > 0]

            if wound_data.empty:
                return success(
                    {"files": [], "message": "No wound data available for positional heatmap"}
                )

            # Extract centroid coordinates
            x_coords = wound_data["centroid_x"].values
            y_coords = wound_data["centroid_y"].values

            # Create figure
            plt.figure(figsize=(10, 10))

            # Create 2D histogram heatmap
            # Use hexbin for smooth visualization
            plt.hexbin(x_coords, y_coords, gridsize=30, cmap="YlOrRd", mincnt=1)
            plt.colorbar(label="Wound Count")

            # Set axis limits to 512x512 coordinate space
            plt.xlim(0, 512)
            plt.ylim(0, 512)

            # Invert y-axis (image coordinates have origin at top-left)
            plt.gca().invert_yaxis()

            # Labels and title
            plt.title("Wound Centroid Positional Heatmap", fontsize=14, fontweight="bold")
            plt.xlabel("X Position (pixels)", fontsize=12)
            plt.ylabel("Y Position (pixels)", fontsize=12)

            # Add grid for reference
            plt.grid(True, alpha=0.3, linestyle="--")

            # Save figure
            output_file = os.path.join(output_dir, "positional_heatmap.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            return success({"files": [output_file]})

        except Exception as e:
            return failure(f"Error generating positional heatmap: {e}")

    def generate_extrema_examples(
        self, df: pd.DataFrame, image_data_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """
        Generates composite images showing min/max cases for key metrics.

        **Contract:**

        Args:
            df (pd.DataFrame): The metadata DataFrame.
            image_data_path (str): Path to the root of the image dataset.
            output_dir (str): Directory to save the output PNG.

        Returns:
            Dict[str, Any]: Success/failure dict with paths to the generated files.
        """
        try:
            # Filter to only wound images
            wound_df = df[df["wound_pixel_count"] > 0]

            if wound_df.empty:
                return success(
                    {"files": [], "message": "No wound data available for extrema examples"}
                )

            # Define metrics to analyze
            metrics = [
                ("wound_pixel_count", "Wound Pixel Count"),
                ("bounding_box_aspect_ratio", "Aspect Ratio"),
                ("mean_intensity", "Brightness"),
            ]

            generated_files = []
            errors = []

            for metric_col, metric_name in metrics:
                if metric_col not in wound_df.columns:
                    errors.append(f"Missing column: {metric_col}")
                    continue

                try:
                    # Find min and max rows
                    min_row = wound_df.loc[wound_df[metric_col].idxmin()]
                    max_row = wound_df.loc[wound_df[metric_col].idxmax()]

                    # Load images
                    min_image = None
                    max_image = None

                    for split in ["train", "test"]:
                        min_path = os.path.join(
                            image_data_path, f"{split}_images", min_row["image_id"]
                        )
                        if os.path.exists(min_path):
                            min_image = cv2.imread(min_path)
                            min_image = cv2.cvtColor(min_image, cv2.COLOR_BGR2RGB)

                        max_path = os.path.join(
                            image_data_path, f"{split}_images", max_row["image_id"]
                        )
                        if os.path.exists(max_path):
                            max_image = cv2.imread(max_path)
                            max_image = cv2.cvtColor(max_image, cv2.COLOR_BGR2RGB)

                        if min_image is not None and max_image is not None:
                            break

                    if min_image is None or max_image is None:
                        errors.append(f"Could not load images for {metric_name}")
                        continue

                    # Create composite image
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

                    # MIN image (left)
                    axes[0].imshow(min_image)
                    axes[0].set_title(
                        f"MIN {metric_name}\n{metric_col}={min_row[metric_col]:.2f}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    axes[0].axis("off")

                    # MAX image (right)
                    axes[1].imshow(max_image)
                    axes[1].set_title(
                        f"MAX {metric_name}\n{metric_col}={max_row[metric_col]:.2f}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    axes[1].axis("off")

                    plt.tight_layout()

                    # Save
                    filename = f"extrema_{metric_col.replace('_', '_')}.png"
                    if metric_col == "wound_pixel_count":
                        filename = "extrema_wound_pixel_count.png"
                    elif metric_col == "bounding_box_aspect_ratio":
                        filename = "extrema_aspect_ratio.png"
                    elif metric_col == "mean_intensity":
                        filename = "extrema_brightness.png"

                    output_path = os.path.join(output_dir, filename)
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    generated_files.append(output_path)

                except Exception as e:
                    errors.append(f"Error generating extrema for {metric_name}: {e}")

            return success(
                {
                    "files": generated_files,
                    "extrema_generated": len(generated_files),
                    "errors": errors if errors else [],
                }
            )

        except Exception as e:
            return failure(f"Error generating extrema examples: {e}")

    def generate_confusion_matrix(self, results_df: pd.DataFrame, output_dir: str) -> Dict[str, Any]:
        """
        Generates and saves a confusion matrix visualization for model performance.

        **Contract:**

        Args:
            results_df (pd.DataFrame): DataFrame containing per-image results, including
                                     columns for TP, FP, TN, FN (True/False Positives/Negatives).
            output_dir (str): The directory where the output PNG file will be saved.

        Returns:
            Dict[str, Any]: A structured dictionary indicating success or failure.
                            On success, `data['files']` contains the path to the generated chart.
        """
        try:
            # Check for required columns
            required_columns = ["TP", "FP", "TN", "FN"]
            missing_columns = [col for col in required_columns if col not in results_df.columns]
            if missing_columns:
                return failure(f"Missing required columns: {', '.join(missing_columns)}")

            # Sum up confusion matrix values across all images
            tp = results_df["TP"].sum()
            fp = results_df["FP"].sum()
            tn = results_df["TN"].sum()
            fn = results_df["FN"].sum()

            # Check if all values are zero (edge case)
            total = tp + fp + tn + fn
            if total == 0:
                return failure("All confusion matrix values are zero, cannot generate visualization")

            # Create confusion matrix array
            confusion_matrix = np.array([[tn, fp], [fn, tp]])

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Generate heatmap
            sns.heatmap(
                confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=True,
                square=True,
                linewidths=2,
                linecolor="black",
                ax=ax,
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"],
            )

            # Styling
            ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold", pad=20)
            ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
            ax.set_ylabel("True Label", fontsize=12, fontweight="bold")

            # Rotate labels for better readability
            plt.xticks(rotation=0, fontsize=10)
            plt.yticks(rotation=0, fontsize=10)

            # Save figure
            file_path = os.path.join(output_dir, "confusion_matrix.png")
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()

            return success({"files": [file_path]})

        except Exception as e:
            return failure(f"Error generating confusion matrix: {e}")

    def generate_example_predictions(
        self, results_df: pd.DataFrame, image_data_path: str, output_dir: str
    ) -> Dict[str, Any]:
        """
        Generates a composite image showing best, worst, and average prediction examples.

        **Contract:**

        Args:
            results_df (pd.DataFrame): DataFrame with per-image results, including `dice` score and `image_id`.
            image_data_path (str): Path to the root of the image dataset.
            output_dir (str): The directory where the output PNG file will be saved.

        Returns:
            Dict[str, Any]: A structured dictionary indicating success or failure.
                            On success, `data['files']` contains the path to the generated chart.
        """
        try:
            # Check for required columns
            required_columns = ["dice", "image_id"]
            missing_columns = [col for col in required_columns if col not in results_df.columns]
            if missing_columns:
                return failure(f"Missing required columns: {', '.join(missing_columns)}")

            if results_df.empty:
                return success(
                    {"files": [], "message": "No results data available for example predictions"}
                )

            # Find best, worst, and median dice score examples
            best_row = results_df.loc[results_df["dice"].idxmax()]
            worst_row = results_df.loc[results_df["dice"].idxmin()]

            # Find median (average) dice score
            median_dice = results_df["dice"].median()
            avg_row = results_df.iloc[(results_df["dice"] - median_dice).abs().argsort()[:1]].iloc[0]

            # Load the three images
            examples = [
                ("Best", best_row),
                ("Average", avg_row),
                ("Worst", worst_row),
            ]

            loaded_examples = []

            for example_type, row in examples:
                # Find image in train or test directory
                image_path = None
                for split in ["train", "test"]:
                    potential_image_path = os.path.join(
                        image_data_path, f"{split}_images", row["image_id"]
                    )
                    if os.path.exists(potential_image_path):
                        image_path = potential_image_path
                        break

                if image_path is None:
                    return failure(f"Could not find image: {row['image_id']}")

                # Load the image
                image = cv2.imread(image_path)
                if image is None:
                    return failure(f"Could not read image: {image_path}")

                # Convert BGR to RGB for matplotlib display
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                loaded_examples.append((example_type, image_rgb, row["dice"]))

            # Create composite image with 1 row, 3 columns
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for idx, (example_type, image, dice_score) in enumerate(loaded_examples):
                axes[idx].imshow(image)
                axes[idx].set_title(
                    f"{example_type} Dice Score\n{dice_score:.4f}", fontsize=12, fontweight="bold"
                )
                axes[idx].axis("off")

            # Add main title
            fig.suptitle("Example Predictions: Best, Average, and Worst Dice Scores", fontsize=14, fontweight="bold")
            plt.tight_layout()

            # Save figure
            output_file = os.path.join(output_dir, "example_predictions.png")
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            return success({"files": [output_file]})

        except Exception as e:
            return failure(f"Error generating example predictions: {e}")


def run_eda_pipeline(
    data_path: str,
    source_map_path: str,
    output_csv_path: str,
    output_viz_dir: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Executes the full EDA pipeline by orchestrating the components.
    """
    try:
        pipeline_errors = []
        pipeline_warnings = []

        # Initialize components
        analyzer = Analyzer(source_map_path)

        # Check analyzer initialization
        init_status = analyzer.get_initialization_status()
        if init_status["errors"]:
            pipeline_errors.extend(init_status["errors"])
        if init_status["warnings"]:
            pipeline_warnings.extend(init_status["warnings"])

        # Define CSV fieldnames
        fieldnames = [
            "image_id",
            "set",
            "source",
            "width",
            "height",
            "wound_pixel_count",
            "wound_percentage",
            "bounding_box_aspect_ratio",
            "mean_intensity",
            "mean_r",
            "mean_g",
            "mean_b",
            "centroid_x",
            "centroid_y",
        ]

        reporter = Reporter(output_csv_path, fieldnames)
        visualizer = Visualizer(config)

        # Process images from both train and test sets
        total_processed = 0
        total_errors = 0
        image_errors = []

        for image_set in ["train", "test"]:
            image_dir = os.path.join(data_path, f"{image_set}_images")
            mask_dir = os.path.join(data_path, f"{image_set}_masks")

            if not os.path.isdir(image_dir):
                warning_msg = f"Directory not found, skipping: {image_dir}"
                pipeline_warnings.append(warning_msg)
                continue

            image_files = [
                f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            for image_filename in image_files:
                image_path = os.path.join(image_dir, image_filename)
                mask_path = os.path.join(mask_dir, image_filename)

                # Analyze the image
                result = analyzer.analyze_image(image_path, mask_path)

                if result["success"]:
                    # Add the set information
                    metadata = result["data"]
                    metadata["set"] = image_set

                    # Write to CSV
                    reporter.write_row(metadata)
                    total_processed += 1
                else:
                    error_msg = f"Error processing {image_filename}: {result['error']}"
                    image_errors.append(error_msg)
                    total_errors += 1

        # Close the CSV file
        reporter.close_report()

        # Generate visualizations
        viz_result = None
        if total_processed > 0:
            viz_result = visualizer.generate_all_plots(output_csv_path, data_path, output_viz_dir)
            if not viz_result["success"]:
                pipeline_errors.append(f"Visualization error: {viz_result['error']}")

        # Generate summary table
        summary_result = None
        if total_processed > 0 and os.path.exists(output_csv_path):
            try:
                summary_reporter = SummaryReporter(metadata_csv_path=output_csv_path)
                summary_result = summary_reporter.generate_summary_table()
                if not summary_result["success"]:
                    pipeline_errors.append(f"Summary table error: {summary_result['error']}")
            except Exception as e:
                pipeline_errors.append(f"Failed to generate summary table: {str(e)}")
                summary_result = failure(f"Failed to generate summary table: {str(e)}")

        return success(
            {
                "message": "EDA pipeline completed",
                "total_processed": total_processed,
                "total_errors": total_errors,
                "visualization_result": viz_result,
                "summary_table_result": summary_result,
                "errors": pipeline_errors + image_errors,
                "warnings": pipeline_warnings,
            }
        )

    except Exception as e:
        return failure(f"Pipeline failed: {e}")
