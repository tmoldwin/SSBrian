"""
Create basic regression plots comparing connection strengths against key metrics.
"""

from pathlib import Path
import pickle
import numpy as np

from constants import (
    GRID_SEARCH_BASE_DIR,
    GRID_SEARCH_SUMMARY_FILENAME,
    FIGURES_SUBDIR_NAME,
)
from plots import plot_connection_metric_regression_grid, plot_multiple_regression_coefficients


def find_latest_grid_search_summary(base_dir: Path = GRID_SEARCH_BASE_DIR) -> tuple[Path, Path]:
    """
    Locate the newest timestamped grid search folder that contains the summary file.

    Returns:
        summary_path (Path): path to the grid_search_summary.pkl file
        folder_path (Path): parent folder containing figures/results
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Grid search base directory not found: {base_dir}")

    timestamp_dirs = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name != "old":
            summary_path = child / GRID_SEARCH_SUMMARY_FILENAME
            if summary_path.exists():
                timestamp_dirs.append((child.name, summary_path, child))

    if not timestamp_dirs:
        raise FileNotFoundError(f"No {GRID_SEARCH_SUMMARY_FILENAME} files found in {base_dir}")

    timestamp_dirs.sort(key=lambda item: item[0], reverse=True)
    _, summary_path, folder_path = timestamp_dirs[0]
    return summary_path, folder_path


def _safe_rate_ratio(result: dict) -> float:
    """Compute E/I firing rate ratio, handling invalid values."""
    e_rate = result.get("e_rate")
    i_rate = result.get("i_rate")
    if e_rate is None or i_rate in (None, 0):
        return np.nan
    return float(e_rate) / float(i_rate)


def _safe_connection_ratio(result: dict, num_key: str, den_key: str) -> float:
    """Compute connection ratio, handling invalid values."""
    num = result.get(num_key)
    den = result.get(den_key)
    if num is None or den in (None, 0):
        return np.nan
    return float(num) / float(den)


def build_metrics_config():
    """Return configuration for the connection-versus-metric regressions."""
    return [
        {
            "name": "E firing rate (Hz)",
            "value_fn": lambda result: float(result.get("e_rate", np.nan)),
            "color": "#d7301f",
            "log_y": True,
        },
        {
            "name": "I firing rate (Hz)",
            "value_fn": lambda result: float(result.get("i_rate", np.nan)),
            "color": "#045a8d",
            "log_y": True,
        },
        {
            "name": "E/I rate ratio",
            "value_fn": _safe_rate_ratio,
            "color": "#54278f",
            "log_y": True,
        },
        {
            "name": "E-I rate correlation",
            "value_fn": lambda result: float(result.get("rate_correlation", np.nan)),
            "color": "#238b45",
        },
        {
            "name": "Pairwise corr (E)",
            "value_fn": lambda result: float(result.get("mean_pairwise_corr_e", np.nan)),
            "color": "#cb181d",
        },
        {
            "name": "Pairwise corr (I)",
            "value_fn": lambda result: float(result.get("mean_pairwise_corr_i", np.nan)),
            "color": "#08519c",
        },
    ]


def build_connection_fields():
    """Return list of connection fields (single scales and their ratios)."""
    base_fields = [
        ("E→E", "e_to_e_scale"),
        ("E→I", "e_to_i_scale"),
        ("I→E", "i_to_e_scale"),
        ("I→I", "i_to_i_scale"),
    ]

    ratio_fields = []
    for num_label, num_key in base_fields:
        for den_label, den_key in base_fields:
            if num_key == den_key:
                continue
            ratio_fields.append({
                "label": f"{num_label} / {den_label}",
                "value_fn": lambda result, nk=num_key, dk=den_key: _safe_connection_ratio(result, nk, dk),
            })

    return base_fields + ratio_fields


def main():
    summary_path, folder_path = find_latest_grid_search_summary()
    print(f"Loading grid search summary: {summary_path}")

    with summary_path.open("rb") as handle:
        all_results = pickle.load(handle)

    successful_results = [result for result in all_results if result.get("success")]
    if not successful_results:
        raise RuntimeError("No successful runs available for regression plotting.")

    figures_dir = folder_path / FIGURES_SUBDIR_NAME
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_config = build_metrics_config()
    connection_fields = build_connection_fields()
    output_path = figures_dir / "grid_search_basic_regressions.png"

    plot_connection_metric_regression_grid(
        successful_results=successful_results,
        metrics_config=metrics_config,
        connection_fields=connection_fields,
        save_path=output_path,
        figure_title="Connection Strength vs Performance Metrics",
        log_x_axis=True,
    )

    print(f"Saved regression grid to: {output_path}")
    
    # Create multiple regression analysis
    regression_output_path = figures_dir / "grid_search_multiple_regression.png"
    plot_multiple_regression_coefficients(
        successful_results=successful_results,
        metrics_config=metrics_config,
        save_path=regression_output_path,
        figure_title="Multiple Regression: Connection Factors vs Metrics",
    )
    
    print(f"Saved multiple regression analysis to: {regression_output_path}")


if __name__ == "__main__":
    main()

