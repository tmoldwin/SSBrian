from pathlib import Path

# Global constants for the project

# Random seed for reproducibility
RANDOM_SEED = 42

# Base paths
BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "Runs"
NO_FEEDFORWARD_DIR = RUNS_DIR / "no_feedforward"
GRID_SEARCH_FOLDER_NAME = "grid_search_EtoI_ItoE_E200_I200"
GRID_SEARCH_BASE_DIR = NO_FEEDFORWARD_DIR / GRID_SEARCH_FOLDER_NAME
GRID_SEARCH_SUMMARY_FILENAME = "grid_search_summary.pkl"
FIGURES_SUBDIR_NAME = "Figures"

# Feedforward configuration storage
FEEDFWD_CONFIG_DIR = (BASE_DIR.parent.parent / "feedforward" / "configs").resolve()
FEEDFWD_RUN_CONFIG_DIR = FEEDFWD_CONFIG_DIR / "runs"
FEEDFWD_SWEEP_CONFIG_DIR = FEEDFWD_CONFIG_DIR / "sweeps"
DEFAULT_FEEDFWD_RUN_CONFIG = FEEDFWD_RUN_CONFIG_DIR / "random_baseline.json"
DEFAULT_FEEDFWD_SWEEP_CONFIG = FEEDFWD_SWEEP_CONFIG_DIR / "alpha_dirichlet.json"

# Feedforward output folders
FEEDFWD_RUNS_ROOT = (BASE_DIR.parent.parent / "feedforward" / "Runs").resolve()
FEEDFWD_SINGLE_RUNS_DIR = FEEDFWD_RUNS_ROOT / "single_runs"
FEEDFWD_SWEEP_RUNS_DIR = FEEDFWD_RUNS_ROOT / "sweeps"
FEEDFWD_PLASTICITY_RUNS_DIR = FEEDFWD_RUNS_ROOT / "plasticity"

# Cell type network configuration storage
CELL_TYPE_CONFIG_DIR = (BASE_DIR.parent.parent / "cell_type_network" / "configs").resolve()
CELL_TYPE_RUN_CONFIG_DIR = CELL_TYPE_CONFIG_DIR / "runs"
DEFAULT_CELL_TYPE_RUN_CONFIG = CELL_TYPE_RUN_CONFIG_DIR / "test_quick_run.json"

# Cell type network output folders
CELL_TYPE_RUNS_ROOT = (BASE_DIR.parent.parent / "cell_type_network" / "Runs").resolve()
CELL_TYPE_SINGLE_RUNS_DIR = CELL_TYPE_RUNS_ROOT / "single_runs"

# File paths and locations
SAVE_FOLDER = 'figures/'

# Add other constants below as needed
