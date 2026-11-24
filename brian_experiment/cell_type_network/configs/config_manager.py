from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from brian_experiment.shared.core import constants as consts


@dataclass
class LoadedConfig:
    """Container for any JSON-driven configuration."""

    name: str
    path: Optional[Path]
    params: Dict[str, Any]
    description: Optional[str] = None


def _normalize_candidate(path_like: str) -> Path:
    candidate = Path(path_like).expanduser()
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".json")
    return candidate


def resolve_config_path(selection: Optional[str]) -> Optional[Path]:
    """
    Resolve a cell type network run config path.

    selection may be:
        - absolute/relative path to a JSON file
        - bare filename (with or without .json)
        - None (fallback to default)
    """
    base_dir = consts.CELL_TYPE_RUN_CONFIG_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    if selection:
        candidate = _normalize_candidate(selection)
        if candidate.exists():
            return candidate

        relative_candidate = (base_dir / candidate.name)
        if relative_candidate.exists():
            return relative_candidate

    default_path = consts.DEFAULT_CELL_TYPE_RUN_CONFIG
    if default_path and Path(default_path).exists():
        return Path(default_path)

    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_config(selection: Optional[str], defaults: Optional[Dict[str, Any]] = None) -> LoadedConfig:
    """
    Load a single-run configuration for cell type network, merging it with provided defaults.
    """
    defaults = defaults or {}
    path = resolve_config_path(selection)

    if path is None and not defaults:
        raise FileNotFoundError("No run config provided and no defaults available.")

    payload = _load_json(path) if path else {}
    merged = {**defaults, **payload}

    name = payload.get("name") or (path.stem if path else "inline_run")
    description = payload.get("description")

    return LoadedConfig(name=name, path=path, params=merged, description=description)


def prepare_single_run_folders(config_name: str) -> Tuple[Path, Path, Path]:
    """
    Create the folder hierarchy for a single run based on the config name.
    Overwrites previous run (no timestamping).

    Returns:
        run_root, figures_dir, data_dir
    """
    run_root = consts.CELL_TYPE_SINGLE_RUNS_DIR / config_name / "latest"
    figures_dir = run_root / "Figures"
    data_dir = run_root / "Data"

    # Remove old run if it exists
    if run_root.exists():
        import shutil
        shutil.rmtree(run_root)

    for folder in (figures_dir, data_dir):
        folder.mkdir(parents=True, exist_ok=True)

    return run_root, figures_dir, data_dir

