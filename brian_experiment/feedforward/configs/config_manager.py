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


def _kind_to_base_dir(kind: str) -> Path:
    if kind == "run":
        return consts.FEEDFWD_RUN_CONFIG_DIR
    if kind == "sweep":
        return consts.FEEDFWD_SWEEP_CONFIG_DIR
    raise ValueError(f"Unknown config kind: {kind}")


def _kind_to_default(kind: str) -> Optional[Path]:
    if kind == "run":
        return getattr(consts, "DEFAULT_FEEDFWD_RUN_CONFIG", None)
    if kind == "sweep":
        return getattr(consts, "DEFAULT_FEEDFWD_SWEEP_CONFIG", None)
    return None


def _normalize_candidate(path_like: str) -> Path:
    candidate = Path(path_like).expanduser()
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".json")
    return candidate


def resolve_config_path(kind: str, selection: Optional[str]) -> Optional[Path]:
    """
    Resolve a run/sweep config path.

    selection may be:
        - absolute/relative path to a JSON file
        - bare filename (with or without .json)
        - None (fallback to default for the kind)
    """
    base_dir = _kind_to_base_dir(kind)
    base_dir.mkdir(parents=True, exist_ok=True)

    if selection:
        candidate = _normalize_candidate(selection)
        if candidate.exists():
            return candidate

        relative_candidate = (base_dir / candidate.name)
        if relative_candidate.exists():
            return relative_candidate

    default_path = _kind_to_default(kind)
    if default_path and Path(default_path).exists():
        return Path(default_path)

    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_config(selection: Optional[str], defaults: Optional[Dict[str, Any]] = None) -> LoadedConfig:
    """
    Load a single-run configuration, merging it with provided defaults.
    """
    defaults = defaults or {}
    path = resolve_config_path("run", selection)

    if path is None and not defaults:
        raise FileNotFoundError("No run config provided and no defaults available.")

    payload = _load_json(path) if path else {}
    merged = {**defaults, **payload}

    name = payload.get("name") or (path.stem if path else "inline_run")
    description = payload.get("description")

    return LoadedConfig(name=name, path=path, params=merged, description=description)


def load_sweep_config(selection: Optional[str]) -> LoadedConfig:
    """
    Load a sweep configuration.
    """
    path = resolve_config_path("sweep", selection)
    if path is None:
        raise FileNotFoundError("No sweep configuration found for the provided selector.")

    payload = _load_json(path)
    name = payload.get("name") or path.stem
    description = payload.get("description")

    return LoadedConfig(name=name, path=path, params=payload, description=description)


def prepare_single_run_folders(config_name: str) -> Tuple[Path, Path, Path]:
    """
    Create the folder hierarchy for a single run based on the config name.

    Returns:
        run_root, figures_dir, data_dir
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = consts.FEEDFWD_SINGLE_RUNS_DIR / config_name / timestamp
    figures_dir = run_root / "Figures"
    data_dir = run_root / "Data"

    for folder in (figures_dir, data_dir):
        folder.mkdir(parents=True, exist_ok=True)

    return run_root, figures_dir, data_dir


def prepare_sweep_folder(sweep_name: str) -> Tuple[Path, str]:
    """
    Create the root folder for a sweep execution.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_root = consts.FEEDFWD_SWEEP_RUNS_DIR / sweep_name / timestamp
    sweep_root.mkdir(parents=True, exist_ok=True)
    return sweep_root, timestamp

