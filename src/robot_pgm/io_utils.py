"""Dataset loading and validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Dataset:
    root: Path
    config: Dict[str, Any]
    prior: pd.DataFrame
    odometry: pd.DataFrame
    gps: pd.DataFrame
    loops: pd.DataFrame
    landmarks: pd.DataFrame
    landmark_observations: pd.DataFrame

    @property
    def n_poses(self) -> int:
        return int(self.config["number_of_poses"])

    @property
    def noise(self) -> Dict[str, np.ndarray]:
        model = self.config["noise_model_nominal"]
        return {k: np.asarray(v, dtype=float) for k, v in model.items()}

    @property
    def landmark_xy(self) -> Dict[int, np.ndarray]:
        return {int(r.landmark_id): np.array([r.x, r.y]) for r in self.landmarks.itertuples()}

    def summary(self) -> Dict[str, int]:
        return {
            "n_poses": self.n_poses,
            "n_odometry_edges": len(self.odometry),
            "n_gps": len(self.gps),
            "n_loop_closures": len(self.loops),
            "n_landmarks": len(self.landmarks),
            "n_landmark_observations": len(self.landmark_observations),
        }


def _read_csv(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.exists():
        raise FileNotFoundError(f"Required dataset file missing: {path}")
    return pd.read_csv(path)


def load_dataset(root) -> Dataset:
    root = Path(root).resolve()
    with (root / "configs" / "reference.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    files = config["files"]
    ds = Dataset(
        root=root,
        config=config,
        prior=_read_csv(root, files["prior"]),
        odometry=_read_csv(root, files["odometry"]),
        gps=_read_csv(root, files["gps"]),
        loops=_read_csv(root, files["loop_closures"]),
        landmarks=_read_csv(root, files["landmarks"]),
        landmark_observations=_read_csv(root, files["landmark_observations"]),
    )
    validate_dataset(ds)
    return ds


def load_ground_truth_for_evaluation_only(ds: Dataset) -> pd.DataFrame:
    """The only function in the whole codebase allowed to touch ground truth.
    Never imported by factor_graph.py or pgm_analysis.py."""
    relative = ds.config["files"]["ground_truth_for_evaluation_only"]
    gt = _read_csv(ds.root, relative).sort_values("pose_id").reset_index(drop=True)
    if not np.array_equal(gt["pose_id"].to_numpy(), np.arange(ds.n_poses)):
        raise ValueError("ground_truth.csv pose_id column must be exactly 0..N-1 in order.")
    return gt


def validate_dataset(ds: Dataset) -> None:
    n = ds.n_poses
    expected_columns = {
        "prior": {"pose_id", "mean_x", "mean_y", "mean_theta", "sigma_x", "sigma_y", "sigma_theta"},
        "odometry": {"from_id", "to_id", "dx", "dy", "dtheta"},
        "gps": {"meas_id", "pose_id", "x", "y"},
        "loops": {"closure_id", "from_id", "to_id", "dx", "dy", "dtheta"},
        "landmarks": {"landmark_id", "x", "y"},
        "landmark_observations": {"obs_id", "pose_id", "landmark_id", "range", "bearing"},
    }
    tables = {
        "prior": ds.prior, "odometry": ds.odometry, "gps": ds.gps,
        "loops": ds.loops, "landmarks": ds.landmarks,
        "landmark_observations": ds.landmark_observations,
    }
    for name, table in tables.items():
        missing = expected_columns[name] - set(table.columns)
        if missing:
            raise ValueError(f"{name} missing columns: {sorted(missing)}")
        numeric = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{name} contains non-finite values.")

    if len(ds.prior) != 1 or int(ds.prior.iloc[0]["pose_id"]) != 0:
        raise ValueError("Exactly one prior row, on pose_id 0, is required.")

    if len(ds.odometry) != n - 1:
        raise ValueError(f"Expected {n - 1} odometry rows, found {len(ds.odometry)}.")
    from_ids = ds.odometry["from_id"].to_numpy(dtype=int)
    to_ids = ds.odometry["to_id"].to_numpy(dtype=int)
    if not (np.array_equal(from_ids, np.arange(n - 1)) and np.array_equal(to_ids, np.arange(1, n))):
        raise ValueError("Odometry must connect each consecutive pose exactly once, in temporal order.")

    for name, table, cols in [
        ("gps", ds.gps, ["pose_id"]),
        ("loops", ds.loops, ["from_id", "to_id"]),
        ("landmark_observations", ds.landmark_observations, ["pose_id"]),
    ]:
        for c in cols:
            v = table[c].to_numpy(dtype=int)
            if np.any(v < 0) or np.any(v >= n):
                raise ValueError(f"{name}.{c} has an out-of-range pose id.")

    known_landmarks = set(ds.landmarks["landmark_id"].astype(int))
    observed_landmarks = set(ds.landmark_observations["landmark_id"].astype(int))
    if not observed_landmarks.issubset(known_landmarks):
        raise ValueError("A landmark observation references an unknown landmark_id.")

    prior_sigma = ds.prior[["sigma_x", "sigma_y", "sigma_theta"]].to_numpy(dtype=float)
    if np.any(prior_sigma <= 0):
        raise ValueError("Prior sigma must be strictly positive.")
    for name, sigma in ds.noise.items():
        if np.any(sigma <= 0):
            raise ValueError(f"noise_model_nominal.{name} must be strictly positive.")
