"""Evaluation utilities used strictly after all MAP and sensitivity solves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "starter"))
from se2_utils import between as se2_between, wrap_angle as se2_wrap  # noqa: E402


def trajectory_metrics(estimate: np.ndarray, truth_df: pd.DataFrame, relative_lag: int = 5) -> dict:
    truth = truth_df[["x", "y", "theta"]].to_numpy(dtype=float)
    pos_err = np.linalg.norm(estimate[:, :2] - truth[:, :2], axis=1)
    heading_err = np.abs(se2_wrap(estimate[:, 2] - truth[:, 2]))
    rel_t, rel_r = [], []
    for i in range(len(estimate) - relative_lag):
        e = se2_between(estimate[i], estimate[i + relative_lag])
        g = se2_between(truth[i], truth[i + relative_lag])
        rel_t.append(np.linalg.norm(e[:2] - g[:2]))
        rel_r.append(abs(float(se2_wrap(e[2] - g[2]))))
    rel_t, rel_r = np.asarray(rel_t), np.asarray(rel_r)
    return {
        "rmse_ate_m": float(np.sqrt(np.mean(pos_err ** 2))),
        "mean_position_error_m": float(np.mean(pos_err)),
        "max_position_error_m": float(np.max(pos_err)),
        "final_drift_m": float(pos_err[-1]),
        "heading_rmse_rad": float(np.sqrt(np.mean(heading_err ** 2))),
        "relative_translation_rmse_m_lag5": float(np.sqrt(np.mean(rel_t ** 2))),
        "relative_rotation_rmse_rad_lag5": float(np.sqrt(np.mean(rel_r ** 2))),
        "position_error_series": pos_err,
    }


CHI2_95_2DOF = 5.991464547107979  # 95% quantile of chi-square with 2 dof


def covariance_summary(cov_blocks: dict, pose_ids) -> dict:
    """95%-confidence position-uncertainty ellipse area per pose:
    area = pi * chi2_{0.95,2} * sqrt(det(Sigma_xy)) -- the standard
    confidence-region area for a 2-D Gaussian marginal, i.e. exactly the
    covariance-form object K&F's Gaussian canonical form gives you once you
    convert the local information block back to (mu, Sigma)."""
    rows = []
    for pid in pose_ids:
        block = np.asarray(cov_blocks[pid])[:2, :2]
        block = 0.5 * (block + block.T)
        eigvals = np.clip(np.linalg.eigvalsh(block), 0.0, None)
        area = float(np.pi * CHI2_95_2DOF * np.sqrt(max(np.linalg.det(block), 0.0)))
        rows.append({
            "pose_id": pid, "var_x": float(block[0, 0]), "var_y": float(block[1, 1]),
            "cov_xy": float(block[0, 1]), "ellipse_area_95_m2": area,
            "largest_std_m": float(np.sqrt(eigvals[-1])) if len(eigvals) else 0.0,
        })
    areas = np.array([r["ellipse_area_95_m2"] for r in rows])
    return {"rows": rows, "mean_ellipse_area_95_m2": float(np.mean(areas)),
            "max_ellipse_area_95_m2": float(np.max(areas))}
