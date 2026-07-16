"""Explicit factor graph for 2-D robot localization.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, eye as sp_eye
from scipy.sparse.linalg import splu

from .io_utils import Dataset


def wrap(theta):
    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi

@dataclass
class Factor:
    kind: str
    factor_id: int
    pose_ids: Tuple[int, ...]
    measurement: np.ndarray
    sigma: np.ndarray
    landmark_xy: Optional[np.ndarray] = None
    robust_eligible: bool = False
    row_slice: Optional[slice] = None

    @property
    def dim(self) -> int:
        return int(len(self.measurement))

    @property
    def label(self) -> str:
        if self.kind == "odometry":
            return f"odom({self.pose_ids[0]},{self.pose_ids[1]})"
        if self.kind == "loop":
            return f"loop_{self.factor_id}({self.pose_ids[0]},{self.pose_ids[1]})"
        if self.kind == "gps":
            return f"gps_{self.factor_id}(pose {self.pose_ids[0]})"
        if self.kind == "landmark":
            return f"lm_{self.factor_id}(pose {self.pose_ids[0]})"
        return "prior(pose 0)"


@dataclass
class SolveResult:
    model_name: str
    trajectory: np.ndarray
    success: bool
    message: str
    nfev: int
    solver_used: str
    build_time_s: float
    solve_time_s: float
    objective_initial: float
    objective_final: float
    robust: bool
    irls_iterations: int
    robust_converged: bool
    weights: np.ndarray
    jacobian: csr_matrix
    factor_diagnostics: List[dict] = field(default_factory=list)

class FactorGraph:
    def __init__(
        self,
        ds: Dataset,
        *,
        include_gps: bool = True,
        include_loops: bool = True,
        include_landmarks: bool = True,
        sigma_scale: Optional[Dict[str, float]] = None,
    ):
        t0 = perf_counter()
        self.ds = ds
        self.n_poses = ds.n_poses
        self.n_vars = 3 * self.n_poses
        sigma_scale = sigma_scale or {}
        self.factors: List[Factor] = []
        self._build(include_gps, include_loops, include_landmarks, sigma_scale)
        row = 0
        for f in self.factors:
            f.row_slice = slice(row, row + f.dim)
            row += f.dim
        self.n_residuals = row
        self.build_time_s = perf_counter() - t0

    def _build(self, include_gps, include_loops, include_landmarks, sigma_scale):
        ds = self.ds
        noise = ds.noise
        prior = ds.prior.iloc[0]
        self.factors.append(Factor(
            kind="prior", factor_id=0, pose_ids=(0,),
            measurement=np.array([prior.mean_x, prior.mean_y, prior.mean_theta]),
            sigma=np.array([prior.sigma_x, prior.sigma_y, prior.sigma_theta]),
            robust_eligible=False,
        ))
        for r in ds.odometry.itertuples(index=False):
            self.factors.append(Factor(
                kind="odometry", factor_id=int(r.from_id), pose_ids=(int(r.from_id), int(r.to_id)),
                measurement=np.array([r.dx, r.dy, r.dtheta]),
                sigma=noise["odometry_sigma"].copy(), robust_eligible=False,
            ))
        if include_gps:
            s = noise["gps_sigma"] * sigma_scale.get("gps", 1.0)
            for r in ds.gps.itertuples(index=False):
                self.factors.append(Factor(
                    kind="gps", factor_id=int(r.meas_id), pose_ids=(int(r.pose_id),),
                    measurement=np.array([r.x, r.y]), sigma=s, robust_eligible=True,
                ))
        if include_loops:
            s = noise["loop_closure_sigma"] * sigma_scale.get("loop", 1.0)
            for r in ds.loops.itertuples(index=False):
                self.factors.append(Factor(
                    kind="loop", factor_id=int(r.closure_id), pose_ids=(int(r.from_id), int(r.to_id)),
                    measurement=np.array([r.dx, r.dy, r.dtheta]), sigma=s, robust_eligible=True,
                ))
        if include_landmarks:
            s = noise["landmark_sigma"] * sigma_scale.get("landmark", 1.0)
            lut = ds.landmark_xy
            for r in ds.landmark_observations.itertuples(index=False):
                self.factors.append(Factor(
                    kind="landmark", factor_id=int(r.obs_id), pose_ids=(int(r.pose_id),),
                    measurement=np.array([r.range, r.bearing]), sigma=s,
                    landmark_xy=lut[int(r.landmark_id)], robust_eligible=True,
                ))

    @staticmethod
    def _pose(x, pid):
        return x[3 * pid: 3 * pid + 3]

    def _residual_and_jacobian(self, factor: Factor, x: np.ndarray):
        if factor.kind == "prior":
            pid = factor.pose_ids[0]
            pose = self._pose(x, pid)
            raw = pose - factor.measurement
            raw[2] = wrap(raw[2])
            blocks = [(pid, np.eye(3))]

        elif factor.kind in ("odometry", "loop"):
            i, j = factor.pose_ids
            pi, pj = self._pose(x, i), self._pose(x, j)
            xi, yi, thi = pi
            xj, yj, thj = pj
            c, s = np.cos(thi), np.sin(thi)
            dxw, dyw = xj - xi, yj - yi
            pred_dx = c * dxw + s * dyw
            pred_dy = -s * dxw + c * dyw
            raw = np.array([pred_dx - factor.measurement[0],
                             pred_dy - factor.measurement[1],
                             wrap(wrap(thj - thi) - factor.measurement[2])])
            Ji = np.array([[-c, -s, pred_dy], [s, -c, -pred_dx], [0.0, 0.0, -1.0]])
            Jj = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
            blocks = [(i, Ji), (j, Jj)]

        elif factor.kind == "gps":
            pid = factor.pose_ids[0]
            pose = self._pose(x, pid)
            raw = pose[:2] - factor.measurement
            blocks = [(pid, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))]

        elif factor.kind == "landmark":
            pid = factor.pose_ids[0]
            pose = self._pose(x, pid)
            lx, ly = factor.landmark_xy
            dx, dy = lx - pose[0], ly - pose[1]
            q = max(dx * dx + dy * dy, 1e-12)
            rng = np.sqrt(q)
            bearing = wrap(np.arctan2(dy, dx) - pose[2])
            raw = np.array([rng - factor.measurement[0], wrap(bearing - factor.measurement[1])])
            J = np.array([[-dx / rng, -dy / rng, 0.0], [dy / q, -dx / q, -1.0]])
            blocks = [(pid, J)]
        else:
            raise ValueError(factor.kind)

        whitened = raw / factor.sigma
        whitened_blocks = [(pid, J / factor.sigma[:, None]) for pid, J in blocks]
        return raw, whitened, whitened_blocks

    def factor_norms(self, x: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.factors))
        for k, f in enumerate(self.factors):
            _, whitened, _ = self._residual_and_jacobian(f, x)
            out[k] = float(np.linalg.norm(whitened))
        return out

    def residual_vector(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        if weights is None:
            weights = np.ones(len(self.factors))
        out = np.zeros(self.n_residuals)
        for k, f in enumerate(self.factors):
            _, whitened, _ = self._residual_and_jacobian(f, x)
            out[f.row_slice] = np.sqrt(weights[k]) * whitened
        return out

    def jacobian(self, x: np.ndarray, weights: Optional[np.ndarray] = None) -> csr_matrix:
        if weights is None:
            weights = np.ones(len(self.factors))
        rows, cols, vals = [], [], []
        for k, f in enumerate(self.factors):
            _, _, blocks = self._residual_and_jacobian(f, x)
            r0 = f.row_slice.start
            scale = np.sqrt(weights[k])
            for pid, block in blocks:
                block = scale * block
                nz_r, nz_c = np.nonzero(block)
                for rr, cc in zip(nz_r, nz_c):
                    rows.append(r0 + rr)
                    cols.append(3 * pid + cc)
                    vals.append(float(block[rr, cc]))
        return coo_matrix((vals, (rows, cols)), shape=(self.n_residuals, self.n_vars)).tocsr()

    def objective(self, x: np.ndarray, robust: bool, huber_delta: float) -> float:
        norms = self.factor_norms(x)
        total = 0.0
        for norm, f in zip(norms, self.factors):
            if robust and f.robust_eligible and norm > huber_delta:
                total += 0.5 * (2.0 * huber_delta * norm - huber_delta ** 2)
            else:
                total += 0.5 * norm ** 2
        return float(total)

    def robust_weights(self, x: np.ndarray, huber_delta: float) -> np.ndarray:
        norms = self.factor_norms(x)
        w = np.ones_like(norms)
        for k, f in enumerate(self.factors):
            if f.robust_eligible and norms[k] > huber_delta:
                w[k] = huber_delta / max(norms[k], 1e-12)
        return w

    def diagnostics(self, x: np.ndarray, weights: np.ndarray) -> List[dict]:
        out = []
        for k, f in enumerate(self.factors):
            raw, whitened, _ = self._residual_and_jacobian(f, x)
            out.append({
                "factor_index": k, "kind": f.kind, "factor_id": f.factor_id,
                "label": f.label, "pose_ids": list(f.pose_ids),
                "raw_residual": raw.tolist(),
                "normalized_norm": float(np.linalg.norm(whitened)),
                "robust_weight": float(weights[k]),
            })
        return out

    def solve(
        self,
        model_name: str,
        x0: np.ndarray,
        *,
        robust: bool = False,
        huber_delta: float = 2.5,
        max_irls_iterations: int = 12,
        max_nfev: int = 400,
    ) -> SolveResult:
        x0 = np.asarray(x0, dtype=float).reshape(-1).copy()
        x0[2::3] = wrap(x0[2::3])
        t0 = perf_counter()
        objective_initial = self.objective(x0, robust, huber_delta)

        weights = np.ones(len(self.factors))
        x_cur = x0
        total_nfev = 0
        irls_iters = 0
        robust_converged = not robust
        n_outer = max_irls_iterations if robust else 1
        result = None
        near_square = 0 <= (self.n_residuals - self.n_vars) <= 50
        solver_used = "lm (dense)" if near_square and not robust else "trf (sparse, lsmr)"

        for outer in range(n_outer):
            irls_iters = outer + 1
            if solver_used.startswith("lm"):
                result = least_squares(
                    fun=lambda v, w=weights: self.residual_vector(v, w),
                    x0=x_cur,
                    jac=lambda v, w=weights: self.jacobian(v, w).toarray(),
                    method="lm", x_scale="jac",
                    xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=max_nfev,
                )
            else:
                result = least_squares(
                    fun=lambda v, w=weights: self.residual_vector(v, w),
                    x0=x_cur,
                    jac=lambda v, w=weights: self.jacobian(v, w),
                    method="trf", tr_solver="lsmr", x_scale="jac",
                    xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=max_nfev,
                )
            total_nfev += int(result.nfev)
            x_new = result.x.copy()
            x_new[2::3] = wrap(x_new[2::3])
            if not robust:
                x_cur = x_new
                break
            new_weights = self.robust_weights(x_new, huber_delta)
            weight_change = float(np.max(np.abs(new_weights - weights)))
            state_change = float(np.linalg.norm(x_new - x_cur) / np.sqrt(len(x_new)))
            x_cur, weights = x_new, new_weights
            if weight_change < 1e-4 and state_change < 1e-6:
                robust_converged = True
                break

        solve_time = perf_counter() - t0
        final_jac = self.jacobian(x_cur, weights)
        return SolveResult(
            model_name=model_name, trajectory=x_cur.reshape(self.n_poses, 3),
            success=bool(result.success), message=str(result.message), nfev=total_nfev,
            solver_used=solver_used, build_time_s=self.build_time_s, solve_time_s=solve_time,
            objective_initial=objective_initial,
            objective_final=self.objective(x_cur, robust, huber_delta),
            robust=robust, irls_iterations=irls_iters, robust_converged=robust_converged,
            weights=weights, jacobian=final_jac, factor_diagnostics=self.diagnostics(x_cur, weights),
        )

    def marginal_covariance_full(self, result: SolveResult, reg: float = 1e-9) -> np.ndarray:
        H = (result.jacobian.T @ result.jacobian).toarray()
        H = 0.5 * (H + H.T)
        H.flat[::H.shape[0] + 1] += reg
        return np.linalg.inv(H)

    def selected_pose_covariances(self, result: SolveResult, pose_ids, reg: float = 1e-9):
        H = (result.jacobian.T @ result.jacobian).tocsc()
        H = H + reg * sp_eye(self.n_vars, format="csc")
        lu = splu(H)
        out = {}
        for pid in pose_ids:
            idx = np.arange(3 * pid, 3 * pid + 3)
            rhs = np.zeros((self.n_vars, 3))
            rhs[idx, np.arange(3)] = 1.0
            cols = lu.solve(rhs)
            block = cols[idx, :]
            out[int(pid)] = 0.5 * (block + block.T)
        return out
