"""
End-to-end pipeline. Run with: python3 scripts/run_project.py
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scipy
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))

import se2_utils  # noqa: E402
from robot_pgm.io_utils import load_dataset, load_ground_truth_for_evaluation_only  # noqa: E402
from robot_pgm.factor_graph import FactorGraph  # noqa: E402
from robot_pgm import pgm_analysis as pgm  # noqa: E402
from robot_pgm import evaluation as ev  # noqa: E402
from robot_pgm import plotting as pl  # noqa: E402

FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
RESULTS: dict = {}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, tuple):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return o


def metric_payload(metrics: dict) -> dict:
    """Remove the long per-pose series before writing summary tables/JSON."""
    return {k: v for k, v in metrics.items() if k != "position_error_series"}


def make_smooth_perturbed_initialization(
    x0: np.ndarray,
    position_amplitude_m: float,
    heading_amplitude_rad: float,
) -> np.ndarray:
    poses = np.asarray(x0, dtype=float).reshape(-1, 3).copy()
    phase = np.linspace(0.0, 1.0, len(poses))
    poses[:, 0] += position_amplitude_m * np.sin(2.0 * np.pi * phase)
    poses[:, 1] += position_amplitude_m * np.sin(np.pi * phase)
    poses[:, 2] += heading_amplitude_rad * np.sin(3.0 * np.pi * phase)
    poses[:, 2] = (poses[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    return poses.reshape(-1)


def main():
    t_start = time.perf_counter()
    ds = load_dataset(ROOT)
    RESULTS["dataset_summary"] = ds.summary()
    RESULTS["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "PyYAML": yaml.__version__,
    }

    prior = ds.prior.iloc[0]
    x0_pose = np.array([prior.mean_x, prior.mean_y, prior.mean_theta])
    dr = se2_utils.integrate_odometry(x0_pose, ds.odometry.to_dict("records"))
    x0 = dr.flatten()

    traj_by_model = {"dead_reckoning": dr}

    model_defs = {
        "odometry_only": dict(include_gps=False, include_loops=False, include_landmarks=False),
        "odometry_gps": dict(include_gps=True, include_loops=False, include_landmarks=False),
        "odometry_loops": dict(include_gps=False, include_loops=True, include_landmarks=False),
        "odometry_landmarks": dict(include_gps=False, include_loops=False, include_landmarks=True),
        "all_gaussian": dict(include_gps=True, include_loops=True, include_landmarks=True),
    }
    results_by_model = {}
    for name, kw in model_defs.items():
        graph = FactorGraph(ds, **kw)
        result = graph.solve(name, x0, robust=False)
        results_by_model[name] = (graph, result)
        traj_by_model[name] = result.trajectory
        print(
            f"[{name:20s}] solver={result.solver_used:16s} nfev={result.nfev:4d} "
            f"obj {result.objective_initial:10.1f} -> {result.objective_final:8.2f} "
            f"build={result.build_time_s:.3f}s solve={result.solve_time_s:.2f}s "
            f"ok={result.success}"
        )

    # ---- full robust graph, also initialized directly from dead reckoning ---
    g_full = FactorGraph(ds, include_gps=True, include_loops=True, include_landmarks=True)
    res_robust = g_full.solve(
        "all_robust",
        x0,
        robust=True,
        huber_delta=2.5,
        max_irls_iterations=20,
    )
    traj_by_model["all_robust"] = res_robust.trajectory
    print(
        f"[{'all_robust':20s}] solver={res_robust.solver_used:16s} nfev={res_robust.nfev:4d} "
        f"IRLS_iters={res_robust.irls_iterations} converged={res_robust.robust_converged} "
        f"obj {res_robust.objective_initial:10.1f} -> {res_robust.objective_final:8.2f} "
        f"build={res_robust.build_time_s:.3f}s solve={res_robust.solve_time_s:.2f}s"
    )

    res_gaussian_full = results_by_model["all_gaussian"][1]

    initialization_specs = {
        "dead_reckoning": {
            "position_amplitude_m": 0.0,
            "heading_amplitude_rad": 0.0,
            "x0": x0,
        },
        "mild_smooth_perturbation": {
            "position_amplitude_m": 0.5,
            "heading_amplitude_rad": 0.05,
            "x0": make_smooth_perturbed_initialization(x0, 0.5, 0.05),
        },
        "strong_smooth_perturbation": {
            "position_amplitude_m": 2.0,
            "heading_amplitude_rad": 0.20,
            "x0": make_smooth_perturbed_initialization(x0, 2.0, 0.20),
        },
    }
    initialization_runs = {
        "dead_reckoning": {
            **initialization_specs["dead_reckoning"],
            "all_gaussian": {"graph": results_by_model["all_gaussian"][0], "result": res_gaussian_full},
            "all_robust": {"graph": g_full, "result": res_robust},
        }
    }
    for initialization_name in ("mild_smooth_perturbation", "strong_smooth_perturbation"):
        spec = initialization_specs[initialization_name]
        gaussian_graph = FactorGraph(
            ds, include_gps=True, include_loops=True, include_landmarks=True
        )
        gaussian_result = gaussian_graph.solve(
            f"all_gaussian__{initialization_name}", spec["x0"], robust=False
        )
        robust_graph = FactorGraph(
            ds, include_gps=True, include_loops=True, include_landmarks=True
        )
        robust_result = robust_graph.solve(
            f"all_robust__{initialization_name}",
            spec["x0"],
            robust=True,
            huber_delta=2.5,
            max_irls_iterations=20,
        )
        initialization_runs[initialization_name] = {
            **spec,
            "all_gaussian": {"graph": gaussian_graph, "result": gaussian_result},
            "all_robust": {"graph": robust_graph, "result": robust_result},
        }
        traj_by_model[f"init_{initialization_name}_all_gaussian"] = gaussian_result.trajectory
        traj_by_model[f"init_{initialization_name}_all_robust"] = robust_result.trajectory
        print(
            f"[init {initialization_name:27s}] "
            f"gaussian: obj={gaussian_result.objective_final:.2f} nfev={gaussian_result.nfev} ok={gaussian_result.success}; "
            f"robust: obj={robust_result.objective_final:.2f} nfev={robust_result.nfev} "
            f"IRLS={robust_result.irls_iterations} ok={robust_result.robust_converged}"
        )

    sensitivity_pose_ids = list(range(0, ds.n_poses, 20))
    sensitivity_runs = {}
    for scale in (0.2, 1.0, 5.0):
        model_name = f"gps_sigma_x{scale:g}"
        graph = FactorGraph(
            ds,
            include_gps=True,
            include_loops=True,
            include_landmarks=True,
            sigma_scale={"gps": scale},
        )
        result = graph.solve(model_name, x0, robust=False)
        covariance = graph.selected_pose_covariances(result, sensitivity_pose_ids)
        covariance_stats = ev.covariance_summary(covariance, sensitivity_pose_ids)
        sensitivity_runs[scale] = {
            "graph": graph,
            "result": result,
            "covariance": covariance,
            "covariance_summary": covariance_stats,
        }
        scale_tag = f"{scale:g}".replace(".", "p")
        traj_by_model[f"gps_sigma_x{scale_tag}"] = result.trajectory
        print(
            f"[{model_name:20s}] solver={result.solver_used:16s} nfev={result.nfev:4d} "
            f"obj {result.objective_initial:10.1f} -> {result.objective_final:8.2f} "
            f"build={result.build_time_s:.3f}s solve={result.solve_time_s:.2f}s "
            f"ok={result.success}"
        )

    # ---- factor diagnostics: no ground truth or outlier labels used --------
    gauss_diag = pd.DataFrame(res_gaussian_full.factor_diagnostics)
    robust_diag = pd.DataFrame(res_robust.factor_diagnostics)
    gauss_diag.to_csv(TAB / "gaussian_factor_residuals.csv", index=False)
    robust_diag.to_csv(TAB / "robust_factor_residuals.csv", index=False)

    top_bad = gauss_diag.sort_values("normalized_norm", ascending=False).head(17)
    merged = top_bad.merge(
        robust_diag[["factor_index", "normalized_norm", "robust_weight"]],
        on="factor_index",
        suffixes=("_gaussian", "_robust"),
    )
    merged.to_csv(TAB / "inconsistent_factors.csv", index=False)
    RESULTS["top_inconsistent_factors"] = merged[
        ["label", "kind", "normalized_norm_gaussian", "normalized_norm_robust", "robust_weight_robust"]
    ].to_dict(orient="records")

    selected_poses = [0, 20, 40, 60, 84, 100, 130, 160, 176, 200, 230, 259]
    cov_full = g_full.selected_pose_covariances(res_gaussian_full, selected_poses)
    cov_robust = g_full.selected_pose_covariances(res_robust, selected_poses)
    g_odom, res_odom = results_by_model["odometry_only"]
    cov_odom = g_odom.selected_pose_covariances(res_odom, selected_poses)

    cs_full = ev.covariance_summary(cov_full, selected_poses)
    cs_robust = ev.covariance_summary(cov_robust, selected_poses)
    cs_odom = ev.covariance_summary(cov_odom, selected_poses)
    uncertainty_rows = []
    for pid in selected_poses:
        row_full = next(r for r in cs_full["rows"] if r["pose_id"] == pid)
        row_robust = next(r for r in cs_robust["rows"] if r["pose_id"] == pid)
        row_odom = next(r for r in cs_odom["rows"] if r["pose_id"] == pid)
        uncertainty_rows.append({
            "pose_id": pid,
            "ellipse_area95_odom_only_m2": row_odom["ellipse_area_95_m2"],
            "ellipse_area95_all_gaussian_m2": row_full["ellipse_area_95_m2"],
            "ellipse_area95_all_robust_m2": row_robust["ellipse_area_95_m2"],
        })
    uncertainty_df = pd.DataFrame(uncertainty_rows)
    uncertainty_df.to_csv(TAB / "uncertainty_summary.csv", index=False)
    pd.DataFrame(cs_full["rows"]).to_csv(TAB / "selected_pose_covariances.csv", index=False)
    RESULTS["uncertainty_table"] = uncertainty_df.to_dict(orient="records")
    RESULTS["uncertainty_selected_poses"] = selected_poses

    mb0 = pgm.markov_blanket(ds, 0)
    mb_loop_pose = pgm.markov_blanket(ds, 118)
    RESULTS["markov_blanket_pose0"] = mb0
    RESULTS["markov_blanket_pose118"] = mb_loop_pose
    with open(TAB / "markov_blanket_pose0.json", "w", encoding="utf-8") as f:
        json.dump(clean(mb0), f, ensure_ascii=False, indent=2)
    with open(TAB / "markov_blanket_pose118.json", "w", encoding="utf-8") as f:
        json.dump(clean(mb_loop_pose), f, ensure_ascii=False, indent=2)

    adjacency = pgm.pose_adjacency(ds, include_loops=True)
    orderings = {
        "temporal_0_to_259": list(range(ds.n_poses)),
        "reverse_259_to_0": list(range(ds.n_poses - 1, -1, -1)),
        "greedy_min_degree": pgm.greedy_min_degree_order(adjacency),
    }
    fill_rows = []
    for name, order in orderings.items():
        fill_rows.append({"ordering": name, **pgm.symbolic_fill_in(adjacency, order)})
    fill_df = pd.DataFrame(fill_rows)
    fill_df.to_csv(TAB / "symbolic_fill_in.csv", index=False)
    RESULTS["symbolic_fill_in"] = fill_df.to_dict(orient="records")
    print(fill_df.to_string(index=False))

    lambda_full = (res_gaussian_full.jacobian.T @ res_gaussian_full.jacobian).toarray()
    from scipy.sparse.csgraph import reverse_cuthill_mckee
    import scipy.sparse as sp

    permutation = reverse_cuthill_mckee(sp.csr_matrix((lambda_full != 0).astype(int)))
    lambda_permuted = lambda_full[np.ix_(permutation, permutation)]
    RESULTS["information_matrix"] = {
        "dim": lambda_full.shape[0],
        "nnz": int(np.count_nonzero(lambda_full)),
    }

    gt_df = load_ground_truth_for_evaluation_only(ds)
    gt_xy = gt_df[["x", "y"]].to_numpy(float)

    dr_metrics = ev.trajectory_metrics(dr, gt_df)
    model_metrics = {
        name: ev.trajectory_metrics(result.trajectory, gt_df)
        for name, (_, result) in results_by_model.items()
    }
    robust_metrics = ev.trajectory_metrics(res_robust.trajectory, gt_df)
    sensitivity_metrics = {
        scale: ev.trajectory_metrics(run["result"].trajectory, gt_df)
        for scale, run in sensitivity_runs.items()
    }
    initialization_metrics = {
        initialization_name: {
            model_name: ev.trajectory_metrics(model_run["result"].trajectory, gt_df)
            for model_name, model_run in run.items()
            if model_name in ("all_gaussian", "all_robust")
        }
        for initialization_name, run in initialization_runs.items()
    }

    RESULTS["dead_reckoning"] = metric_payload(dr_metrics)
    print(
        f"[dead reckoning evaluation] RMSE={dr_metrics['rmse_ate_m']:.3f} m "
        f"drift={dr_metrics['final_drift_m']:.3f} m"
    )

    ablation_rows = [{
        "model": "dead_reckoning",
        **metric_payload(dr_metrics),
        "objective_initial": np.nan,
        "objective_final": np.nan,
        "build_time_s": np.nan,
        "solve_time_s": np.nan,
        "nfev": np.nan,
        "solver_used": "-",
        "converged": True,
    }]
    for name, (_, result) in results_by_model.items():
        ablation_rows.append({
            "model": name,
            **metric_payload(model_metrics[name]),
            "objective_initial": result.objective_initial,
            "objective_final": result.objective_final,
            "build_time_s": result.build_time_s,
            "solve_time_s": result.solve_time_s,
            "nfev": result.nfev,
            "solver_used": result.solver_used,
            "converged": result.success,
        })
    ablation_rows.append({
        "model": "all_robust",
        **metric_payload(robust_metrics),
        "objective_initial": res_robust.objective_initial,
        "objective_final": res_robust.objective_final,
        "build_time_s": res_robust.build_time_s,
        "solve_time_s": res_robust.solve_time_s,
        "nfev": res_robust.nfev,
        "solver_used": res_robust.solver_used,
        "converged": res_robust.robust_converged,
    })
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(TAB / "ablation_metrics.csv", index=False)
    RESULTS["ablation"] = ablation_df.to_dict(orient="records")
    print(
        ablation_df[
            ["model", "rmse_ate_m", "final_drift_m", "objective_final", "build_time_s", "solve_time_s", "nfev", "converged"]
        ].to_string(index=False)
    )

    sensitivity_rows = []
    sensitivity_trajectories = {}
    sensitivity_errors = {}
    for scale, run in sensitivity_runs.items():
        result = run["result"]
        metrics = sensitivity_metrics[scale]
        covariance_stats = run["covariance_summary"]
        sensitivity_trajectories[scale] = result.trajectory
        sensitivity_errors[scale] = metrics["position_error_series"]
        sensitivity_rows.append({
            "gps_sigma_scale": scale,
            **metric_payload(metrics),
            "objective_initial": result.objective_initial,
            "objective_final": result.objective_final,
            "build_time_s": result.build_time_s,
            "solve_time_s": result.solve_time_s,
            "nfev": result.nfev,
            "converged": result.success,
            "mean_ellipse_area_95_m2": covariance_stats["mean_ellipse_area_95_m2"],
            "max_ellipse_area_95_m2": covariance_stats["max_ellipse_area_95_m2"],
        })
    sensitivity_df = pd.DataFrame(sensitivity_rows).sort_values("gps_sigma_scale").reset_index(drop=True)
    sensitivity_df.to_csv(TAB / "noise_sensitivity.csv", index=False)
    RESULTS["noise_sensitivity"] = sensitivity_df.to_dict(orient="records")
    print(
        sensitivity_df[
            ["gps_sigma_scale", "rmse_ate_m", "final_drift_m", "objective_final", "mean_ellipse_area_95_m2", "converged"]
        ].to_string(index=False)
    )

    initialization_rows = []
    for initialization_name, run in initialization_runs.items():
        for model_name in ("all_gaussian", "all_robust"):
            result = run[model_name]["result"]
            metrics = initialization_metrics[initialization_name][model_name]
            initialization_rows.append({
                "initialization": initialization_name,
                "position_amplitude_m": run["position_amplitude_m"],
                "heading_amplitude_rad": run["heading_amplitude_rad"],
                "model": model_name,
                **metric_payload(metrics),
                "objective_initial": result.objective_initial,
                "objective_final": result.objective_final,
                "build_time_s": result.build_time_s,
                "solve_time_s": result.solve_time_s,
                "nfev": result.nfev,
                "irls_iterations": result.irls_iterations,
                "converged": result.robust_converged if model_name == "all_robust" else result.success,
            })
    initialization_df = pd.DataFrame(initialization_rows)
    initialization_df.to_csv(TAB / "initialization_sensitivity.csv", index=False)
    RESULTS["initialization_sensitivity"] = initialization_df.to_dict(orient="records")
    print(
        initialization_df[
            ["initialization", "model", "rmse_ate_m", "objective_final", "nfev", "irls_iterations", "converged"]
        ].to_string(index=False)
    )

    gaussian_robust_df = pd.DataFrame([
        {"model": "all_gaussian", **metric_payload(model_metrics["all_gaussian"])},
        {"model": "all_robust", **metric_payload(robust_metrics)},
    ])
    gaussian_robust_df.to_csv(TAB / "gaussian_vs_robust.csv", index=False)
    RESULTS["gaussian_vs_robust"] = gaussian_robust_df.to_dict(orient="records")

    pl.plot_factor_graph_schematic(FIG / "factor_graph_schematic.png")
    pl.plot_trajectory_comparison(
        gt_xy,
        dr[:, :2],
        res_robust.trajectory[:, :2],
        FIG / "trajectory_comparison.png",
        "Ground truth vs. dead reckoning vs. MAP (robust) trajectory",
        optimized_label="Optimized (MAP, robust)",
    )

    error_series = {
        "dead_reckoning": dr_metrics["position_error_series"],
        "odometry_only": model_metrics["odometry_only"]["position_error_series"],
        "all_gaussian": model_metrics["all_gaussian"]["position_error_series"],
        "all_robust": robust_metrics["position_error_series"],
    }
    pl.plot_error_vs_time(error_series, FIG / "error_vs_time.png")
    pl.plot_loop_closures(res_gaussian_full.trajectory, ds.loops, FIG / "loop_closures.png")

    landmarks_xy = np.array([ds.landmark_xy[k] for k in sorted(ds.landmark_xy)])
    pl.plot_covariance_ellipses(
        res_gaussian_full.trajectory,
        cov_full,
        FIG / "covariance_ellipses.png",
        landmarks_xy=landmarks_xy,
    )

    pl.bar_plot(
        ablation_df["model"],
        ablation_df["rmse_ate_m"],
        "RMSE / ATE (m)",
        "Ablation study: position RMSE by sensor combination",
        FIG / "ablation_rmse.png",
    )
    pl.bar_plot(
        sensitivity_df["gps_sigma_scale"].astype(str),
        sensitivity_df["rmse_ate_m"],
        "RMSE / ATE (m)",
        "Sensitivity of RMSE to GPS covariance scale (0.2x / 1x / 5x)",
        FIG / "noise_sensitivity_rmse.png",
        rotate=0,
    )

    gps_xy = ds.gps[["x", "y"]].to_numpy(dtype=float)
    pl.plot_noise_sensitivity_trajectories(
        ground_truth_xy=gt_xy,
        dead_reckoning_xy=dr[:, :2],
        gps_xy=gps_xy,
        sensitivity_trajectories=sensitivity_trajectories,
        sensitivity_summary=sensitivity_df,
        path=FIG / "noise_sensitivity_trajectories.png",
    )
    pl.plot_noise_sensitivity_error_vs_pose(
        dead_reckoning_error=dr_metrics["position_error_series"],
        sensitivity_errors=sensitivity_errors,
        path=FIG / "noise_sensitivity_error_vs_pose.png",
    )
    pl.plot_noise_sensitivity_tradeoff(
        sensitivity_df=sensitivity_df,
        path=FIG / "noise_sensitivity_accuracy_uncertainty.png",
    )

    pl.plot_initialization_sensitivity(
        initialization_df=initialization_df,
        path=FIG / "initialization_sensitivity.png",
    )

    pl.bar_plot(
        gaussian_robust_df["model"],
        gaussian_robust_df["rmse_ate_m"],
        "RMSE / ATE (m)",
        "Gaussian vs. IRLS-robust (Huber, per-factor norm): position RMSE",
        FIG / "gaussian_vs_robust_rmse.png",
        rotate=0,
    )
    pl.grouped_bar_plot(
        merged["label"].tolist(),
        {
            "Gaussian solution": merged["normalized_norm_gaussian"].tolist(),
            "Robust solution": merged["normalized_norm_robust"].tolist(),
        },
        "Normalized (whitened) residual norm",
        "Most inconsistent factors: Gaussian vs. robust solution",
        FIG / "gaussian_vs_robust_residuals.png",
    )
    pl.fill_in_bar_plot(
        fill_df["ordering"],
        fill_df["fill_edges"],
        fill_df["max_induced_clique_size"],
        FIG / "symbolic_fill_in.png",
    )
    pl.plot_sparsity(
        lambda_full,
        lambda_permuted,
        FIG / "information_sparsity.png",
        "natural (temporal) order",
        "RCM-reordered (numeric analogue)",
        "Numeric information-matrix sparsity (companion to the symbolic fill-in analysis)",
    )

    # ---- save summary metadata and all trajectories -------------------------
    RESULTS["all_gaussian_info"] = {
        "objective_initial": res_gaussian_full.objective_initial,
        "objective_final": res_gaussian_full.objective_final,
        "build_time_s": res_gaussian_full.build_time_s,
        "solve_time_s": res_gaussian_full.solve_time_s,
        "nfev": res_gaussian_full.nfev,
        "solver_used": res_gaussian_full.solver_used,
        "success": res_gaussian_full.success,
    }
    RESULTS["all_robust_info"] = {
        "objective_initial": res_robust.objective_initial,
        "objective_final": res_robust.objective_final,
        "build_time_s": res_robust.build_time_s,
        "solve_time_s": res_robust.solve_time_s,
        "nfev": res_robust.nfev,
        "solver_used": res_robust.solver_used,
        "irls_iterations": res_robust.irls_iterations,
        "robust_converged": res_robust.robust_converged,
    }
    RESULTS["huber_delta"] = 2.5
    RESULTS["ground_truth_policy"] = {
        "loaded_after_all_optimization_and_model_analysis": True,
        "used_only_for_final_metrics_and_comparison_figures": True,
    }
    RESULTS["total_runtime_sec"] = time.perf_counter() - t_start

    with open(ROOT / "results" / "results.json", "w", encoding="utf-8") as f:
        json.dump(clean(RESULTS), f, ensure_ascii=False, indent=2)

    trajectory_dir = ROOT / "results" / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    for name, trajectory in traj_by_model.items():
        pd.DataFrame(trajectory, columns=["x", "y", "theta"]).to_csv(
            trajectory_dir / f"{name}.csv",
            index=False,
        )

    print(
        f"\nDONE in {RESULTS['total_runtime_sec']:.1f}s "
        "-> results/figures, results/tables, results/trajectories, results/results.json"
    )


if __name__ == "__main__":
    main()
