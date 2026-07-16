import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle


def plot_factor_graph_schematic(path):
    fig, ax = plt.subplots(figsize=(11, 4))
    xs = [0, 2, 4, 6, 8]
    for k, x in enumerate(xs):
        ax.add_patch(Circle((x, 0), 0.28, facecolor="white", edgecolor="black", zorder=3))
        ax.text(x, 0, f"x{k}", ha="center", va="center", zorder=4, fontsize=11)
    for k in range(len(xs) - 1):
        xm = (xs[k] + xs[k + 1]) / 2
        ax.plot([xs[k], xs[k + 1]], [0, 0], color=f"C{k}", lw=2, zorder=1)
        ax.add_patch(Rectangle((xm - 0.15, -0.15), 0.3, 0.3, facecolor=f"C{k}", zorder=2))
        ax.text(xm, 0.35, rf"$\psi_{k+1}$", ha="center", fontsize=10)
    ax.add_patch(Rectangle((xs[0] - 0.15, -1.15), 0.3, 0.3, facecolor="purple", zorder=2))
    ax.plot([xs[0], xs[0]], [0, -1.0], color="purple", lw=2, zorder=1)
    ax.text(xs[0], -1.45, r"$\phi_0$ prior", ha="center", fontsize=10)
    ax.add_patch(Rectangle((xs[2] - 0.15, -1.15), 0.3, 0.3, facecolor="brown", zorder=2))
    ax.plot([xs[2], xs[2]], [0, -1.0], color="brown", lw=2, zorder=1)
    ax.text(xs[2], -1.45, r"$\gamma$ GPS", ha="center", fontsize=10)
    ax.add_patch(Rectangle((xs[4] - 0.15, -1.15), 0.3, 0.3, facecolor="deeppink", zorder=2))
    ax.plot([xs[4], xs[4]], [0, -1.0], color="deeppink", lw=2, zorder=1)
    ax.text(xs[4], -1.45, r"$\eta$ landmark", ha="center", fontsize=10)
    ax.add_patch(Rectangle((xs[2] - 0.15, 0.95), 0.3, 0.3, facecolor="gray", zorder=2))
    ax.plot([xs[1], xs[1]], [0, 1.1], color="gray", lw=1.6, zorder=1)
    ax.plot([xs[1], xs[2]], [1.1, 1.1], color="gray", lw=1.6, zorder=1)
    ax.plot([xs[2], xs[2]], [1.1, 1.1], color="gray", lw=1.6, zorder=1)
    ax.plot([xs[4], xs[4]], [0, 1.1], color="deepskyblue", lw=1.6, zorder=1)
    ax.plot([xs[2], xs[4]], [1.1, 1.1], color="gray", lw=1.6, zorder=1)
    ax.text((xs[1] + xs[4]) / 2, 1.35, r"$\lambda$ loop closure", ha="center", fontsize=10)
    ax.set_xlim(-1, 9); ax.set_ylim(-2, 2); ax.axis("off")
    ax.set_title("Factor-graph schematic (bipartite: circles=variables, squares=factors)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_trajectory_comparison(gt, dead_reck, optimized, path, title, optimized_label="Optimized (MAP)"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(gt[:, 0], gt[:, 1], "k-", lw=2, label="Ground truth")
    ax.plot(dead_reck[:, 0], dead_reck[:, 1], "r--", lw=1.3, label="Dead reckoning (odometry only)")
    ax.plot(optimized[:, 0], optimized[:, 1], "b-", lw=1.6, label=optimized_label)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(title); ax.legend(loc="best", fontsize=9); ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_error_vs_time(errors_dict, path, title="Position error vs. pose index"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, err in errors_dict.items():
        ax.plot(np.arange(len(err)), err, label=label, lw=1.4)
    ax.set_xlabel("pose id (proxy for time)"); ax.set_ylabel("position error (m)")
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_loop_closures(traj, loop_df, path, title="Optimized trajectory with loop-closure links"):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=1.3, label="Optimized trajectory", zorder=1)
    for r in loop_df.itertuples():
        p1, p2 = traj[r.from_id, :2], traj[r.to_id, :2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="orange", lw=1.0, alpha=0.85, zorder=2)
    ax.scatter(traj[0, 0], traj[0, 1], c="green", s=40, zorder=3, label="start")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(title)
    ax.legend(fontsize=9); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _cov_ellipse_95(ax, mean_xy, cov2x2, **kwargs):
    chi2_95 = 5.991464547107979
    vals, vecs = np.linalg.eigh(cov2x2)
    vals = np.clip(vals, 0, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(chi2_95 * vals)
    ax.add_patch(Ellipse(xy=mean_xy, width=width, height=height, angle=angle, **kwargs))


def plot_covariance_ellipses(traj, cov_dict, path, title="95%-confidence position-uncertainty ellipses",
                              landmarks_xy=None):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=1.0, alpha=0.6, zorder=1)
    if landmarks_xy is not None:
        ax.scatter(landmarks_xy[:, 0], landmarks_xy[:, 1], marker="^", c="green", s=50, label="landmarks", zorder=2)
    for pid, C in cov_dict.items():
        C2 = C[:2, :2] if np.asarray(C).shape == (3, 3) else C
        _cov_ellipse_95(ax, traj[pid, :2], C2, facecolor="none", edgecolor="red", lw=1.5, zorder=3)
        ax.scatter([traj[pid, 0]], [traj[pid, 1]], c="red", s=15, zorder=4)
        ax.annotate(str(pid), (traj[pid, 0], traj[pid, 1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title(title)
    ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_sparsity(Lambda_a, Lambda_b, path, title_a, title_b, suptitle):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, M, t in zip(axes, [Lambda_a, Lambda_b], [title_a, title_b]):
        ax.spy(M, markersize=0.4)
        ax.set_title(f"{t}\nnnz={np.count_nonzero(M)}")
    fig.suptitle(suptitle)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def bar_plot(labels, values, ylabel, title, path, rotate=25, color="steelblue"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(labels, values, color=color)
    ax.set_ylabel(ylabel); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def grouped_bar_plot(categories, series_dict, ylabel, title, path, rotate=45):
    fig, ax = plt.subplots(figsize=(12, 5))
    n = len(series_dict)
    width = 0.8 / n
    x = np.arange(len(categories))
    for i, (label, vals) in enumerate(series_dict.items()):
        ax.bar(x + i * width - 0.4 + width / 2, vals, width=width, label=label)
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=rotate, ha="right", fontsize=7)
    ax.set_ylabel(ylabel); ax.set_title(title); ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def fill_in_bar_plot(ordering_names, fill_edges, max_cliques, path,
                      title="Symbolic fill-in by elimination ordering"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(ordering_names, fill_edges, color="indianred")
    axes[0].set_title("Fill edges added during elimination"); axes[0].set_ylabel("# fill edges")
    axes[1].bar(ordering_names, max_cliques, color="seagreen")
    axes[1].set_title("Largest induced clique size"); axes[1].set_ylabel("nodes in largest clique")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_noise_sensitivity_trajectories(
    ground_truth_xy,
    dead_reckoning_xy,
    gps_xy,
    sensitivity_trajectories,
    sensitivity_summary,
    path,
    title="Trajectory sensitivity to GPS noise scale",
):
    fig, ax = plt.subplots(figsize=(8.5, 7))

    ax.plot(
        ground_truth_xy[:, 0],
        ground_truth_xy[:, 1],
        linewidth=2.2,
        label="Ground truth",
        zorder=5,
    )

    ax.plot(
        dead_reckoning_xy[:, 0],
        dead_reckoning_xy[:, 1],
        linestyle="--",
        linewidth=1.2,
        alpha=0.75,
        label="Dead reckoning",
        zorder=1,
    )

    if gps_xy is not None and len(gps_xy) > 0:
        ax.scatter(
            gps_xy[:, 0],
            gps_xy[:, 1],
            marker="x",
            s=24,
            alpha=0.55,
            label="GPS measurements",
            zorder=2,
        )

    rmse_lookup = {
        float(row.gps_sigma_scale): float(row.rmse_ate_m)
        for row in sensitivity_summary.itertuples(index=False)
    }

    descriptions = {
        0.2: "overconfident",
        1.0: "nominal",
        5.0: "underconfident",
    }

    for scale in sorted(sensitivity_trajectories):
        trajectory = sensitivity_trajectories[scale]

        label = (
            f"GPS sigma x{scale:g} "
            f"({descriptions.get(float(scale), '')}), "
            f"RMSE={rmse_lookup[float(scale)]:.3f} m"
        )

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            linewidth=1.7,
            label=label,
            zorder=3,
        )

    ax.scatter(
        ground_truth_xy[0, 0],
        ground_truth_xy[0, 1],
        marker="o",
        s=65,
        label="Start",
        zorder=7,
    )

    ax.scatter(
        ground_truth_xy[-1, 0],
        ground_truth_xy[-1, 1],
        marker="s",
        s=65,
        label="End",
        zorder=7,
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def plot_noise_sensitivity_error_vs_pose(
    dead_reckoning_error,
    sensitivity_errors,
    path,
    title="Position-error sensitivity to GPS noise scale",
):
    """Show full-scale and zoomed position errors."""

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        sharex=True,
    )

    pose_ids = np.arange(len(dead_reckoning_error))

    labels = {
        0.2: "GPS sigma x0.2 (overconfident)",
        1.0: "GPS sigma x1 (nominal)",
        5.0: "GPS sigma x5 (underconfident)",
    }

    for ax in axes:
        ax.plot(
            pose_ids,
            dead_reckoning_error,
            linestyle="--",
            linewidth=1.2,
            alpha=0.7,
            label="Dead reckoning",
        )

        for scale in sorted(sensitivity_errors):
            error = np.asarray(
                sensitivity_errors[scale],
                dtype=float,
            )

            ax.plot(
                np.arange(len(error)),
                error,
                linewidth=1.5,
                label=labels.get(
                    float(scale),
                    f"GPS sigma x{scale:g}",
                ),
            )

        ax.set_ylabel("Position error (m)")
        ax.grid(alpha=0.3)

    axes[0].set_title(f"{title} — full scale")
    axes[0].legend(fontsize=8)

    all_sensitivity_errors = np.concatenate([
        np.asarray(error, dtype=float)
        for error in sensitivity_errors.values()
    ])

    zoom_limit = max(
        0.5,
        float(
            np.quantile(
                all_sensitivity_errors,
                0.98,
            ) * 1.15
        ),
    )

    axes[1].set_ylim(0, zoom_limit)
    axes[1].set_title("Zoomed view of optimized trajectories")
    axes[1].set_xlabel("Pose id")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def plot_noise_sensitivity_tradeoff(
    sensitivity_df,
    path,
    title="Accuracy–uncertainty trade-off under GPS noise misspecification",
):
    """Compare actual trajectory error with reported posterior uncertainty."""

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ellipse_areas = sensitivity_df[
        "mean_ellipse_area_95_m2"
    ].to_numpy(float)

    rmse_values = sensitivity_df[
        "rmse_ate_m"
    ].to_numpy(float)

    scales = sensitivity_df[
        "gps_sigma_scale"
    ].to_numpy(float)

    ax.scatter(
        ellipse_areas,
        rmse_values,
        s=95,
        zorder=3,
    )

    descriptions = {
        0.2: "overconfident",
        1.0: "nominal",
        5.0: "underconfident",
    }

    for area, rmse, scale in zip(
        ellipse_areas,
        rmse_values,
        scales,
    ):
        ax.annotate(
            f"{scale:g}x\n{descriptions.get(float(scale), '')}",
            (area, rmse),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=9,
        )

    nominal = sensitivity_df[
        np.isclose(
            sensitivity_df["gps_sigma_scale"],
            1.0,
        )
    ]

    if not nominal.empty:
        nominal_area = float(
            nominal["mean_ellipse_area_95_m2"].iloc[0]
        )

        nominal_rmse = float(
            nominal["rmse_ate_m"].iloc[0]
        )

        ax.axvline(
            nominal_area,
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

        ax.axhline(
            nominal_rmse,
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

    ax.set_xlabel("Mean 95% covariance-ellipse area (m²)")
    ax.set_ylabel("Position RMSE / ATE (m)")
    ax.set_title(title)
    ax.grid(alpha=0.3)

    ax.text(
        0.02,
        0.98,
        "Lower-left is desirable:\nlow actual error and low uncertainty",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
        },
    )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

def plot_initialization_sensitivity(
    initialization_df,
    path,
    title="Sensitivity of the complete models to initialization",
):
    """Compare final accuracy and objective across deterministic initializations."""
    order = [
        "dead_reckoning",
        "mild_smooth_perturbation",
        "strong_smooth_perturbation",
    ]
    labels = ["Dead reckoning", "Mild perturbation", "Strong perturbation"]
    models = ["all_gaussian", "all_robust"]
    model_labels = {"all_gaussian": "All Gaussian", "all_robust": "All robust"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(order))
    width = 0.36

    for offset_index, model in enumerate(models):
        subset = (
            initialization_df[initialization_df["model"] == model]
            .set_index("initialization")
            .reindex(order)
        )
        offset = (offset_index - 0.5) * width
        rmse_bars = axes[0].bar(
            x + offset,
            subset["rmse_ate_m"].to_numpy(float),
            width=width,
            label=model_labels[model],
        )
        objective_bars = axes[1].bar(
            x + offset,
            subset["objective_final"].to_numpy(float),
            width=width,
            label=model_labels[model],
        )
        axes[0].bar_label(rmse_bars, fmt="%.3f", padding=2, fontsize=8)
        axes[1].bar_label(objective_bars, fmt="%.1f", padding=2, fontsize=8)

    axes[0].set_ylabel("RMSE / ATE (m)")
    axes[0].set_title("Final trajectory accuracy")
    axes[1].set_ylabel("Final objective")
    axes[1].set_title("Final optimization objective")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
