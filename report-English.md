# 2D Robot Localization and Sensor Fusion Using a MAP Model on a Factor Graph
## Comprehensive Project Report

**Course:** Probabilistic Graphical Models, Ferdowsi University of Mashhad &nbsp;|&nbsp; **Instructor:** Dr. Ahad Harati &nbsp;|&nbsp; Academic Year 1404–1405

The complete code is located in `src/robot_pgm/`, the execution script is in `scripts/run_project.py`, and the tests are in `tests/test_project.py`. **All numbers, tables, and figures in this report were generated directly by running this code** — no number was entered manually and no figure was edited manually.

---

## Table of Contents

0. Implementation architecture and how the code works
1. Problem definition, variables, observations, and measurement conventions
2. Graphical model, posterior factorization, factor scopes, and conditional independencies
3. Solution method, tools, initialization, noise model, and settings
4. Experiments (ablation, noise sensitivity, loop closure, inconsistent data, robust model)
5. Numerical and graphical results
6. Uncertainty analysis and covariance ellipses
7. PGM discussion (Markov blanket, marginalization, sparsity, latent reliability variable)
8. Limitations, failure cases, and exact execution guide

---

## 0. Implementation Architecture and How the Code Works

### Overall Structure

```text
robot/
├── configs/reference.yaml       # Data conventions and nominal noise model
├── data/                        # Reference dataset (prior, odometry, GPS, loops, landmarks, ground truth)
├── starter/se2_utils.py         # Provided SE(2) utilities (angle wrapping, pose composition/difference)
├── src/robot_pgm/
│   ├── io_utils.py              # Complete data loading and validation
│   ├── factor_graph.py          # PGM core: factors, analytical Jacobians, MAP solution, robust IRLS, covariance
│   ├── pgm_analysis.py          # Markov blanket, fill-in simulation, min-degree ordering
│   ├── evaluation.py            # Final evaluation metrics; no graph construction or tuning
│   └── plotting.py              # All figures
├── requirements.txt              # Exact package versions for the reference run
├── README.md                     # Concise installation, test, and execution guide
├── scripts/run_project.py       # Complete end-to-end execution
├── tests/test_project.py        # Unit tests (Jacobians, data conventions, robust-weight behavior)
└── results/                     # Outputs: figures/, tables/, results.json, trajectories/
```

### Execution Flow (`run_project.py`)

1. `load_dataset` reads and validates all CSV and YAML files (Section 1 provides complete details).
2. `se2_utils.integrate_odometry` constructs the dead-reckoning trajectory — this is both the evaluation baseline and the common initial value for all subsequent models.
3. For each sensor combination (`FactorGraph(ds, include_gps=..., include_loops=..., include_landmarks=...)`), a new factor graph is created and solved using `.solve()`.
4. The complete model is solved once more using `robust=True`, initialized from the **same common dead-reckoning trajectory** used by every other model.
5. `pgm_analysis` is applied to the pose graph: the Markov blanket is computed for two sample poses, and fill-in is simulated for three variable-elimination orderings.
6. After all Gaussian and robust solves, sensitivity runs, covariance calculations, and PGM structural analyses are complete, `run_project.py` loads `ground_truth.csv` through the dedicated `load_ground_truth_for_evaluation_only` function and passes it to `evaluation` only for final metrics and comparison figures.
7. `plotting` generates all figures; everything is stored in `results/`.

### Computational Core: The `FactorGraph` Class

Each row of every CSV file is converted into a `Factor` object containing: type (`prior`/`odometry`/`gps`/`loop`/`landmark`), identifier, scope (the list of pose identifiers connected to it), measured value, and $\sigma$. Two functions are implemented for each factor:

- **Residual:** exactly the same $h(\cdot)-z$ described by the formulas in Section 3, with `wrap` applied to the angular component.
- **Analytical Jacobian:** the exact derivative of the same function with respect to every pose in its scope (the formulas are given later in Section 3). These Jacobians were verified by a unit test (`test_analytic_jacobian_matches_finite_differences`) against central finite differences over 24 randomly selected columns of the state vector; the maximum observed error was below $10^{-4}$.

The residual of every factor is whitened using its own $\sigma$, and all factors are stacked into one large residual vector and one sparse Jacobian matrix (only the columns associated with the poses in each factor's scope are nonzero). Numerical optimization is performed using `scipy.optimize.least_squares` on this residual vector and Jacobian matrix.

### Why This Architecture?

The code is deliberately designed around the idea of a **factor as a first-class unit** — an explicit `Factor` object with a defined scope, rather than merely a row in a large matrix. This choice directly reflects the **factor-graph** representation used in probabilistic graphical models: instead of defining the model through preconstructed cliques on a graph, the model is built directly from a collection of local factors, and the underlying graph (which poses are related to one another) is induced by the factors' own scopes. This data structure also allows the analyses in Section 7 (Markov blanket and fill-in) to be extracted directly from the same factor definitions without writing a separate model representation.

---

## 1. Problem Definition, Variables, Observations, and Measurement Conventions

**Latent variable.** The robot trajectory is $X=\{x_0,\dots,x_{259}\}$ with $x_t=(x_t,y_t,\theta_t)\in SE(2)$; there are 260 poses, each with three dimensions, producing a total state-vector dimension of 780.

**Observations.** There are 401 observational factors in total:

| Set | File | Count | Factor scope | Nominal noise model ($\sigma$) |
|---|---|---:|---|---|
| prior | `initial_prior.csv` | 1 | $x_0$ | $(0.150,\,0.150,\,0.080)$ |
| odometry | `odometry.csv` | 259 | $(x_{t-1},x_t)$ | $(0.055,\,0.035,\,0.018)$ |
| GPS | `gps.csv` | 33 | $x_t$ | $(0.62,\,0.62)$ |
| loop closure | `loop_closures.csv` | 8 | $(x_i,x_j)$, $j-i\gg1$ | $(0.16,\,0.16,\,0.045)$ |
| landmark | `landmark_observations.csv` | 100 | $x_t$ (landmark $\ell_k$ is fixed and known, not a latent variable) | $(0.22,\,0.045)$ range-bearing |

**Prediction functions** (exactly as specified and implemented in `factor_graph.py`):

$$h_{\text{rel}}(x_i,x_j)=\begin{bmatrix}R(\theta_i)^T(p_j-p_i)\\ \operatorname{wrap}(\theta_j-\theta_i)\end{bmatrix},\qquad
h_{\text{gps}}(x_t)=\begin{bmatrix}x_t\\y_t\end{bmatrix},\qquad
h_{\text{lm}}(x_t,\ell_k)=\begin{bmatrix}\sqrt{(\ell_k^x-x_t)^2+(\ell_k^y-y_t)^2}\\ \operatorname{wrap}(\operatorname{atan2}(\ell_k^y-y_t,\ell_k^x-x_t)-\theta_t)\end{bmatrix}$$

The residual of each factor is $r=h(\cdot)-z$, with `wrap` applied to the angular residual component.

**Data validation before any modeling.** `io_utils.validate_dataset` checks: the presence of all required columns; finiteness of all numerical values; the existence of exactly one prior on pose zero; exact odometry-chain connectivity ($0\to1\to\dots\to259$, with no missing transition); validity and range of every `pose_id`/`from_id`/`to_id` in the GPS, loop, and landmark files; valid `landmark_id` references (every observed landmark must be defined in `landmarks.csv`); and positivity of every $\sigma$. If any condition is violated, loading stops with an error — therefore, no factor graph is ever constructed from invalid data.

**Rule for using the reference trajectory.** All graph construction, parameter choices, Gaussian and robust optimization, residual diagnostics, noise-sensitivity solves, covariance calculations, and PGM structural analyses are completed before the reference trajectory is loaded. Only then does `run_project.py` call the dedicated `load_ground_truth_for_evaluation_only` function and pass the result to `evaluation.py` solely for final metrics and comparison figures. The loader is never imported into `factor_graph.py` or `pgm_analysis.py`, the `Dataset` object has no ground-truth field, and `test_ground_truth_not_smuggled_into_dataset` explicitly verifies this structural separation.

---

## 2. Graphical Model, Posterior Factorization, Factor Scopes, and Conditional Independencies

The posterior distribution is factorized as follows:

$$p(X\mid Z)\ \propto\ \phi_0(x_0)\ \prod_{t=1}^{259}\psi_t(x_{t-1},x_t)\ \prod_{m\in G}\gamma_m(x_m)\ \prod_{(i,j)\in L}\lambda_{ij}(x_i,x_j)\ \prod_{(t,k)\in M}\eta_{tk}(x_t;\ell_k)$$

where $\phi_0\leftrightarrow$ prior, $\psi_t\leftrightarrow$ odometry, $\gamma_m\leftrightarrow$ GPS, $\lambda_{ij}\leftrightarrow$ loop closure, and $\eta_{tk}\leftrightarrow$ landmark. **Assumed conditional independence:** given the complete trajectory $X$, the observations are independent of one another (sensor noises are assumed independent), and each factor connects to the graph only through its own scope of one or two nodes.

Assuming Gaussian noise for every factor, $-\log p(X\mid Z)$ (up to an additive constant) is exactly the sum of Mahalanobis distances of the residuals. Therefore, MAP estimation is equivalent to:

$$X^\star=\arg\min_X\sum_{a\in\mathcal F}\rho_a\Big(\lVert r_a(X_a;z_a)\rVert^2_{\Sigma_a^{-1}}\Big),\qquad \rho_a(s)=s \text{ for the standard Gaussian model}$$

### Pose Graph and Markov Structure

The underlying Markov graph between poses is formed only by binary factors (odometry and loop closure) — each row in either of these files adds one edge between two poses. Unary factors (GPS, landmark, prior) add no edge; they only modify the local potential of one pose. This graph is constructed by `pgm_analysis.pose_adjacency` directly from the data files and forms the basis of all analyses in Section 7.

### Markov Blanket — Two Real Examples

The Markov blanket of a pose is the union of the scopes of all factors connected to it, excluding the pose itself. Conditioned on this blanket, the pose is conditionally independent of the remainder of the trajectory — because any factor whose scope does not contain that pose no longer depends on it once the blanket is conditioned upon.

**Pose 0** (actual output of `markov_blanket`):

```json
{"neighbor_pose_ids": [1, 84],
 "odometry_factors": ["odom(0,1)"], "loop_factors": ["loop_0(0,84)"],
 "gps_factors": ["gps_0"], "landmark_factors": ["lm_0", "lm_1"]}
```

Thus, conditioned on $\{x_1,x_{84}\}$ (together with its own unary observations), $x_0$ is conditionally independent of the other 258 states.

**Pose 118** (one endpoint of a loop closure):

```json
{"neighbor_pose_ids": [117, 119, 228],
 "odometry_factors": ["odom(117,118)", "odom(118,119)"],
 "loop_factors": ["loop_7(118,228)"], "gps_factors": [], "landmark_factors": []}
```

This pose has no unary factor (neither GPS nor landmark); its blanket is entirely structural. The only way absolute information, rather than merely relative information, can reach this pose is through that single loop closure — an important point in Section 4 because this particular loop closure is highly inconsistent.

---

## 3. Solution Method, Tools, Initialization, Noise Model, and Settings

**Numerical tool.** As explicitly permitted by the project specification (“it is allowed to use a ready-made numerical tool such as GTSAM, g2o, or scipy.optimize.least_squares”), `scipy.optimize.least_squares` is used as the optimizer; no Gauss–Newton or Levenberg–Marquardt solver was implemented from scratch. What was implemented manually is the model itself: factors, scopes, residuals, **analytical Jacobians**, the noise model, and robustification logic.

**Initial value $X_0$.** The same value is used for every model: raw odometry is chained using `se2_utils.integrate_odometry` (a provided dataset utility), starting from the prior mean. Ground truth is never used.

**Internal linear-solver selection based on subproblem size and determination:**

| Subproblem | Residual-to-variable ratio | Solver |
|---|---|---|
| `odometry_only`, `odometry_loops` (nearly exactly determined) | Number of residuals approximately equals number of variables | Dense Levenberg–Marquardt |
| `odometry_gps`, `odometry_landmarks`, `all_gaussian`, `all_robust` (well determined) | Number of residuals substantially exceeds number of variables | Trust-region reflective + sparse linear solution (`lsmr`) using the true Jacobian sparsity pattern |

This choice changes only the linear subproblem solved within each Gauss–Newton iteration and leaves the **model, factors, $\Sigma$ values, and initial value exactly identical across all experiments**. The justification is purely numerical: in nearly determined subproblems, the approximate Krylov solution (`lsmr`) does not provide sufficient step-direction accuracy, whereas an exact dense linear solution is both accurate and inexpensive at this scale (≤780 variables). With this selection, all six Gaussian models, from dead reckoning to `all_gaussian`, converge successfully with `success=True`.

**Reference timing for the complete Gaussian model:** initial objective $120{,}755.2456$; final objective $10{,}446.4579$; 60 function evaluations; graph-build time 0.0013 s and solve time 0.5913 s. Runtime is hardware- and load-dependent; exact values for every run are recorded in `results/tables/ablation_metrics.csv` and `results/results.json`.

**Nominal noise model:** used unchanged from `configs/reference.yaml` (Table in Section 1).

---

## 4. Experiments

### 4.1 Sensor-Component Ablation

All models were initialized from the same $X_0$ (dead reckoning):

| Model | RMSE (m) | Final drift (m) | Solver | nfev | Converged? |
|---|---:|---:|---|---:|---|
| dead reckoning | 8.921 | 12.212 | - | - | - |
| odometry_only | 8.921 | 12.212 | dense LM | 2 | ✓ |
| odometry_gps | 1.341 | 0.431 | trf/lsmr | 120 | ✓ |
| odometry_loops | 5.734 | 8.613 | dense LM | 26 | ✓ |
| odometry_landmarks | 0.266 | 0.077 | trf/lsmr | 15 | ✓ |
| all_gaussian | 1.243 | 0.034 | trf/lsmr | 60 | ✓ |
| **all_robust** | **0.155** | 0.029 | trf/lsmr + IRLS | 143 | ✓ |

![ablation](results/figures/ablation_rmse.png)

The reference-run timing table shows that graph construction takes only a few milliseconds; numerical optimization dominates the computational cost:

| Model | Build time (s) | Solve time (s) | nfev |
|---|---:|---:|---:|
| odometry_only | 0.0006 | 0.1835 | 2 |
| odometry_gps | 0.0008 | 0.9946 | 120 |
| odometry_loops | 0.0009 | 4.3961 | 26 |
| odometry_landmarks | 0.0025 | 0.2565 | 15 |
| all_gaussian | 0.0013 | 0.5913 | 60 |
| all_robust | 0.0014 | 1.3965 | 143 |
Landmarks are the most effective individual sensor source (RMSE≈0.27 m) because they provide both range and bearing relative to an absolute reference. Loop closures alone, together with odometry, provide the smallest improvement (RMSE≈5.73 m), because two of the eight loop closures are highly inconsistent (Section 4.3), and in the absence of other sensors that could balance this inconsistency, their negative effect dominates.

### 4.2 Sensitivity to GPS Noise

Three settings over a wide range ($0.2\times$, $1\times$, and $5\times$ the nominal $\sigma_{\text{gps}}$) were examined so that the effect of overconfidence versus underconfidence — explicitly requested in the project specification — could be clearly observed:

| $\sigma_{\text{gps}}$ | RMSE (m) | Final drift (m) | Final objective | Mean 95% ellipse area (m²) |
|---|---:|---:|---:|---:|
| $0.2\times$ (overconfident) | **1.426** | 0.331 | 13751.16 | **0.109** |
| $1\times$ (nominal) | 1.243 | 0.034 | 10446.46 | 0.133 |
| $5\times$ (less confidence) | 1.258 | 0.014 | 10221.59 | 0.134 |

![sensitivity](results/figures/noise_sensitivity_rmse.png)

![Trajectory sensitivity to GPS noise](results/figures/noise_sensitivity_trajectories.png)

Changing the assumed GPS covariance alters not only the numerical RMSE but also the estimated trajectory. At 0.2× nominal sigma, the graph becomes overconfident in GPS and the inconsistent measurement exerts substantially greater influence.

![Position-error sensitivity to GPS noise](results/figures/noise_sensitivity_error_vs_pose.png)

The effect of covariance misspecification is not uniform along the trajectory. The largest differences occur in regions affected by inconsistent measurements.

![Accuracy–uncertainty trade-off](results/figures/noise_sensitivity_accuracy_uncertainty.png)

The accuracy–uncertainty plot shows that the 0.2× model reports a smaller covariance ellipse despite having a larger true RMSE. It is therefore simultaneously less accurate and more confident in its incorrect estimate.
The key observation is that reducing $\sigma_{\text{gps}}$ makes the **true RMSE worse (1.243→1.426), while the reported ellipse area becomes smaller (0.133→0.109)**. In other words, the model simultaneously becomes *more wrong* and *more confident in itself*. This happens because one outlying GPS measurement (`meas_id=28`, pose 231) receives extremely high weight when $\sigma$ is small and forcibly pulls the trajectory toward itself. This phenomenon shows that the covariance reported by a Gaussian model is only as valid as the agreement between its noise model and reality.

### 4.3 Inconsistent Data — Which Factors Produced the Largest Inconsistency?

Without using any outlier labels or ground truth, the whitened residuals of the Gaussian solution were:

| Factor | Whitened norm in the Gaussian solution |
|---|---:|
| `loop_7 (pose 118→228)` | **103.6** |
| `lm_45 (pose 120, landmark 3)` | 34.5 |
| `lm_44 (pose 115, landmark 3)` | 28.6 |
| `loop_6 (pose 42→146)` | 24.0 |
| `lm_89 (pose 230)` | 23.9 |

The sum of squared whitened residuals in the complete Gaussian solution is approximately $2\times10{,}446.5\approx20{,}892$ over 1,070 residual components (mean ≈19.5, compared with an expected value of 1 for a model fully consistent with its noise assumptions) — a clear sign of systematic inconsistency in part of the data. Only about 15 factors out of 401 account for more than half of this total.

**Why can these factors not simply be deleted?** There are no outlier labels in the student data, and thresholding based on ground truth would violate the project rules. The adopted solution is **soft gating based on normalized residuals from the same run** (Section 4.4).

### 4.4 Robust Model: IRLS-Huber on the Whole-Factor Norm

In the MAP objective, $\rho_a$ is defined over the **whole-factor norm** ($\lVert r_a\rVert_{\Sigma_a^{-1}}$), not over individual numerical components. For this reason, robustification is implemented through an IRLS cycle that operates on the same vector norm:

$$w_a=\begin{cases}1 & \lVert r_a\rVert_{\Sigma_a^{-1}}\le\delta\\ \delta/\lVert r_a\rVert_{\Sigma_a^{-1}} & \text{otherwise}\end{cases}\qquad(\delta=2.5)$$

Using the current weights, a reweighted Gaussian subproblem is solved ($r_a\to\sqrt{w_a}\,r_a$), after which the weights are updated from the new solution. This process is repeated until both the weights and the state stabilize. In the reference run, the robust model was initialized **directly from the same dead-reckoning trajectory used by all models** and converged after 14 outer iterations. Its improvement therefore cannot be attributed to a Gaussian warm start.

**Result:** RMSE decreased from 1.243 m for the Gaussian model to **0.155 m** for the robust model:

![Gaussian versus robust RMSE](results/figures/gaussian_vs_robust_rmse.png)

| Factor | Norm, Gaussian solution | Norm, robust solution | Final weight |
|---|---:|---:|---:|
| `loop_7(118,228)` | 103.6 | 165.4 | **0.015** |
| `loop_6(42,146)` | 24.0 | 35.6 | **0.070** |
| `lm_88(pose 230)` | 21.3 | 16.8 | 0.149 |
| `lm_89(pose 230)` | 23.9 | 3.3 | 0.758 |
| `lm_45(pose 120)` | 34.5 | 1.4 | 1.0 |
| `lm_44(pose 115)` | 28.6 | 0.4 | 1.0 |

![Gaussian vs robust residuals](results/figures/gaussian_vs_robust_residuals.png)

An important point is that the weight of a factor is not permanent. `lm_45` and `lm_44` appeared highly inconsistent in the Gaussian solution, but once the trajectory was corrected around the two bad loop closures, their residuals became small and their weights returned to 1. These two landmark factors were actually “victims” of the two inconsistent loop closures rather than independently faulty measurements. The increase in the norm of `loop_6` and `loop_7` in the robust solution is also not a failure: it means the trajectory no longer bends itself to satisfy these two low-weight factors.

### 4.5 Real Sensitivity to Initialization

A separate experiment was added to test initialization sensitivity directly rather than infer it merely from the use of a common starting point. The complete Gaussian and robust models were solved from three deterministic initial trajectories, all constructed without ground truth:

- raw dead reckoning;
- a smooth mild perturbation with position amplitude 0.5 m and heading amplitude 0.05 rad;
- a smooth strong perturbation with position amplitude 2.0 m and heading amplitude 0.20 rad.

The perturbations vary smoothly along the trajectory and are zero at the first and final poses, avoiding unrealistic discontinuities between consecutive poses.

| Initialization | Gaussian RMSE (m) | Gaussian objective | Gaussian nfev | Robust RMSE (m) | Robust objective | Robust nfev |
|---|---:|---:|---:|---:|---:|---:|
| Dead reckoning | 1.242647 | 10446.457855 | 60 | 0.154658 | 906.199336 | 143 |
| Mild smooth perturbation | 1.242636 | 10446.457854 | 70 | 0.154658 | 906.199336 | 153 |
| Strong smooth perturbation | 1.242633 | 10446.457853 | 74 | 0.154658 | 906.199336 | 157 |

![Initialization sensitivity](results/figures/initialization_sensitivity.png)

All six runs converged. The final Gaussian objectives agree to better than $2\times10^{-6}$ and the robust objectives agree to numerical precision. The strong perturbation increases the number of function evaluations, but it does not change the final robust trajectory or its RMSE. Thus, within the tested perturbation range, both complete models converge to the same local solution basin. This is evidence of practical stability for these initializations, not a proof of global convergence for arbitrary starting states. Full values are stored in `results/tables/initialization_sensitivity.csv`.

---

## 5. Numerical and Graphical Results

![factor graph schematic](results/figures/factor_graph_schematic.png)

*Schematics of part of the factor graph: circles are variables (poses), and squares are factors — exactly the bipartite factor-graph representation described in Section 5 of the project specification.*

![trajectory](results/figures/trajectory_comparison.png)

*Dead reckoning (red) rapidly diverges from the reference trajectory (black) due to accumulated odometry drift (RMSE≈8.92 m). The robust optimized trajectory (blue) lies almost directly on the reference trajectory.*

![error vs time](results/figures/error_vs_time.png)

*Position error for four models. Under the robust model (red), the error remains nearly flat and close to zero throughout the trajectory; the local peaks of the Gaussian model (green, around poses 118 and 230) align exactly with the two inconsistent loop closures and are almost eliminated by the robust model.*

![loop closures](results/figures/loop_closures.png)

*Orange connections show loop closures over the optimized trajectory.*

The complete ablation, noise-sensitivity, initialization-sensitivity, and Gaussian/robust comparison tables are available respectively in `results/tables/ablation_metrics.csv`, `noise_sensitivity.csv`, `initialization_sensitivity.csv`, and `gaussian_vs_robust.csv` (including RMSE, final drift, relative error, angular error (`heading RMSE`), objective value, execution time, and iteration count).

Additional metrics reported in the tables are `heading_rmse_rad` (full-trajectory angular RMS error against the reference) and `relative_translation/rotation_rmse_m_lag5` (relative-motion error over five-step windows, used to evaluate local trajectory consistency independently of global drift). For the complete model, heading RMSE improved from 0.101 rad in the Gaussian model to 0.032 rad in the robust model; relative rotational error improved from 0.121 to 0.040 rad.

---

## 6. Uncertainty Analysis and Covariance Ellipses

Marginal covariance is computed using a Laplace approximation around the MAP point: $\Lambda=J^\top J$ (the Gauss–Newton information matrix formed from whitened residuals). For every selected pose, the corresponding $3\times3$ block of $\Lambda^{-1}$ is extracted. However, instead of densely inverting the entire 780×780 matrix, sparse LU factorization is used to solve only for the required blocks (`selected_pose_covariances`), making the method scalable to much larger graphs.

The uncertainty-ellipse area is computed using the standard confidence-region formula for a two-dimensional Gaussian:
$\text{area}=\pi\,\chi^2_{0.95,2}\sqrt{\det\Sigma_{xy}}$ with $\chi^2_{0.95,2}=5.991$ — therefore, the area corresponds to a statistically meaningful 95% confidence level.

![covariance ellipses](results/figures/covariance_ellipses.png)

| Pose | Odometry only (m²) | Complete Gaussian (m²) | Complete robust (m²) |
|---|---:|---:|---:|
| 0 | 0.42 | 0.135 | 0.136 |
| 130 | 95.6 | 0.122 | 0.122 |
| 230 | 211.4 | 0.125 | 0.161 |
| 259 | 143.0 | 0.450 | 0.448 |

The three- to four-order-of-magnitude reduction in ellipse area from the odometry-only model to the complete model shows how strongly GPS, landmarks, and loop closures constrain the graph. Among the selected poses, the odometry-only model reaches its largest ellipse at pose 230 (211.357 m²). In the complete models, the largest selected ellipse is instead at pose 259: 0.450 m² for the Gaussian model and 0.448 m² for the robust model. Pose 259 has a GPS observation, but it has no direct landmark observation or loop-closure endpoint; as the final state, it is connected to odometry only from its predecessor, and GPS does not directly observe heading. Position–heading coupling therefore leaves more uncertainty at this endpoint than at the other selected poses.

In summary, **a loop closure** reduces uncertainty across the entire segment between its two endpoints because it creates a direct constraint between two states far apart in time. **GPS** reduces only the local positional uncertainty, not the heading uncertainty, at its own pose. A **landmark**, because it provides both range and relative bearing, behaves like a weaker GPS measurement but with additional angular information.

---

## 7. PGM Discussion: Markov Blanket, Sparsity, Variable-Elimination Ordering, and a Latent Reliability Variable

### Sparsity of the Information Matrix

The Gauss–Newton information matrix $\Lambda=J^\top J$ at the MAP point has dimensions $780\times780$ with 6,074 numerically nonzero entries in the reference result (approximately 1% density) — a direct reflection of the Markov structure described in Section 2: each $x_t$ block is connected only to its temporal neighbors and, when applicable, to the opposite endpoint of a loop closure.

![Information-matrix sparsity before and after reordering](results/figures/information_sparsity.png)

### Actual Fill-In: Simulation of Variable Elimination

To measure the cost of exact inference, variable elimination is simulated directly on the pose graph (`symbolic_fill_in`). Each node is eliminated according to a specified order; whenever a node is removed, all of its remaining neighbors are connected to one another, creating fill-in edges, and the size of the largest clique is recorded. This is exactly the definition of the induced graph in exact inference on probabilistic graphical models.

| Elimination order | Fill-in edges | Largest clique |
|---|---:|---:|
| Temporal ($0\to259$) | 873 | 8 |
| Reverse temporal ($259\to0$) | 873 | 8 |
| **Greedy min-degree** | **255** | **6** |

![fill-in](results/figures/symbolic_fill_in.png)

Each of the eight loop closures adds exactly one edge between two poses far apart in time. Under temporal elimination, removing intermediate poses between the two endpoints of a loop causes their dependence to “spread” across all poses in between, producing heavy fill-in. Greedy min-degree ordering — which removes the node with the fewest remaining neighbors at every step (`greedy_min_degree_order`) — avoids much of this spread: fill-in is reduced to less than one third, and the largest clique decreases from 8 to 6. This demonstrates that the cost of exact inference depends strongly on variable-elimination ordering, and loop closures are precisely the elements that transform the graph from a simple chain, which is extremely inexpensive to eliminate, into a graph with long-range edges.

### Relationship Between the Robust Model and a Latent Reliability Variable

One theoretical way to model outliers is to introduce a latent reliability variable $s_m\in\{0,1\}$ for every measurement and use the mixture model

$$p(z_m\mid X)=\pi\,\mathcal N(r_m;0,\Sigma_m)+(1-\pi)\,\mathcal N(r_m;0,\kappa\Sigma_m),\qquad\kappa\gg1$$

IRLS-Huber can be interpreted heuristically through this latent reliability variable: measurements with large residuals receive lower influence, similarly to observations with lower posterior reliability. However, the weight $w_a=\delta/\lVert r_a\rVert$ is **not the exact posterior responsibility** of the binary Gaussian-mixture model above. The relationship is therefore a probabilistic interpretation and computational approximation rather than an exact EM equivalence. Its cycle of solving with fixed weights, updating weights from new residuals, and repeating until convergence is EM-like, but it does not calculate the soft responsibility $p(s_m\mid Z,X)$.

This relationship is also supported by the numerical evidence in Section 4.4: the same two factors (`loop_6`, `loop_7`) identified as the most inconsistent in Section 4.3 receive the smallest final weights (0.07 and 0.015, corresponding to $s_m\approx0$), whereas factors that only appeared inconsistent because of their proximity to these two loops (`lm_44`, `lm_45`) return to full weight after the trajectory is corrected (corresponding to $s_m\approx1$).

---

## 8. Limitations, Failure Cases, and Exact Execution Guide

### Limitations and Failure Cases

- The two inconsistent loop closures (`loop_6`, `loop_7`) remain the main limitation of the Gaussian model. If many such loop closures are present, even IRLS-Huber may become trapped in an unsuitable local minimum because IRLS is only a local subgradient optimizer, not complete Bayesian inference over $s_m$.
- Marginal covariance is a Laplace approximation around the MAP point. In the presence of strong nonlinearities (long-range loop closures and range-bearing observations), it may underestimate the true uncertainty.
- The Huber threshold ($\delta=2.5$) is a selected hyperparameter rather than a value learned from the data. Section 4.2 showed that an incorrect $\Sigma$ can make a model simultaneously more wrong and more confident in itself; the same risk applies to a poorly selected $\delta$.
- Variable elimination and min-degree ordering are used only for structural analysis, to demonstrate the importance of ordering, not for the numerical MAP solution itself, which is obtained using least squares.
- The assumption of fully independent sensor noise — no systematic odometry bias and no temporal correlation in GPS noise — is rarely exactly valid in a real robot.

### Response to the Final Checklist in the Project Specification

1. **Odometry-only drift** is cumulative in its overall trend, but the position error is neither uniform nor monotonic along the trajectory. In the reference result, the maximum position error is 20.294 m at pose 223, the final drift is 12.212 m, and the full-trajectory RMSE is 8.921 m.
2. **The most effective individual factor type** is the landmark factor (RMSE≈0.27 m); the most effective overall combination is the complete robust model (RMSE≈0.155 m).
3. A **very small $\sigma$** creates false confidence: RMSE becomes worse while the covariance ellipse becomes misleadingly smaller (Section 4.2). A very large $\sigma$ effectively removes the sensor's influence.
4. A **correct loop closure** reduces uncertainty throughout the segment between its endpoints; an **inconsistent loop closure** distorts the trajectory and, in the absence of other sensors, can severely damage the final result (`odometry_loops`: RMSE≈5.73 m).
5. **The greatest uncertainty depends on the model.** Among the selected poses, the odometry-only model has its largest 95% ellipse at pose 230 (211.357 m²), reflecting accumulated chain uncertainty. In the complete models, the largest selected ellipse occurs at pose 259 (0.450 m² Gaussian; 0.448 m² robust). Although pose 259 has GPS, it has no direct landmark observation or loop-closure endpoint, it is the one-sided end of the odometry chain, and GPS does not directly constrain heading; position–heading coupling therefore leaves greater endpoint uncertainty.
6. **Relationship between the robust model and PGM:** IRLS-Huber can be interpreted as a computational approximation with behavior analogous to a latent reliability variable $s_m$, but it is not exact EM for the mixture model (Section 7).
7. **Sensitivity to initialization:** a direct three-initialization experiment was performed for the complete Gaussian and robust models (dead reckoning, mild smooth perturbation, and strong smooth perturbation). All six runs converged to numerically equivalent final solutions; stronger perturbations increased `nfev` but did not change final RMSE or objective materially (Section 4.5).
8. **Potentially violated assumptions:** complete independence of sensor noise, absence of systematic odometry bias, fixed and deterministic landmarks, and lack of temporal correlation in GPS noise.

### Exact Execution Guide

**Reference environment:** Python 3.12.2, `numpy==2.3.5`, `pandas==2.2.3`, `scipy==1.17.0`, `matplotlib==3.10.8`, and `PyYAML==6.0.3`. These versions are pinned in `requirements.txt`.

Run the following commands from the directory that contains the extracted `robot/` folder:

```bash
cd robot
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v      # Validate analytical Jacobians and data conventions
python3 scripts/run_project.py                 # Complete execution and generation of all figures/tables
```

**Outputs:**

- Figures: `results/figures/*.png`
- Tables: `results/tables/*.csv` (including `markov_blanket_pose0.json`, `symbolic_fill_in.csv`)
- Complete raw numbers: `results/results.json`
- Estimated trajectories for every model: `results/trajectories/*.csv`

In the reference run recorded in `results/results.json`, the complete pipeline required approximately 16.1 seconds. Runtime is hardware-, library-, and system-load dependent. All figures are generated directly by the code without manual editing.
