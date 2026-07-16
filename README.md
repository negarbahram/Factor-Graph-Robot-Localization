# SE(2) Factor Graph Robot Localization

Two-dimensional robot localization and sensor fusion using a nonlinear factor graph over robot poses in $SE(2)$.

The project demonstrates core probabilistic graphical model concepts, including posterior factorization, local measurement factors, conditional independence, MAP estimation, robust inference, marginal covariance analysis, Markov blankets, and variable-elimination fill-in.

## Key Features

* Explicit nonlinear $SE(2)$ factor-graph construction
* Initial-pose prior factor
* Relative odometry factors between consecutive poses
* Sparse GPS position factors
* Loop-closure factors between non-consecutive poses
* Known-landmark range and bearing factors
* Gaussian MAP estimation using nonlinear weighted least squares
* Robust Huber estimation using iteratively reweighted least squares
* Dead-reckoning and sensor-ablation comparisons
* Initialization and sensor-noise sensitivity analysis
* Marginal covariance estimation and uncertainty ellipses
* Normalized residual diagnostics for inconsistent measurements
* Markov blanket and symbolic fill-in analysis
* Automated tests for data validation, Jacobians, and robust weighting
* Strict separation between estimation and ground-truth evaluation

## Model

The complete robot trajectory is:

```math
X=\{x_0,x_1,\ldots,x_T\},
\qquad
x_t=(x_t,y_t,\theta_t)\in SE(2).
```

Given prior, odometry, GPS, loop-closure, and landmark observations, the posterior is factorized as:

```math
\begin{aligned}
p(X\mid Z)
&\propto
\phi_0(x_0)
\prod_{t=1}^{T}\psi_t(x_{t-1},x_t) \\
&\quad\times
\prod_{m\in G}\gamma_m(x_m)
\prod_{(i,j)\in L}\lambda_{ij}(x_i,x_j)
\prod_{(t,k)\in M}\eta_{tk}(x_t;\ell_k).
\end{aligned}
```

where:

* $x_t$ is the robot pose at time $t$
* $\phi_0$ is the initial-pose prior factor
* $\psi_t$ is the odometry factor between consecutive poses
* $\gamma_m$ is a GPS position factor
* $\lambda_{ij}$ is a loop-closure factor
* $\eta_{tk}$ is a range-bearing observation of known landmark $\ell_k$

Under Gaussian noise assumptions, MAP estimation becomes:

```math
X^\star
=
\arg\min_X
\frac{1}{2}
\sum_{a\in\mathcal{F}}
r_a(X_a;z_a)^{T}
\Sigma_a^{-1}
r_a(X_a;z_a).
```

For inconsistent measurements, the robust model replaces the purely quadratic penalty with Huber loss and solves the resulting problem using iteratively reweighted least squares.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

The project was tested with Python 3.12.2 and the pinned packages in `requirements.txt`.

## Usage

Run the complete experiment:

```bash
python3 scripts/run_project.py
```

The complete pipeline performs:

* dataset loading and validation
* dead-reckoning initialization
* Gaussian MAP estimation
* robust MAP estimation
* sensor-ablation experiments
* initialization-sensitivity experiments
* GPS-noise sensitivity experiments
* covariance and uncertainty analysis
* residual diagnostics
* structural PGM analysis
* final evaluation and visualization

Ground truth is loaded only after all estimation and probabilistic analyses have been completed.

## Data Structure

```text
data/
├── reference/
│   ├── initial_prior.csv
│   ├── odometry.csv
│   ├── gps.csv
│   ├── loop_closures.csv
│   ├── landmarks.csv
│   └── landmark_observations.csv
└── evaluation/
    └── ground_truth.csv
```

`ground_truth.csv` is used only for final metrics and comparison figures. It is not used for graph construction, initialization, covariance tuning, robust weighting, outlier detection, or MAP estimation.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

The tests verify:

* dataset contracts and valid measurement indices
* isolation of ground truth from the estimation pipeline
* analytical Jacobians against finite-difference approximations
* correct Huber-weight behaviour
* fixed weights for prior and odometry factors

## Project Structure

```text
src/robot_pgm/
├── factor_graph.py
├── io_utils.py
├── metrics.py
├── pgm_analysis.py
├── uncertainty.py
└── visualization.py

scripts/
└── run_project.py

starter/
└── se2_utils.py

tests/
└── test_project.py

configs/
└── reference.yaml
```

## Outputs

The program generates:

* optimized trajectories for all model variants
* dead-reckoning, Gaussian, and robust trajectory comparisons
* position-error plots
* loop-closure visualizations
* marginal covariance ellipses
* sensor-ablation tables
* GPS-noise sensitivity tables and figures
* initialization-sensitivity tables and figures
* Gaussian-versus-robust residual diagnostics
* Markov blanket summaries
* variable-elimination and fill-in analysis
* solver diagnostics and runtime information
* machine-readable results in `results/results.json`

Output directories:

```text
results/
├── figures/
├── tables/
├── trajectories/
└── results.json
```

## Reports

* `report-English.md`: complete English report
* `report-Persian.md`: complete Persian report

## Limitations

The model assumes two-dimensional motion, known landmark positions, conditionally independent measurements, fixed sensor covariances, and mostly Gaussian inlier noise. It does not perform loop-closure detection, landmark data association, sensor calibration, or full SLAM with unknown landmarks.

The robust Huber model reduces the influence of inconsistent measurements but does not explicitly estimate a discrete inlier/outlier variable. Marginal covariance is computed as a local Gaussian approximation around the MAP solution and may not represent strongly nonlinear or multimodal posterior distributions.
