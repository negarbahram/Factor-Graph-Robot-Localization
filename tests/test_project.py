from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "starter"))

import se2_utils  # noqa: E402
from robot_pgm.io_utils import load_dataset  # noqa: E402
from robot_pgm.factor_graph import FactorGraph  # noqa: E402


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = load_dataset(ROOT)
        prior = cls.ds.prior.iloc[0]
        cls.x0 = se2_utils.integrate_odometry(
            np.array([prior.mean_x, prior.mean_y, prior.mean_theta]),
            cls.ds.odometry.to_dict("records"),
        )

    def test_dataset_contract(self):
        self.assertEqual(self.ds.n_poses, 260)
        self.assertEqual(len(self.ds.odometry), 259)
        self.assertEqual(len(self.x0), 260)

    def test_ground_truth_not_smuggled_into_dataset(self):
        self.assertFalse(hasattr(self.ds, "ground_truth"))
        self.assertNotIn("ground_truth", self.ds.__dataclass_fields__)

    def test_analytic_jacobian_matches_finite_differences(self):
        g = FactorGraph(self.ds, include_gps=True, include_loops=True, include_landmarks=True)
        x = self.x0.reshape(-1)
        analytic = g.jacobian(x).toarray()
        rng = np.random.default_rng(0)
        columns = rng.choice(g.n_vars, size=24, replace=False)
        step = 1e-6
        for col in columns:
            dx = np.zeros_like(x)
            dx[col] = step
            numeric = (g.residual_vector(x + dx) - g.residual_vector(x - dx)) / (2 * step)
            err = np.max(np.abs(numeric - analytic[:, col]))
            self.assertLess(err, 1e-4, msg=f"column {col} mismatch: {err}")

    def test_robust_weights_are_one_for_small_residuals_and_less_for_large(self):
        g = FactorGraph(self.ds, include_gps=True, include_loops=True, include_landmarks=True)
        x = self.x0.reshape(-1)
        w = g.robust_weights(x, huber_delta=2.5)
        norms = g.factor_norms(x)
        for k, f in enumerate(g.factors):
            if f.robust_eligible and norms[k] > 2.5:
                self.assertLess(w[k], 1.0)
            else:
                self.assertAlmostEqual(w[k], 1.0, places=6)

    def test_odometry_and_prior_are_never_down_weighted(self):
        g = FactorGraph(self.ds)
        x = self.x0.reshape(-1)
        w = g.robust_weights(x, huber_delta=0.01)  # pathologically small threshold
        for k, f in enumerate(g.factors):
            if f.kind in ("prior", "odometry"):
                self.assertEqual(w[k], 1.0)


if __name__ == "__main__":
    unittest.main()
