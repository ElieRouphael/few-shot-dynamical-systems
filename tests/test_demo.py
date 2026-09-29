from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynafm_demo import DemoConfig, RealisticConfig, run_experiment, run_realistic_experiment
from dynafm_demo.system import parameters_from_context


class DemoTests(unittest.TestCase):
    def test_parameter_family_is_positive(self):
        for c1 in (-1.0, 1.0):
            for c2 in (-1.0, 1.0):
                eta = parameters_from_context(np.array([c1, c2])).as_vector()
                self.assertTrue(np.all(eta > 0.0))

    def test_small_experiment_runs_and_is_stable(self):
        cfg = DemoConfig(
            seed=3,
            training_tasks=8,
            training_horizon=180,
            adaptation_horizon=35,
            validation_horizon=80,
            excitation_candidates=20,
            monte_carlo_runs=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_experiment(directory, cfg)
            self.assertLess(metrics["active_spectral_radius"], 1.0)
            self.assertTrue(np.isfinite(metrics["active_excitation_rmse"]))
            self.assertTrue((Path(directory) / "metrics.json").exists())

    def test_realistic_benchmark_has_true_random_baseline(self):
        cfg = RealisticConfig(
            seed=4,
            training_tasks=6,
            training_horizon=160,
            adaptation_horizon=35,
            validation_horizon=60,
            excitation_candidates=12,
            monte_carlo_runs=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_realistic_experiment(directory, cfg)
            self.assertEqual(
                metrics["assumptions"]["random_baseline"],
                "fresh unscored input in every trial",
            )
            self.assertTrue(np.isfinite(metrics["rmse"]["active"]["mean"]))


if __name__ == "__main__":
    unittest.main()
