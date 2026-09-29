from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np

from .model import AdaptationResult, SharedContextModel
from .system import multisine_input, parameters_from_context, simulate


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 12
    dt: float = 0.05
    training_tasks: int = 24
    training_horizon: int = 1200
    adaptation_horizon: int = 25
    validation_horizon: int = 260
    excitation_candidates: int = 160
    monte_carlo_runs: int = 80
    process_std: float = 0.0015
    measurement_std: float = 0.0001
    prior_precision: float = 1.0


def rmse(reference: np.ndarray, estimate: np.ndarray) -> float:
    return float(np.sqrt(np.mean((reference - estimate) ** 2)))


def _adapt_and_score(
    model: SharedContextModel,
    true_eta: np.ndarray,
    adaptation_input: np.ndarray,
    validation_input: np.ndarray,
    x0: np.ndarray,
    rng: np.random.Generator,
    cfg: DemoConfig,
) -> tuple[AdaptationResult, np.ndarray, np.ndarray, float]:
    _, measured = simulate(
        true_eta,
        adaptation_input,
        x0,
        cfg.dt,
        rng,
        cfg.process_std,
        cfg.measurement_std,
    )
    result = model.adapt(measured, adaptation_input, cfg.prior_precision)
    truth, _ = simulate(true_eta, validation_input, x0, cfg.dt)
    prediction = model.rollout(result.eta, validation_input, x0)
    return result, truth, prediction, rmse(truth, prediction)


def run_experiment(output_dir: str | Path, config: DemoConfig | None = None) -> dict:
    cfg = config or DemoConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    x0 = np.array([0.20, 0.0])

    training_data = []
    true_training_eta = []
    for _ in range(cfg.training_tasks):
        context = rng.uniform(-1.0, 1.0, size=2)
        eta = parameters_from_context(context).as_vector()
        controls = multisine_input(cfg.training_horizon, rng, amplitude=1.8)
        _, measured = simulate(
            eta,
            controls,
            x0,
            cfg.dt,
            rng,
            cfg.process_std,
            cfg.measurement_std,
        )
        training_data.append((measured, controls))
        true_training_eta.append(eta)

    model = SharedContextModel(rank=2, dt=cfg.dt).fit(training_data)

    test_context = np.array([0.78, -0.72])
    true_eta = parameters_from_context(test_context).as_vector()
    candidates = [
        multisine_input(cfg.adaptation_horizon, rng, amplitude=1.2)
        for _ in range(cfg.excitation_candidates)
    ]
    training_contexts = (model.task_coefficients - model.mean) @ model.basis
    representative_contexts = training_contexts[:: max(1, len(training_contexts) // 8)]
    information_scores = np.array(
        [model.information_score(u, x0, context_samples=representative_contexts) for u in candidates]
    )
    active_index = int(np.argmax(information_scores))
    median_index = int(np.argsort(information_scores)[len(information_scores) // 2])
    active_input = candidates[active_index]
    random_input = candidates[median_index]
    validation_input = multisine_input(cfg.validation_horizon, rng, amplitude=0.9)

    active_trials = []
    random_trials = []
    active_errors = []
    random_errors = []
    active_coefficient_errors = []
    random_coefficient_errors = []
    for _ in range(cfg.monte_carlo_runs):
        active_trial = _adapt_and_score(model, true_eta, active_input, validation_input, x0, rng, cfg)
        random_trial = _adapt_and_score(model, true_eta, random_input, validation_input, x0, rng, cfg)
        active_trials.append(active_trial)
        random_trials.append(random_trial)
        active_errors.append(active_trial[3])
        random_errors.append(random_trial[3])
        active_coefficient_errors.append(
            np.linalg.norm(active_trial[0].eta - true_eta) / np.linalg.norm(true_eta)
        )
        random_coefficient_errors.append(
            np.linalg.norm(random_trial[0].eta - true_eta) / np.linalg.norm(true_eta)
        )
    active_result, truth, active_prediction, _ = active_trials[0]
    random_result, _, random_prediction, _ = random_trials[0]
    active_rmse = float(np.mean(active_errors))
    random_rmse = float(np.mean(random_errors))
    zero_prediction = model.rollout(model.mean, validation_input, x0)
    zero_rmse = rmse(truth, zero_prediction)

    metrics = {
        "config": asdict(cfg),
        "test_context": test_context.tolist(),
        "true_coefficients": true_eta.tolist(),
        "pretrained_mean_coefficients": model.mean.tolist(),
        "pca_explained_variance_first_two": model.explained_variance[:2].tolist(),
        "active_information_score": float(information_scores[active_index]),
        "median_information_score": float(information_scores[median_index]),
        "no_adaptation_rmse": zero_rmse,
        "random_excitation_rmse": random_rmse,
        "active_excitation_rmse": active_rmse,
        "random_excitation_rmse_std": float(np.std(random_errors)),
        "active_excitation_rmse_std": float(np.std(active_errors)),
        "random_context_covariance_logdet": float(np.linalg.slogdet(random_result.covariance)[1]),
        "active_context_covariance_logdet": float(np.linalg.slogdet(active_result.covariance)[1]),
        "random_coefficient_relative_error": float(np.mean(random_coefficient_errors)),
        "active_coefficient_relative_error": float(np.mean(active_coefficient_errors)),
        "random_coefficient_relative_error_std": float(np.std(random_coefficient_errors)),
        "active_coefficient_relative_error_std": float(np.std(active_coefficient_errors)),
        "random_spectral_radius": random_result.spectral_radius,
        "active_spectral_radius": active_result.spectral_radius,
        "random_nis_mean": random_result.nis_mean,
        "active_nis_mean": active_result.nis_mean,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    arrays = {
        "truth": truth,
        "zero_prediction": zero_prediction,
        "random_prediction": random_prediction,
        "active_prediction": active_prediction,
        "validation_input": validation_input,
        "active_input": active_input,
        "random_input": random_input,
        "information_scores": information_scores,
        "random_rmse_trials": np.asarray(random_errors),
        "active_rmse_trials": np.asarray(active_errors),
        "random_coefficient_error_trials": np.asarray(random_coefficient_errors),
        "active_coefficient_error_trials": np.asarray(active_coefficient_errors),
        "training_coefficients_estimated": model.task_coefficients,
        "training_coefficients_true": np.vstack(true_training_eta),
        "true_eta": true_eta,
        "random_eta": random_result.eta,
        "active_eta": active_result.eta,
    }
    np.savez(output_dir / "experiment_arrays.npz", **arrays)
    return metrics
