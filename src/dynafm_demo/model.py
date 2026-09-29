from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .system import acceleration_targets, regression_features, simulate


@dataclass
class AdaptationResult:
    context: np.ndarray
    eta: np.ndarray
    covariance: np.ndarray
    spectral_radius: float
    stable: bool
    nis_mean: float


class SharedContextModel:
    """PCA task manifold followed by MAP context adaptation.

    Each task has physical coefficient vector eta_i. Pretraining estimates the
    vectors from rich trajectories and learns eta_i ~= mean + basis @ c_i.
    Only c is adapted on a new task.
    """

    def __init__(self, rank: int = 2, ridge: float = 1e-5, dt: float = 0.05):
        self.rank = rank
        self.ridge = ridge
        self.dt = dt
        self.mean: np.ndarray | None = None
        self.basis: np.ndarray | None = None
        self.task_coefficients: np.ndarray | None = None
        self.explained_variance: np.ndarray | None = None
        self.noise_variance: float | None = None
        self.bounds: tuple[np.ndarray, np.ndarray] | None = None

    @staticmethod
    def _ridge_coefficients(phi: np.ndarray, targets: np.ndarray, ridge: float) -> np.ndarray:
        lhs = phi.T @ phi + ridge * np.eye(phi.shape[1])
        return np.linalg.solve(lhs, phi.T @ targets)

    def fit(self, trajectories: list[tuple[np.ndarray, np.ndarray]]) -> "SharedContextModel":
        estimates = []
        residuals = []
        for measured_states, controls in trajectories:
            phi = regression_features(measured_states, controls)
            targets = acceleration_targets(measured_states, self.dt)
            eta = self._ridge_coefficients(phi, targets, self.ridge)
            estimates.append(eta)
            residuals.append(targets - phi @ eta)
        coefficients = np.vstack(estimates)
        self.mean = coefficients.mean(axis=0)
        centered = coefficients - self.mean
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        self.basis = vt[: self.rank].T
        variance = singular_values**2
        self.explained_variance = variance / max(variance.sum(), 1e-12)
        self.task_coefficients = coefficients
        all_residuals = np.concatenate(residuals)
        self.noise_variance = float(max(np.var(all_residuals), 1e-8))
        padding = 0.10 * np.maximum(np.ptp(coefficients, axis=0), 0.05)
        lower = np.maximum(coefficients.min(axis=0) - padding, 1e-3)
        upper = coefficients.max(axis=0) + padding
        self.bounds = (lower, upper)
        return self

    def _check_fitted(self) -> None:
        if self.mean is None or self.basis is None or self.noise_variance is None:
            raise RuntimeError("Call fit before using the shared model.")

    def context_to_eta(self, context: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self.mean + self.basis @ np.asarray(context, dtype=float)

    def spectral_radius(self, eta: np.ndarray) -> float:
        k, d, _, _ = np.asarray(eta, dtype=float)
        # Linearisation of the semi-implicit step at q=v=u=0.
        a = np.array(
            [
                [1.0 - self.dt**2 * k, self.dt * (1.0 - self.dt * d)],
                [-self.dt * k, 1.0 - self.dt * d],
            ]
        )
        return float(np.max(np.abs(np.linalg.eigvals(a))))

    def _stability_project(self, context: np.ndarray, limit: float = 0.9995) -> tuple[np.ndarray, np.ndarray]:
        """Backtrack toward the pretrained mean until the linearisation is stable."""
        trial = np.asarray(context, dtype=float).copy()
        for _ in range(60):
            eta = self.context_to_eta(trial)
            if self.bounds is not None:
                eta = np.clip(eta, self.bounds[0], self.bounds[1])
            if eta[0] > 0.0 and eta[1] > 0.0 and self.spectral_radius(eta) <= limit:
                return trial, eta
            trial *= 0.90
        return np.zeros_like(trial), self.mean.copy()

    def adapt(
        self,
        measured_states: np.ndarray,
        controls: np.ndarray,
        prior_precision: float = 1.0,
    ) -> AdaptationResult:
        self._check_fitted()
        phi = regression_features(measured_states, controls)
        targets = acceleration_targets(measured_states, self.dt)
        design = phi @ self.basis
        centered_targets = targets - phi @ self.mean
        precision = prior_precision * np.eye(self.rank) + design.T @ design / self.noise_variance
        rhs = design.T @ centered_targets / self.noise_variance
        context = np.linalg.solve(precision, rhs)
        covariance = np.linalg.inv(precision)
        context, eta = self._stability_project(context)
        residual = targets - phi @ eta
        nis_mean = float(np.mean(residual**2 / self.noise_variance))
        radius = self.spectral_radius(eta)
        return AdaptationResult(
            context=context,
            eta=eta,
            covariance=covariance,
            spectral_radius=radius,
            stable=radius < 1.0,
            nis_mean=nis_mean,
        )

    def rollout(self, eta: np.ndarray, controls: np.ndarray, x0: np.ndarray) -> np.ndarray:
        states, _ = simulate(eta, controls, x0, self.dt)
        return states

    def information_score(
        self,
        controls: np.ndarray,
        x0: np.ndarray,
        prior_precision: float = 1.0,
        context_samples: np.ndarray | None = None,
    ) -> float:
        """Robust D-optimal score over representative pretrained contexts."""
        self._check_fitted()
        if context_samples is None:
            context_samples = np.zeros((1, self.rank))
        scores = []
        for context in np.atleast_2d(context_samples):
            eta = self.context_to_eta(context)
            nominal_states = self.rollout(eta, controls, x0)
            design = regression_features(nominal_states, controls) @ self.basis
            information = prior_precision * np.eye(self.rank) + design.T @ design / self.noise_variance
            sign, logdet = np.linalg.slogdet(information)
            scores.append(logdet if sign > 0 else -np.inf)
        return float(np.min(scores))
