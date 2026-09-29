"""Imperfect benchmark: position only, noise, mismatch, and true random inputs."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

BASE_MEAN = np.array([1.20, 0.32, 0.90, 0.10])
BASE_MAP = np.array([[.55, .12], [.07, .11], [.16, -.09], [.035, .055]])
LOWER = np.array([.35, .08, .45, .005])
UPPER = np.array([2.25, .65, 1.35, .24])


@dataclass(frozen=True)
class RealisticConfig:
    seed: int = 12
    dt: float = .05
    training_tasks: int = 18
    training_horizon: int = 500
    adaptation_horizon: int = 45
    validation_horizon: int = 180
    excitation_candidates: int = 400
    monte_carlo_runs: int = 120
    measurement_std: float = .012
    process_std: float = .018
    process_rho: float = .82
    prior_precision: float = .5


@dataclass(frozen=True)
class TrueTask:
    context: np.ndarray
    eta: np.ndarray
    quintic: float
    force_bias: float


def physical_coefficients(context: np.ndarray) -> np.ndarray:
    return BASE_MEAN + BASE_MAP @ np.asarray(context, dtype=float)


def sample_true_task(context: np.ndarray, rng: np.random.Generator) -> TrueTask:
    context = np.asarray(context, dtype=float)
    residual = rng.normal(scale=[.015, .004, .008, .004])
    return TrueTask(
        context.copy(), physical_coefficients(context) + residual,
        float(.035 + .012 * rng.uniform()),
        float(.006 * context[0] + .002 * rng.normal()),
    )


def actuator(value):
    return .75 * np.tanh(np.clip(value, -.90, .90) / .75)


def random_multisine(horizon: int, rng: np.random.Generator, amplitude: float = 1.2) -> np.ndarray:
    time = np.arange(horizon, dtype=float)
    signal = np.zeros(horizon)
    for _ in range(4):
        signal += np.sin(2 * np.pi * rng.uniform(.015, .18) * time + rng.uniform(0, 2*np.pi))
    signal /= max(np.max(np.abs(signal)), 1e-12)
    signal += .12 * rng.normal(size=horizon)
    return amplitude * np.clip(signal, -1, 1)


def nominal_rollout_eta(eta, controls, x0, dt):
    controls = np.asarray(controls)
    delayed = np.r_[0., controls[:-1]]
    states = np.zeros((len(controls) + 1, 2)); states[0] = x0
    k, damping, gain, cubic = np.asarray(eta)
    for t, raw in enumerate(delayed):
        q, velocity = states[t]
        acceleration = -k*q - damping*velocity + gain*actuator(raw) - cubic*q**3
        velocity += dt * acceleration
        q += dt * velocity
        states[t+1] = q, velocity
    return states


def simulate_true(task, controls, x0, cfg, rng):
    controls = np.asarray(controls)
    delayed = np.r_[0., controls[:-1]]
    states = np.zeros((len(controls) + 1, 2)); states[0] = x0
    colored = 0.; k, damping, gain, cubic = task.eta
    for t, raw in enumerate(delayed):
        q, velocity = states[t]
        colored = cfg.process_rho * colored + cfg.process_std * rng.normal()
        acceleration = (-k*q - damping*velocity + gain*actuator(raw) - cubic*q**3
                        - task.quintic*q**5 + task.force_bias + colored)
        velocity += cfg.dt * acceleration
        q += cfg.dt * velocity
        states[t+1] = q, velocity
    measurements = states[:, 0] + cfg.measurement_std * rng.normal(size=len(states))
    return states, measurements


def identify_eta(measurements, controls, cfg, prior_weight=.04):
    x0 = np.array([measurements[0], 0.])
    scale = UPPER - LOWER
    def residual(eta):
        prediction = nominal_rollout_eta(eta, controls, x0, cfg.dt)[:, 0]
        return np.r_[(prediction-measurements)/cfg.measurement_std,
                     np.sqrt(prior_weight)*(eta-BASE_MEAN)/scale]
    return least_squares(residual, BASE_MEAN, bounds=(LOWER, UPPER), max_nfev=100).x


class LearnedFamily:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, datasets):
        self.training_coefficients = np.vstack([identify_eta(y, u, self.cfg) for y, u in datasets])
        self.mean = self.training_coefficients.mean(axis=0)
        centered = self.training_coefficients - self.mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        variance = singular**2 / max(len(centered)-1, 1)
        self.explained_variance = variance / variance.sum()
        scale = np.sqrt(np.maximum(variance[:2], 1e-8))
        self.decoder = vt[:2].T * scale
        self.training_contexts = centered @ vt[:2].T / scale
        return self

    def decode(self, context):
        return self.mean + self.decoder @ np.asarray(context)

    def rollout(self, context, controls, x0):
        return nominal_rollout_eta(self.decode(context), controls, x0, self.cfg.dt)

    def adapt(self, measurements, controls):
        x0 = np.array([measurements[0], 0.])
        def residual(context):
            prediction = self.rollout(context, controls, x0)[:, 0]
            return np.r_[(prediction-measurements)/self.cfg.measurement_std,
                         np.sqrt(self.cfg.prior_precision)*context]
        result = least_squares(residual, np.zeros(2), bounds=(-3, 3), max_nfev=80)
        return result.x, self.decode(result.x)

    def information_score(self, controls, x0):
        samples = self.training_contexts[::max(1, len(self.training_contexts)//6)]
        epsilon = 1e-3; scores = []
        for context in samples:
            jacobian = np.empty((len(controls)+1, 2))
            for j in range(2):
                delta = np.zeros(2); delta[j] = epsilon
                plus = self.rollout(context+delta, controls, x0)[:, 0]
                minus = self.rollout(context-delta, controls, x0)[:, 0]
                jacobian[:, j] = (plus-minus)/(2*epsilon)
            information = self.cfg.prior_precision*np.eye(2) + jacobian.T@jacobian/self.cfg.measurement_std**2
            scores.append(np.linalg.slogdet(information)[1])
        return float(np.min(scores))


def rmse(reference, prediction):
    return float(np.sqrt(np.mean((reference-prediction)**2)))


def run_realistic_experiment(output_dir, config=None):
    cfg = config or RealisticConfig(); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed); x0 = np.array([.20, 0.])
    datasets = []; true_training = []
    for _ in range(cfg.training_tasks):
        task = sample_true_task(rng.uniform(-1, 1, 2), rng)
        controls = random_multisine(cfg.training_horizon, rng, 1.6)
        _, measurements = simulate_true(task, controls, x0, cfg, rng)
        datasets.append((measurements, controls)); true_training.append(task.eta)
    family = LearnedFamily(cfg).fit(datasets)

    test_task = sample_true_task(np.array([.78, -.72]), rng)
    candidates = [random_multisine(cfg.adaptation_horizon, rng) for _ in range(cfg.excitation_candidates)]
    candidate_scores = np.array([family.information_score(u, x0) for u in candidates])
    active_input = candidates[int(np.argmax(candidate_scores))]
    names = ("no_adaptation", "from_scratch", "random", "active", "oracle")
    errors = {name: [] for name in names}; coefficient_errors = {name: [] for name in ("from_scratch", "random", "active")}
    random_scores = []; example = {}
    for trial in range(cfg.monte_carlo_runs):
        _, active_y = simulate_true(test_task, active_input, x0, cfg, rng)
        _, active_eta = family.adapt(active_y, active_input)
        random_input = random_multisine(cfg.adaptation_horizon, rng)
        random_scores.append(family.information_score(random_input, x0))
        _, random_y = simulate_true(test_task, random_input, x0, cfg, rng)
        _, random_eta = family.adapt(random_y, random_input)
        scratch_eta = identify_eta(active_y, active_input, cfg, prior_weight=.002)
        validation_input = random_multisine(cfg.validation_horizon, rng, .9)
        truth, _ = simulate_true(test_task, validation_input, x0, cfg, rng)
        predictions = {
            "no_adaptation": nominal_rollout_eta(family.mean, validation_input, x0, cfg.dt),
            "from_scratch": nominal_rollout_eta(scratch_eta, validation_input, x0, cfg.dt),
            "random": nominal_rollout_eta(random_eta, validation_input, x0, cfg.dt),
            "active": nominal_rollout_eta(active_eta, validation_input, x0, cfg.dt),
            "oracle": nominal_rollout_eta(test_task.eta, validation_input, x0, cfg.dt),
        }
        for name in names: errors[name].append(rmse(truth, predictions[name]))
        for name, estimate in (("from_scratch", scratch_eta), ("random", random_eta), ("active", active_eta)):
            coefficient_errors[name].append(np.linalg.norm(estimate-test_task.eta)/np.linalg.norm(test_task.eta))
        if trial == 0:
            example = dict(truth=truth, validation_input=validation_input, active_input=active_input,
                           random_input=random_input, active_prediction=predictions["active"],
                           random_prediction=predictions["random"], no_adaptation_prediction=predictions["no_adaptation"])
    errors = {k: np.asarray(v) for k, v in errors.items()}; coefficient_errors = {k: np.asarray(v) for k, v in coefficient_errors.items()}
    difference = errors["random"]-errors["active"]
    boot = difference[rng.integers(0, len(difference), size=(3000, len(difference)))].mean(axis=1)
    metrics = {
        "config": asdict(cfg),
        "assumptions": {"measurement":"noisy position only", "disturbance":"colored acceleration noise",
            "mismatch":"omitted quintic force, force bias, off-manifold coefficients",
            "actuator":"bounded nonlinear actuator and one-sample delay",
            "random_baseline":"fresh unscored input in every trial"},
        "true_test_coefficients": test_task.eta.tolist(),
        "rank_two_explained_variance": float(family.explained_variance[:2].sum()),
        "active_information_score": float(np.max(candidate_scores)),
        "random_information_score_mean": float(np.mean(random_scores)),
        "rmse": {k:{"mean":float(v.mean()),"std":float(v.std()),"median":float(np.median(v)),
                    "q10":float(np.quantile(v,.1)),"q90":float(np.quantile(v,.9))} for k,v in errors.items()},
        "coefficient_relative_error": {k:{"mean":float(v.mean()),"std":float(v.std())} for k,v in coefficient_errors.items()},
        "active_vs_random": {"mean_rmse_reduction":float(difference.mean()),
            "relative_mean_reduction":float(difference.mean()/errors["random"].mean()),
            "paired_bootstrap_95pct_ci":[float(x) for x in np.quantile(boot,[.025,.975])],
            "active_win_rate":float(np.mean(errors["active"]<errors["random"]))},
    }
    (output_dir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    np.savez(output_dir/"experiment_arrays.npz", **example, candidate_information_scores=candidate_scores,
             random_information_scores=np.asarray(random_scores), training_coefficients_estimated=family.training_coefficients,
             training_coefficients_true=np.vstack(true_training), true_test_eta=test_task.eta,
             **{f"rmse_{k}":v for k,v in errors.items()}, **{f"coefficient_error_{k}":v for k,v in coefficient_errors.items()})
    return metrics


def create_realistic_figures(output_dir):
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir); data = np.load(output_dir/"experiment_arrays.npz")
    metrics = json.loads((output_dir/"metrics.json").read_text()); dt = metrics["config"]["dt"]
    names = ["no_adaptation","from_scratch","random","active","oracle"]
    labels = ["No adapt.","4-D scratch","Random + context","Active + context","Oracle coeff."]
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    axes[0,0].boxplot([data[f"rmse_{n}"] for n in names],tick_labels=labels,showfliers=False)
    axes[0,0].tick_params(axis="x",rotation=18); axes[0,0].set_ylabel("state prediction RMSE"); axes[0,0].set_title("Noisy, mismatched Monte Carlo trials")
    scores=data["candidate_information_scores"]; axes[0,1].hist(scores,bins=24,color=".75")
    axes[0,1].axvline(scores.max(),color="tab:blue",label="selected active")
    axes[0,1].axvline(data["random_information_scores"].mean(),color="tab:orange",label="mean true-random")
    axes[0,1].set_xlabel("robust log-det score"); axes[0,1].legend(); axes[0,1].set_title("Only the proposed input is score-selected")
    truth=data["truth"]; time=dt*np.arange(len(truth)); axes[1,0].plot(time,truth[:,0],"k",lw=2,label="latent truth")
    axes[1,0].plot(time,data["no_adaptation_prediction"][:,0],"--",color=".5",label="no adapt.")
    axes[1,0].plot(time,data["random_prediction"][:,0],color="tab:orange",label="random")
    axes[1,0].plot(time,data["active_prediction"][:,0],color="tab:blue",label="active")
    axes[1,0].set(xlabel="time [s]",ylabel="position",title="Representative held-out rollout"); axes[1,0].legend(ncol=2)
    random=data["rmse_random"]; active=data["rmse_active"]; limit=1.05*max(random.max(),active.max())
    axes[1,1].scatter(random,active,s=22,alpha=.65); axes[1,1].plot([0,limit],[0,limit],"k--")
    axes[1,1].set(xlim=(0,limit),ylim=(0,limit),xlabel="random RMSE",ylabel="active RMSE",
                  title=f"Paired trials: active wins {100*metrics['active_vs_random']['active_win_rate']:.1f}%")
    fig.suptitle("Partial observation, realistic noise, and model mismatch",fontsize=14); fig.tight_layout()
    fig.savefig(output_dir/"realistic_summary.png",dpi=180,bbox_inches="tight"); fig.savefig(output_dir/"realistic_summary.pdf",bbox_inches="tight"); plt.close(fig)
