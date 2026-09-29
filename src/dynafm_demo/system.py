from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OscillatorParameters:
    stiffness: float
    damping: float
    input_gain: float
    cubic: float

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.stiffness, self.damping, self.input_gain, self.cubic],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, eta: np.ndarray) -> "OscillatorParameters":
        return cls(*np.asarray(eta, dtype=float).tolist())


def parameters_from_context(context: np.ndarray) -> OscillatorParameters:
    """Ground-truth two-dimensional task manifold used by the simulator."""
    c1, c2 = np.asarray(context, dtype=float)
    return OscillatorParameters(
        stiffness=1.20 + 0.55 * c1 + 0.12 * c2,
        damping=0.32 + 0.07 * c1 + 0.11 * c2,
        input_gain=0.90 + 0.16 * c1 - 0.09 * c2,
        cubic=0.10 + 0.035 * c1 + 0.055 * c2,
    )


def step(
    state: np.ndarray,
    control: float,
    eta: np.ndarray,
    dt: float,
    acceleration_noise: float = 0.0,
) -> np.ndarray:
    """Semi-implicit Euler step for a damped Duffing oscillator."""
    q, v = np.asarray(state, dtype=float)
    k, d, b, alpha = np.asarray(eta, dtype=float)
    acceleration = -k * q - d * v + b * control - alpha * q**3
    acceleration += acceleration_noise
    v_next = v + dt * acceleration
    q_next = q + dt * v_next
    return np.array([q_next, v_next])


def simulate(
    eta: np.ndarray,
    controls: np.ndarray,
    x0: np.ndarray,
    dt: float,
    rng: np.random.Generator | None = None,
    process_std: float = 0.0,
    measurement_std: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    controls = np.asarray(controls, dtype=float)
    states = np.zeros((len(controls) + 1, 2), dtype=float)
    states[0] = np.asarray(x0, dtype=float)
    for t, control in enumerate(controls):
        noise = 0.0 if rng is None else process_std * rng.normal()
        states[t + 1] = step(states[t], control, eta, dt, noise)
    if rng is None or measurement_std == 0.0:
        measurements = states.copy()
    else:
        measurements = states + measurement_std * rng.normal(size=states.shape)
    return states, measurements


def regression_features(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Features satisfying a_t = phi_t @ [k, d, b, alpha]."""
    states = np.asarray(states, dtype=float)
    controls = np.asarray(controls, dtype=float)
    q = states[:-1, 0]
    v = states[:-1, 1]
    return np.column_stack((-q, -v, controls, -(q**3)))


def acceleration_targets(states: np.ndarray, dt: float) -> np.ndarray:
    states = np.asarray(states, dtype=float)
    return (states[1:, 1] - states[:-1, 1]) / dt


def multisine_input(
    horizon: int,
    rng: np.random.Generator,
    amplitude: float = 1.0,
    components: int = 5,
) -> np.ndarray:
    t = np.arange(horizon, dtype=float)
    signal = np.zeros(horizon, dtype=float)
    for _ in range(components):
        frequency = rng.uniform(0.01, 0.20)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        signal += np.sin(2.0 * np.pi * frequency * t + phase)
    signal /= max(np.max(np.abs(signal)), 1e-12)
    signal += 0.15 * rng.normal(size=horizon)
    return amplitude * np.clip(signal, -1.0, 1.0)
