"""Build the GitHub-ready educational notebook using only the standard library."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "01_learning_a_family_of_dynamical_systems.ipynb"


def _source(text: str) -> list[str]:
    return (dedent(text).strip("\n") + "\n").splitlines(keepends=True)


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


cells = [
    markdown(
        r"""
        # Learning a family of dynamical systems

        ## From many known oscillators to one new oscillator with only a few experiments

        **Research-motivation notebook for dynamical-systems foundation models**

        A foundation model for dynamics should not merely memorize one plant. It should learn
        reusable structure from a *distribution of systems*, then identify a new system using
        little data and predict its future under new inputs.

        This notebook builds the smallest transparent version of that idea. There is no neural
        network and no hidden magic: every matrix can be inspected.

        By the end, you will be able to answer four questions:

        1. What is the ground truth?
        2. What information is available to the learner?
        3. What is transferred from old systems to a new one?
        4. Why can an intelligently chosen experiment be better than an arbitrary one?

        > **Honesty note.** This is a mechanism demonstration, not evidence that the real-world
        > problem is solved. The final sections explain why the benchmark is easy and how a
        > research project should make it harder.
        """
    ),
    markdown(
        r"""
        ## 0. The story in one picture

        We study several spring--mass systems. They share the same equation but have different
        physical coefficients. From the old systems, we learn the few directions along which the
        coefficients change. For a new system, we freeze this shared knowledge and estimate only
        a two-dimensional context.

        $$
        \text{old systems}
        \longrightarrow \text{shared low-dimensional family}
        \longrightarrow \text{informative pushes}
        \longrightarrow \text{new-system context}
        \longrightarrow \text{future prediction}.
        $$

        | Object | Known by simulator? | Given to learner? |
        |---|:---:|:---:|
        | Secret physical coefficients | Yes | No |
        | Applied force $u_t$ | Yes | Yes |
        | Position and velocity | Yes | Yes, with noise |
        | True context of new system | Yes | No |
        | Test trajectory | Yes | Only after prediction |
        """
    ),
    markdown("## 1. Imports and reproducibility\n\nRun from the repository root or from the `notebooks/` directory."),
    code(
        """
        from pathlib import Path
        import sys
        import numpy as np
        import matplotlib.pyplot as plt

        ROOT = Path.cwd()
        if ROOT.name == "notebooks":
            ROOT = ROOT.parent
        if not (ROOT / "src" / "dynafm_demo").exists():
            raise FileNotFoundError("Open this notebook from the dynafm_demo repository.")

        sys.path.insert(0, str(ROOT / "src"))
        from dynafm_demo.model import SharedContextModel
        from dynafm_demo.system import multisine_input, parameters_from_context, simulate

        SEED = 12
        rng = np.random.default_rng(SEED)
        dt = 0.05
        x0 = np.array([0.20, 0.0])
        plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True})
        """
    ),
    markdown(
        r"""
        ## 2. The world: a controlled nonlinear oscillator

        The simulator represents a cart attached to a nonlinear spring:

        $$
        \dot q=v, \qquad
        \dot v=-kq-dv+bu-\alpha q^3+w.
        $$

        Here $q$ is position, $v$ is velocity, $u$ is our force, and $w$ is a disturbance.
        The coefficients $k,d,b,\alpha$ are stiffness, damping, input gain, and nonlinear stiffness.
        We collect them in $\eta=[k\;d\;b\;\alpha]^\top$.

        The **ground truth** is the value of $\eta$ used internally by the simulator. The learner
        observes input--state trajectories, never $\eta$ itself.
        """
    ),
    code(
        """
        example_contexts = np.array([
            [-0.9, -0.7], [-0.4, 0.8], [0.4, -0.6], [0.9, 0.7]
        ])
        eta_examples = np.vstack([
            parameters_from_context(c).as_vector() for c in example_contexts
        ])
        print("Rows are [stiffness, damping, input gain, cubic term]:")
        print(np.round(eta_examples, 3))
        """
    ),
    code(
        """
        horizon = 220
        common_input = multisine_input(horizon, rng, amplitude=1.0)
        time = dt * np.arange(horizon + 1)

        fig, ax = plt.subplots()
        for context, eta in zip(example_contexts, eta_examples):
            states, _ = simulate(eta, common_input, x0, dt)
            ax.plot(time, states[:, 0], label=f"context={context}")
        ax.set(xlabel="time [s]", ylabel="position q", title="Same push, different systems")
        ax.legend(ncol=2)
        plt.show()
        """
    ),
    markdown(
        r"""
        **Observation.** A model of only the average system will make systematic errors on a new
        member of the family.

        ## 3. The hidden family structure

        The simulator generates four coefficients from only two hidden values:

        $$
        \eta(c)=
        \begin{bmatrix}1.20\\0.32\\0.90\\0.10\end{bmatrix}
        +
        \begin{bmatrix}
        0.55&0.12\\0.07&0.11\\0.16&-0.09\\0.035&0.055
        \end{bmatrix}c,
        \qquad c\in[-1,1]^2.
        $$

        The learner is **not given this matrix**. It discovers the family from trajectories.
        With the semi-implicit discretization,

        $$
        a_t=\frac{v_{t+1}-v_t}{\Delta t}
        =\underbrace{[-q_t,-v_t,u_t,-q_t^3]}_{\phi_t^\top}\eta.
        $$

        We first estimate one $\eta_i$ per training task, then learn the low-rank geometry of the
        estimated coefficients.
        """
    ),
    markdown("## 4. Pretraining on a distribution of systems\n\nWe generate rich trajectories from 24 systems. This is the analogue of pretraining data."),
    code(
        """
        n_tasks = 24
        training_horizon = 1200
        process_std = 0.0015
        measurement_std = 0.0001
        training_data, true_training_eta = [], []

        for _ in range(n_tasks):
            context = rng.uniform(-1.0, 1.0, size=2)
            eta = parameters_from_context(context).as_vector()
            controls = multisine_input(training_horizon, rng, amplitude=1.8)
            _, measured = simulate(
                eta, controls, x0, dt, rng,
                process_std=process_std, measurement_std=measurement_std,
            )
            training_data.append((measured, controls))
            true_training_eta.append(eta)

        true_training_eta = np.vstack(true_training_eta)
        model = SharedContextModel(rank=2, dt=dt).fit(training_data)
        print("Learned mean:", np.round(model.mean, 4))
        print("Variance explained by two directions:",
              np.round(model.explained_variance[:2].sum(), 6))
        """
    ),
    code(
        """
        names = ["stiffness k", "damping d", "input gain b", "cubic alpha"]
        fig, axes = plt.subplots(1, 4, figsize=(13, 3))
        for j, ax in enumerate(axes):
            ax.scatter(true_training_eta[:, j], model.task_coefficients[:, j], s=25)
            lo = min(true_training_eta[:, j].min(), model.task_coefficients[:, j].min())
            hi = max(true_training_eta[:, j].max(), model.task_coefficients[:, j].max())
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set(title=names[j], xlabel="ground truth", ylabel="estimated")
        fig.suptitle("Per-task identification during pretraining")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        PCA supplies the best rank-two affine approximation

        $$\widehat\eta_i\approx\bar\eta+Uc_i.$$

        In a large foundation model, a neural encoder and structured latent transition would
        replace the fixed features and PCA. The transfer principle remains the same: keep shared
        knowledge fixed and adapt only a small task representation.
        """
    ),
    markdown("## 5. The exam: one unseen system\n\nThe next line is visible only for evaluation; the learner does not receive the secret context."),
    code(
        """
        true_test_context = np.array([0.78, -0.72])  # hidden from the learner
        true_test_eta = parameters_from_context(true_test_context).as_vector()
        print("Secret context:", true_test_context)
        print("Secret coefficients:", np.round(true_test_eta, 4))
        print("Pretrained average:", np.round(model.mean, 4))
        """
    ),
    markdown(
        r"""
        ## 6. Design an informative experiment

        A weak push may make many candidate systems look alike. For a candidate input $u$,

        $$
        \mathcal I(u)=\beta I+\frac{1}{\widehat\sigma^2}
        U^\top\Phi(u)^\top\Phi(u)U,
        \qquad J(u)=\log\det\mathcal I(u).
        $$

        A larger $J$ corresponds to a smaller posterior uncertainty volume for the context.
        """
    ),
    code(
        """
        adaptation_horizon = 25
        candidates = [multisine_input(adaptation_horizon, rng, amplitude=1.2)
                      for _ in range(160)]
        learned_contexts = (model.task_coefficients - model.mean) @ model.basis
        representatives = learned_contexts[::3]
        scores = np.array([
            model.information_score(u, x0, context_samples=representatives)
            for u in candidates
        ])
        active_index = int(np.argmax(scores))
        median_index = int(np.argsort(scores)[len(scores) // 2])
        active_input = candidates[active_index]
        baseline_input = candidates[median_index]

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
        axes[0].hist(scores, bins=20, color="0.7", edgecolor="white")
        axes[0].axvline(scores[median_index], color="tab:orange", label="median")
        axes[0].axvline(scores[active_index], color="tab:blue", label="selected")
        axes[0].set(xlabel="robust log-det score", ylabel="count", title="Candidate experiments")
        axes[0].legend()
        axes[1].step(range(adaptation_horizon), baseline_input, where="post",
                     color="tab:orange", label="median")
        axes[1].step(range(adaptation_horizon), active_input, where="post",
                     color="tab:blue", label="active")
        axes[1].set(xlabel="sample", ylabel="input u", title="Two admissible experiments")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 7. Few-shot adaptation

        Freeze the mean and basis. From 25 transitions, estimate only two context values:

        $$
        \widehat c=\arg\min_c
        \frac{1}{\widehat\sigma^2}\|a-\Phi(\bar\eta+Uc)\|_2^2
        +\beta\|c\|_2^2.
        $$

        This is the transparent analogue of adapting a prompt, context vector, or low-rank adapter
        while freezing a large pretrained backbone.
        """
    ),
    code(
        """
        def adapt_once(experiment_input, local_rng):
            _, measured = simulate(
                true_test_eta, experiment_input, x0, dt, local_rng,
                process_std=process_std, measurement_std=measurement_std,
            )
            return model.adapt(measured, experiment_input, prior_precision=1.0)

        active_result = adapt_once(active_input, rng)
        baseline_result = adapt_once(baseline_input, rng)
        print("Estimated physical coefficients:", np.round(active_result.eta, 4))
        print("Ground-truth coefficients:", np.round(true_test_eta, 4))
        print("Linearized spectral radius:", round(active_result.spectral_radius, 6))
        """
    ),
    markdown(
        r"""
        The PCA coordinates need not equal the simulator's secret coordinates: PCA may rotate or
        reflect the latent axes. The meaningful comparison is between decoded physical coefficients
        and, ultimately, predicted behavior.

        ## 8. Test on data that adaptation never saw

        A fresh input is generated after adaptation. The simulator produces the ground-truth
        trajectory, and each model predicts it without further fitting.
        """
    ),
    code(
        """
        validation_horizon = 260
        validation_input = multisine_input(validation_horizon, rng, amplitude=0.9)
        truth, _ = simulate(true_test_eta, validation_input, x0, dt)
        average_prediction = model.rollout(model.mean, validation_input, x0)
        baseline_prediction = model.rollout(baseline_result.eta, validation_input, x0)
        active_prediction = model.rollout(active_result.eta, validation_input, x0)

        def rmse(reference, prediction):
            return float(np.sqrt(np.mean((reference - prediction) ** 2)))

        for label, prediction in [
            ("No adaptation", average_prediction),
            ("Median experiment", baseline_prediction),
            ("Active experiment", active_prediction),
        ]:
            print(f"{label:20s}: RMSE = {rmse(truth, prediction):.6f}")

        t_val = dt * np.arange(validation_horizon + 1)
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for j, ylabel in enumerate(["position q", "velocity v"]):
            axes[j].plot(t_val, truth[:, j], "k", linewidth=2, label="ground truth")
            axes[j].plot(t_val, average_prediction[:, j], "--", label="no adaptation")
            axes[j].plot(t_val, active_prediction[:, j], label="active adaptation")
            axes[j].set(ylabel=ylabel)
        axes[0].set_title("Unseen validation trajectory")
        axes[0].legend(ncol=3)
        axes[1].set_xlabel("time [s]")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown("## 9. One trial can be lucky: Monte Carlo evaluation\n\nRepeat adaptation with new noise while keeping horizon and amplitude identical."),
    code(
        """
        n_trials = 60
        active_errors, baseline_errors = [], []
        for _ in range(n_trials):
            active_trial = adapt_once(active_input, rng)
            baseline_trial = adapt_once(baseline_input, rng)
            active_errors.append(rmse(truth, model.rollout(active_trial.eta, validation_input, x0)))
            baseline_errors.append(rmse(truth, model.rollout(baseline_trial.eta, validation_input, x0)))

        active_errors = np.asarray(active_errors)
        baseline_errors = np.asarray(baseline_errors)
        print(f"Median experiment: {baseline_errors.mean():.6f} ± {baseline_errors.std():.6f}")
        print(f"Active experiment: {active_errors.mean():.6f} ± {active_errors.std():.6f}")

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.boxplot([baseline_errors, active_errors], tick_labels=["median", "active"])
        ax.set(ylabel="validation RMSE", title=f"{n_trials} noise realizations")
        plt.show()
        """
    ),
    markdown(
        r"""
        ## 10. Why are the errors suspiciously small?

        The learner has several strong advantages:

        1. **Correct feature map:** it already knows the terms $q$, $v$, $u$, and $q^3$.
        2. **Full state measurement:** velocity is observed rather than reconstructed.
        3. **Exact low-dimensional family:** the coefficients lie exactly on an affine plane.
        4. **In-distribution test:** the new context is inside the training domain.
        5. **Low noise:** numerical differentiation remains easy.
        6. **Matched models:** there are no delays, saturation, drift, or missing physics.

        So the correct conclusion is not “the problem is solved.” It is:

        > When a system family has reusable low-dimensional structure, learning that structure can
        > reduce new-system identification to a small adaptation problem, and input design can make
        > each measurement more informative.
        """
    ),
    markdown(
        r"""
        ## 11. Turn the demonstration into a research programme

        | Research step | Harder setting | Scientific question |
        |---|---|---|
        | Partial observation | $y_t=q_t+\nu_t$; velocity hidden | Can an encoder infer state and task jointly? |
        | Model mismatch | delay, saturation, colored noise, omitted forces | Does adaptation absorb mismatch or become biased? |
        | Approximate family | nonlinear coefficient manifold | When is a small adapter sufficient? |
        | Distribution shift | contexts outside $[-1,1]^2$ | Can uncertainty detect unsafe extrapolation? |
        | Learned backbone | SSM or Transformer dynamics | Which structure improves efficiency and stability? |
        | Safe excitation | state and input constraints | Can the model learn without dangerous experiments? |
        | Theory | finite-sample bounds and Lyapunov certificates | When should adaptation be trusted? |

        Baselines should include independent identification from scratch, pooled training without
        task context, full fine-tuning, random excitation, and an oracle that knows the coefficients.
        """
    ),
    markdown(
        r"""
        ## 12. Connection to a scalable foundation model

        | This notebook | Scalable research version |
        |---|---|
        | Known state $x_t=[q_t,v_t]$ | Encoder from partial/noisy history |
        | Fixed physics features | Learned structured latent dynamics |
        | PCA basis $U$ | Context encoder, hypernetwork, or low-rank adapter |
        | Closed-form MAP adaptation | Gradient or amortized context inference |
        | Log-det information score | Constrained Bayesian experiment design |
        | Local spectral-radius check | Lyapunov or dissipativity certificate |

        > **Research hypothesis:** A foundation model for control should learn not only to predict
        > trajectories, but also how systems vary, how to identify a new system efficiently, and
        > when its adapted model is safe enough to use.
        """
    ),
    markdown(
        r"""
        ## Exercises

        1. Use 10, 50, and 100 adaptation transitions. Plot error against the data budget.
        2. Increase measurement noise. Why does velocity differencing amplify it?
        3. Select the minimum-information candidate. Does the performance gap increase?
        4. Test the out-of-distribution context $[1.3,-1.2]^\top$.
        5. Hide velocity and construct an observer before adapting the context.
        6. Add an unmodelled term $-\gamma q^5$ and plot error against $\gamma$.
        7. Replace PCA with a nonlinear decoder and evaluate on held-out system families.

        ## Takeaway

        We are not guessing a label for the new oscillator. We want to predict how it will move
        under future forces. Secret simulator parameters define the ground truth; short experiments
        provide evidence; pretraining restricts the search to a learned family; and active
        excitation tries to make every new measurement count.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(OUTPUT)
