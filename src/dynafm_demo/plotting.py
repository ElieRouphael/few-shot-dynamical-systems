from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def create_figures(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    data = np.load(results_dir / "experiment_arrays.npz")
    metrics = json.loads((results_dir / "metrics.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.hist(data["information_scores"], bins=22, color="#8fb9dd", edgecolor="white")
    ax.axvline(metrics["active_information_score"], color="#d1495b", lw=2, label="selected")
    ax.axvline(metrics["median_information_score"], color="#4f6d7a", lw=2, label="median baseline")
    ax.set_title("Candidate adaptation experiments")
    ax.set_xlabel(r"$\log\det$ information score")
    ax.set_ylabel("count")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    names = ["Random input", "Active input"]
    values = [metrics["random_coefficient_relative_error"], metrics["active_coefficient_relative_error"]]
    errors = [
        metrics["random_coefficient_relative_error_std"],
        metrics["active_coefficient_relative_error_std"],
    ]
    ax.bar(names, values, yerr=errors, capsize=4, color=["#4f6d7a", "#0077b6"])
    ax.set_title("Coefficient error after adaptation (Monte Carlo)")
    ax.set_ylabel("relative coefficient error")
    ax.tick_params(axis="x", rotation=12)

    time = np.arange(len(data["truth"])) * metrics["config"]["dt"]
    ax = axes[1, 0]
    ax.plot(time, data["truth"][:, 0], "k", lw=2, label="true")
    ax.plot(time, data["zero_prediction"][:, 0], color="#8d99ae", lw=1.2, label="no adaptation")
    ax.plot(time, data["random_prediction"][:, 0], color="#d97904", lw=1.2, label="random")
    ax.plot(time, data["active_prediction"][:, 0], color="#0077b6", lw=1.4, label="active")
    ax.set_title("Position rollout on the unseen system")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1, 1]
    labels = [r"$k$", r"$d$", r"$b$", r"$\alpha$"]
    x = np.arange(4)
    width = 0.26
    ax.bar(x - width, data["true_eta"], width, color="black", label="true")
    ax.bar(x, data["random_eta"], width, color="#d97904", label="random")
    ax.bar(x + width, data["active_eta"], width, color="#0077b6", label="active")
    ax.set_xticks(x, labels)
    ax.set_title("Adapted physical coefficients")
    ax.legend(frameon=False)

    fig.suptitle("Structured dynamical foundation-model demonstration", fontsize=15)
    fig.savefig(results_dir / "demo_summary.png", dpi=180)
    fig.savefig(results_dir / "demo_summary.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    t = np.arange(len(data["active_input"])) * metrics["config"]["dt"]
    ax.plot(t, data["active_input"], color="#0077b6", lw=1.5, label="active")
    ax.plot(t, data["random_input"], color="#d97904", lw=1.2, label="median random candidate")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("adaptation input")
    ax.set_title("Selected informative excitation versus baseline")
    ax.legend(frameon=False)
    fig.savefig(results_dir / "adaptation_inputs.png", dpi=180)
    plt.close(fig)
