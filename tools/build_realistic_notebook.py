"""Generate the main educational notebook for the realistic benchmark."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "02_realistic_few_shot_system_identification.ipynb"

def source(text): return (dedent(text).strip("\n")+"\n").splitlines(keepends=True)
def md(text): return {"cell_type":"markdown","metadata":{},"source":source(text)}
def code(text): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":source(text)}

cells = [
md(r"""
# Few-shot identification of an imperfect dynamical system

## Can pretraining and active excitation help when measurements are noisy and the model is wrong?

This is the main research-motivation notebook. It replaces the exact-model toy result with a
benchmark containing partial observation, colored disturbances, actuator constraints, and
unmodelled dynamics.

![Realistic benchmark summary](../results/realistic_summary.png)

The intended claim is deliberately modest: **under the tested assumptions, selecting an input by
a robust information criterion improves two-dimensional context adaptation relative to genuinely
unscored random inputs.** It is not a claim of universal superiority.
"""),
md(r"""
## 1. Why the first toy experiment was insufficient

| Earlier mechanism demo | Realistic benchmark |
|---|---|
| Position and velocity measured | Noisy position only; velocity hidden |
| Tiny white noise | Sensor noise plus colored process disturbance |
| Exact affine task family | Task coefficients include off-manifold residuals |
| Predictor matches simulator | Simulator has omitted quintic force and bias |
| Linear input gain | Saturating nonlinear actuator with delay |
| “Random” chosen near median score | Fresh unscored random input in every trial |
| One attractive trajectory | 120 paired Monte Carlo trials and bootstrap interval |

The comparison becomes meaningful only after these advantages are removed.
"""),
md(r"""
## 2. Ground truth and learner are intentionally different

The hidden plant is

$$
\dot q=v,\qquad
\dot v=-kq-dv+b\,\varphi(u_{t-1})-\alpha q^3-\gamma q^5+r+w_t,
\qquad y_t=q_t+\nu_t,
$$

where $w_t=0.82w_{t-1}+\epsilon_t$ is colored, $\varphi$ is a bounded actuator, and only position
$y_t$ is measured. The learner uses

$$
\dot{\hat v}=-\hat k\hat q-\hat d\hat v+\hat b\,\varphi(u_{t-1})-\hat\alpha\hat q^3.
$$

Thus, even an oracle that knows $(k,d,b,\alpha)$ cannot predict future disturbances or the omitted
terms $-\gamma q^5+r$. Its nonzero error is the benchmark's irreducible model-mismatch floor.
"""),
code("""
from pathlib import Path
import sys, json
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if ROOT.name == "notebooks": ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/"src"))
from dynafm_demo.realistic import RealisticConfig, run_realistic_experiment, create_realistic_figures

cfg = RealisticConfig()
cfg
"""),
md(r"""
## 3. Pretraining learns an approximate family

For each of 18 training plants, the method performs output-error identification from a long noisy
position trajectory:

$$
\widehat\eta_i=\arg\min_\eta
\sum_t\frac{(y_{i,t}-\widehat q_t(\eta))^2}{\sigma_y^2}+\lambda R(\eta).
$$

It then learns a normalized rank-two model

$$\widehat\eta_i\approx\bar\eta+Uz_i,\qquad z_i\sim\mathcal N(0,I).$$

The retained rank-two subspace explains about 82%—not 100%—of the coefficient variation. This is
important: the representation is useful but imperfect.
"""),
md(r"""
## 4. A new plant: adapt two values, not four

For a new system, $\bar\eta$ and $U$ are frozen. From only 45 input/output transitions,

$$
\widehat z=\arg\min_z
\sum_t\frac{(y_t-\widehat q_t(\bar\eta+Uz))^2}{\sigma_y^2}
+\beta\|z\|_2^2.
$$

Because velocity is hidden, this is a nonlinear output-error problem. The full four-parameter
identification baseline receives the same short active trajectory but estimates
$(k,d,b,\alpha)$ directly.
"""),
md(r"""
## 5. Proposed excitation versus a true random baseline

For each candidate input, position sensitivity to the context is approximated by finite differences:

$$S_{tj}=\frac{\widehat q_t(z+\varepsilon e_j)-\widehat q_t(z-\varepsilon e_j)}{2\varepsilon}.$$

The robust score is

$$J(u)=\min_{z\in\mathcal Z_{\rm train}}\log\det\left(\beta I+\sigma_y^{-2}S(u,z)^\top S(u,z)\right).$$

The proposed method selects the best of 400 admissible signals. The random baseline draws a new
signal in each trial and **never evaluates its score before using it**.
"""),
md("## 6. Reproduce or load the included experiment\n\nSet `RUN_FULL_EXPERIMENT=True` to rerun the complete benchmark; it takes roughly 20 seconds on the reference CPU."),
code("""
RUN_FULL_EXPERIMENT = False
results_dir = ROOT/"results"
if RUN_FULL_EXPERIMENT or not (results_dir/"metrics.json").exists():
    metrics = run_realistic_experiment(results_dir, cfg)
    create_realistic_figures(results_dir)
else:
    metrics = json.loads((results_dir/"metrics.json").read_text())

{name: round(values["mean"], 5) for name, values in metrics["rmse"].items()}
"""),
code("""
comparison = metrics["active_vs_random"]
print(f"Active vs random mean reduction: {100*comparison['relative_mean_reduction']:.1f}%")
print(f"Active wins paired trials:       {100*comparison['active_win_rate']:.1f}%")
print("95% bootstrap CI for absolute mean reduction:",
      np.round(comparison["paired_bootstrap_95pct_ci"], 5))
"""),
code("""
arrays = np.load(results_dir/"experiment_arrays.npz")
names = ["no_adaptation", "from_scratch", "random", "active", "oracle"]
labels = ["no adapt.", "4-D scratch", "random", "active", "oracle"]
plt.figure(figsize=(9,4))
plt.boxplot([arrays[f"rmse_{name}"] for name in names], tick_labels=labels, showfliers=False)
plt.ylabel("held-out state RMSE")
plt.title("Distributions matter more than one attractive trajectory")
plt.grid(axis="y", alpha=.3)
plt.show()
"""),
md(r"""
## 7. Interpretation of the fixed-seed result

| Method | Mean state RMSE | Role |
|---|---:|---|
| No adaptation | 0.0541 | Learned population mean |
| Four parameters from scratch | 0.0319 | No shared low-dimensional prior |
| Random input + context | 0.0337 | Same adapter, no input selection |
| **Active input + context** | **0.0286** | Proposed combination |
| Oracle coefficients | 0.0184 | Mismatch/disturbance floor, not zero |

Active context adaptation reduces mean RMSE by about 15% relative to random context adaptation and
wins 69% of paired trials. The paired bootstrap interval for the absolute improvement excludes zero.
The method is better on average, not in every realization.

The four-parameter baseline remains competitive. That is useful: the benefit of shared structure is
not automatic and must be studied as a function of data budget, family mismatch, and task dimension.
"""),
md(r"""
## 8. What this demonstrates—and what it still does not

It demonstrates a fair computational mechanism for:

- learning an approximate task family from noisy trajectories;
- few-shot adaptation from position-only measurements;
- selecting excitation before observing the new plant's response;
- comparing against true random excitation, from-scratch identification, and an oracle;
- reporting distributions, win rate, and uncertainty rather than one run.

It still uses a known nominal model class and a two-state simulator. A postdoctoral research project
should replace the fixed nominal model with a learned structured encoder/state-space backbone, add
safe constrained input design, study out-of-distribution tasks, and establish identifiability,
finite-sample, stability, and calibration guarantees.
"""),
md(r"""
## 9. Research questions suggested by the benchmark

1. At what adaptation horizon does a learned task prior outperform identification from scratch?
2. How much off-manifold variation can a two-dimensional adapter tolerate?
3. Can an innovation-based mismatch detector decide when the adapter should be enlarged?
4. How should information gain be balanced against state and input safety constraints?
5. Can a learned observer infer latent state and task context jointly from heterogeneous sensors?
6. Can predictive uncertainty distinguish process noise from representation error and distribution shift?

These are the questions that turn the demonstration into a postdoctoral research initiative rather
than a polished toy example.
"""),
md(r"""
## Exercises and ablations

1. Sweep the adaptation horizon over $\{20,30,45,60,100\}$ and plot all five baselines.
2. Increase off-manifold coefficient residuals until context adaptation loses to from-scratch fitting.
3. Replace colored noise by white noise with equal marginal variance. Which estimator is biased?
4. Evaluate a test context outside the pretraining square.
5. Add a state constraint to input selection and plot information versus safety margin.
6. Compare log-det, trace, and minimum-eigenvalue experiment-design criteria.
7. Replace the nominal output-error model by an SSM encoder and retain the same evaluation protocol.
"""),
]

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.10"}},"nbformat":4,"nbformat_minor":5}
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(nb,indent=1)+"\n",encoding="utf-8"); print(OUT)
