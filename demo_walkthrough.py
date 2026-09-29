# %% [markdown]
# # Structured foundation model for dynamical systems
#
# This VS Code-compatible notebook script demonstrates three claims:
# 1. trajectories from many related systems define a low-dimensional task manifold;
# 2. a new system can be adapted by estimating only its task context;
# 3. an information-maximising input improves few-shot adaptation.

# %%
from pathlib import Path
import sys

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from dynafm_demo import DemoConfig, run_experiment
from dynafm_demo.plotting import create_figures

# %% [markdown]
# The simulated family is a controlled Duffing oscillator:
#
# $$\dot q=v,\qquad \dot v=-kq-dv+bu-\alpha q^3.$$
#
# Its coefficient vector $\eta=[k,d,b,\alpha]^\top$ lies on an unknown
# two-dimensional affine task manifold. Pretraining estimates this manifold
# from rich trajectories of 24 systems. On a new system, only the two-dimensional
# context is estimated using only 25 transitions.

# %%
config = DemoConfig(seed=12)
metrics = run_experiment(ROOT / "results", config)
create_figures(ROOT / "results")
metrics

# %% [markdown]
# Open `results/demo_summary.png`. The relevant comparison is the rollout RMSE
# after adaptation with a median random candidate versus the candidate selected
# by the D-optimal information score.
