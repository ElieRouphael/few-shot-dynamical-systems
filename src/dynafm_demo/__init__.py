"""Minimal structured foundation-model demo for dynamical systems."""

from .experiment import DemoConfig, run_experiment
from .realistic import RealisticConfig, run_realistic_experiment

__all__ = ["DemoConfig", "run_experiment", "RealisticConfig", "run_realistic_experiment"]
