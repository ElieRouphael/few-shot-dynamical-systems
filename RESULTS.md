# Results and limitations

The main benchmark uses noisy position-only measurements, colored process
disturbances, bounded nonlinear actuation, a one-sample delay, off-manifold
coefficient variation, and an unmodelled quintic force and bias.

Over 120 paired trials:

| Method | Mean ± standard deviation RMSE |
|---|---:|
| No adaptation | 0.05414 ± 0.01355 |
| Four-dimensional identification from scratch | 0.03190 ± 0.01552 |
| Random excitation + context adaptation | 0.03372 ± 0.01313 |
| Active excitation + context adaptation | **0.02862 ± 0.01052** |
| Oracle physical coefficients | 0.01843 ± 0.00549 |

Active excitation reduces mean RMSE by 15.1% relative to true random
excitation and wins 69.2% of paired trials. See `metrics.json` for the exact
bootstrap interval and quantiles.

These results do not establish general superiority. They show that the input
design criterion contains useful information under one imperfect benchmark.
The from-scratch baseline remains competitive, and the oracle error remains
nonzero because the predictor cannot know omitted forces or future noise.
