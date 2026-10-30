# Polyherbal GPR Prototype

A small research prototype exploring Gaussian Process Regression (GPR) for mixture-formula data. The current code contains a Matérn-kernel model and synthetic examples that show how a model can rank candidate ingredient proportions.

## What is implemented

- GPR fitting for numeric mixture inputs and measured response values.
- Predictive means and uncertainty estimates through scikit-learn.
- A synthetic holdout example for inspecting how ranking behaves with a small dataset.

## What is not implemented

This repository does not contain a complete Bayesian-optimization workflow: the current version does not implement Expected Hypervolume Improvement (EHVI), active experiment selection, or experimental validation against published formula datasets. The synthetic examples are demonstrations only and do not support pharmacological conclusions.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Quick example

```python
import numpy as np
from gpr_polyherbal.validation import train_gpr

# Rows are synthetic ingredient proportions; y is a measured response in a real use case.
X = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.3, 0.2, 0.5]])
y = np.array([0.62, 0.71, 0.58])

model = train_gpr(X, y, n_restarts=2)
mean, uncertainty = model.predict(X, return_std=True)
```

For any real experiment, use a pre-specified design, keep replicate and batch information, validate predictions on held-out observations, and discuss model limits with a domain researcher. A model score or predicted optimum alone is not evidence of efficacy.

## Technology

Python, NumPy, scikit-learn GaussianProcessRegressor.

## License

MIT License, see [LICENSE](LICENSE).
