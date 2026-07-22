"""
GPR-Polyherbal: Bayesian Optimization Pipeline for Multi-Herb Formula Design

Gaussian Process Regression (GPR) + Expected Hypervolume Improvement (EHVI)
for navigating the combinatorial explosion of polyherbal formulation.

Core components:
  - Uniform Design (UD) for initial sampling
  - GPR with Matérn 5/2 kernel for response surface modeling
  - EHVI acquisition for multi-objective Bayesian optimization
  - Retrospective validation against 4 historical cases

Author: Xukuran Abuduxuku
License: MIT
Repository: github.com/xkrn321/gpr-polyherbal
"""

__version__ = "1.0.0"
