# Binary Matrix Completion via Convex Relaxation

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Optuna](https://img.shields.io/badge/optuna-hyperparameter_optimization-orange.svg)
![Algorithm](https://img.shields.io/badge/algorithm-ADMM-success.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> [cite_start]A robust, from-scratch Python implementation of the Alternating Direction Method of Multipliers (ADMM) for recovering latent low-rank matrices from partially observed, noisy binary data[cite: 1].

[cite_start]This repository contains the complete mathematical derivation, convex relaxation proofs, and an optimized Python implementation for solving the binary matrix completion problem[cite: 1, 158]. [cite_start]It demonstrates how to effectively handle combinatorially intractable $l_0$ norms and rank constraints using their tightest convex surrogates[cite: 27, 29].

## 📑 Table of Contents
- [Features](#-features)
- [Mathematical Formulation](#-mathematical-formulation)
- [Repository Structure](#-repository-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#-usage)
- [Experimental Results](#-experimental-results)
- [Author](#-author)

## ✨ Features
* [cite_start]**Custom ADMM Solver:** Implements a dual-residual stopping criterion and fixed scalar penalty for stable convergence[cite: 99, 128].
* [cite_start]**Majorization-Minimization (MM):** Resolves the loss of rotational invariance caused by Hadamard masking during the X-step update[cite: 74, 75].
* [cite_start]**Singular Value Thresholding (SVT):** Exact proximal operator implementation for the nuclear norm using Fan's inequality[cite: 81, 86].
* [cite_start]**Hyperparameter Optimization:** Integrates Optuna with Tree-structured Parzen Estimator (TPE) sampling to jointly optimize regularization and penalty parameters[cite: 133, 135].
* [cite_start]**Synthetic Data Pipeline:** Built-in generator for low-rank ground-truth matrices with customizable observation masks and label noise[cite: 285, 324].

## 🧮 Mathematical Formulation

The natural, non-convex objective minimizes the Hadamard distance on observed entries subject to a low-rank constraint:

$$\min_{X \in \mathbb{R}^{m \times n}} \| [cite_start]R \odot (Y - X) \|_0 + \lambda \text{rank}(X)$$ [cite: 18]

**Convex Relaxation:**
[cite_start]Because the $l_0$ norm and rank constraints are NP-hard, we decouple the $l_1$ term from the linear operator by introducing a sparse error matrix $E$[cite: 26, 39]. The reformulated convex program is:

$$\min_{X, E \in \mathbb{R}^{m \times n}} \| E \|_1 + \lambda \| [cite_start]X \|_* \quad \text{subject to} \quad R \odot X + E = R \odot Y$$ [cite: 43]

[cite_start]This is solved iteratively via an Augmented Lagrangian formulation, ensuring the constraint is enforced exclusively on the observed entries defined by the mask $R$[cite: 41, 47].

## 📂 Repository Structure

```text
.
├── solution.py          # Main ADMM algorithm, synthetic dataset generation, and evaluation metrics [cite: 158]
├── report_final.pdf     # Comprehensive report containing full mathematical proofs and lemmas 
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
