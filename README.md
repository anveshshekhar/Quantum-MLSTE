# Maximum Likelihood Quantum State Tomography (MLE-QST)

A vectorised $N$-qubit Quantum State Tomography (QST) and State Preparation and Measurement (SPAM) error-mitigation engine built in Python.

## Overview

Linear Inversion Tomography back-calculates density matrices $\rho$ directly from measurement frequencies. Under finite shot noise, this unconstrained approach routinely yields unphysical density matrices with negative eigenvalues ($\lambda < 0$).

This engine enforces physical validity ($\rho \ge 0$, $\text{Tr}(\rho) = 1$) via Cholesky parameterization:
$$\rho(T) = \frac{T^\dagger T}{\text{Tr}(T^\dagger T)}$$

Parameters are optimized using L-BFGS-B minimization over a vectorized multinomial Negative Log-Likelihood (NLL) loss function.

## Key Features

- **Cholesky Parameterization:** Guarantees positive semi-definite density matrices across all shot scales.
- **SPAM Readout Mitigation:** Inverts readout confusion matrices $A_{kj}$ inside the NLL loss to recover true state fidelity under hardware readout noise.
- **Vectorized Likelihood Evaluation:** Tensor contraction via `numpy.einsum` accelerates multi-qubit optimization loops.
- **Physical Observables Suite:** Evaluates State Fidelity $\mathcal{F}$, Purity $\gamma = \text{Tr}(\rho^2)$, Von Neumann Entropy $S(\rho)$, and 2-Qubit Concurrence $C(\rho)$.

## Results

### 1. Physical Validity Check
Linear Inversion produces negative minimum eigenvalues at low shot counts ($N < 1000$). Cholesky MLE strictly preserves $\lambda_i \ge 0$.

### 2. SPAM Error Mitigation
Under 3% $0\to1$ and 2% $1\to0$ readout error, unmitigated MLE plateaus at $\mathcal{F} \approx 0.927$. SPAM-mitigated MLE recovers state fidelity to $\mathcal{F} \approx 0.998$.

## Installation & Execution

```bash
pip install -r requirements.txt
python mle_qst.py
```
