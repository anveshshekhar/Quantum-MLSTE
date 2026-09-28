import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I, X, Y, Z]

def get_n_qubit_projectors(n_qubits: int):
    proj_z = [0.5 * (I + Z), 0.5 * (I - Z)]
    proj_x = [0.5 * (I + X), 0.5 * (I - X)]
    proj_y = [0.5 * (I + Y), 0.5 * (I - Y)]
    bases = [proj_z, proj_x, proj_y]
    
    basis_combos = list(itertools.product(range(3), repeat=n_qubits))
    outcome_combos = list(itertools.product(range(2), repeat=n_qubits))
    
    all_basis_projectors = []
    for combo in basis_combos:
        qubit_bases = [bases[idx] for idx in combo]
        basis_projs = []
        for out in outcome_combos:
            P = qubit_bases[0][out[0]]
            for q in range(1, n_qubits):
                P = np.kron(P, qubit_bases[q][out[q]])
            basis_projs.append(P)
        all_basis_projectors.append(basis_projs)
    return all_basis_projectors

def get_readout_matrix(n_qubits: int, e01: float = 0.03, e10: float = 0.02) -> np.ndarray:
    A1 = np.array([[1 - e01, e10], [e01, 1 - e10]])
    A_n = A1
    for _ in range(n_qubits - 1):
        A_n = np.kron(A_n, A1)
    return A_n

def params_to_rho(params: np.ndarray, d: int) -> np.ndarray:
    T = np.zeros((d, d), dtype=complex)
    idx = 0
    for i in range(d):
        T[i, i] = params[idx]
        idx += 1
    for i in range(d):
        for j in range(i):
            T[i, j] = complex(params[idx], params[idx+1])
            idx += 2
    T_dag_T = T.conj().T @ T
    tr = np.trace(T_dag_T)
    if np.abs(tr) < 1e-15:
        return np.eye(d, dtype=complex) / d
    return T_dag_T / tr

def vectorized_nll_loss(params: np.ndarray, counts_matrix: np.ndarray, projs_tensor: np.ndarray, A_matrix: np.ndarray, d: int) -> float:
    rho = params_to_rho(params, d)
    probs_ideal = np.real(np.einsum('ij,bkji->bk', rho, projs_tensor))
    probs_obs = np.einsum('kj,bj->bk', A_matrix, probs_ideal)
    probs_obs = np.clip(probs_obs, 1e-12, 1.0)
    return -np.sum(counts_matrix * np.log(probs_obs))

def mle_qst_nqubit(counts_matrix: np.ndarray, projs_tensor: np.ndarray, A_matrix: np.ndarray, n_qubits: int) -> np.ndarray:
    d = 2**n_qubits
    n_params = d**2
    t_init = np.zeros(n_params)
    t_init[:d] = 1.0
    res = minimize(vectorized_nll_loss, t_init, args=(counts_matrix, projs_tensor, A_matrix, d), method='L-BFGS-B', options={'maxiter': 100})
    return params_to_rho(res.x, d)

def linear_inversion_nqubit(all_basis_counts: list, n_qubits: int, shots_per_basis: int) -> np.ndarray:
    d = 2**n_qubits
    pauli_to_basis = {0: 0, 1: 1, 2: 2, 3: 0}
    
    basis_combos = list(itertools.product(range(3), repeat=n_qubits))
    outcome_combos = list(itertools.product(range(2), repeat=n_qubits))
    
    rho_lin = np.zeros((d, d), dtype=complex)
    
    for pauli_indices in itertools.product(range(4), repeat=n_qubits):
        basis_tuple = tuple(pauli_to_basis[p] for p in pauli_indices)
        basis_idx = basis_combos.index(basis_tuple)
        counts = all_basis_counts[basis_idx] / shots_per_basis
        
        exp_val = 0.0
        for outcome, prob in zip(outcome_combos, counts):
            sign = 1.0
            for p, o in zip(pauli_indices, outcome):
                if p != 0:
                    sign *= (-1.0 if o == 1 else 1.0)
            exp_val += sign * prob
            
        P_tensor = PAULIS[pauli_indices[0]]
        for q in range(1, n_qubits):
            P_tensor = np.kron(P_tensor, PAULIS[pauli_indices[q]])
            
        rho_lin += exp_val * P_tensor
        
    return rho_lin / float(d)

def pure_state_fidelity(rho_true: np.ndarray, rho_est: np.ndarray) -> float:
    return float(np.real(np.trace(rho_true @ rho_est)))

def run_benchmarks(n_qubits: int = 2, n_points: int = 100, n_trials: int = 25):
    os.makedirs("plots", exist_ok=True)
    d = 2**n_qubits
    all_projectors = get_n_qubit_projectors(n_qubits)
    projs_tensor = np.array(all_projectors)
    A_matrix = get_readout_matrix(n_qubits, e01=0.03, e10=0.02)
    
    psi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    rho_true = np.outer(psi, psi.conj())
    shots_grid = np.unique(np.logspace(1, 4, n_points).astype(int))

    mean_min_eig_lin, mean_min_eig_mle = [], []
    mean_fid_mle = []

    for N_shots in shots_grid:
        lin_eigs, mle_eigs, fids = [], [], []
        for _ in range(n_trials):
            counts = []
            for basis_projs in all_projectors:
                probs = [np.real(np.trace(rho_true @ P)) for P in basis_projs]
                c = np.random.multinomial(N_shots, probs)
                counts.append(c)
            counts_matrix = np.array(counts)
            
            rho_lin = linear_inversion_nqubit(counts, n_qubits, N_shots)
            rho_mle = mle_qst_nqubit(counts_matrix, projs_tensor, np.eye(d), n_qubits)
            
            lin_eigs.append(np.min(np.real(np.linalg.eigvals(rho_lin))))
            mle_eigs.append(np.min(np.real(np.linalg.eigvals(rho_mle))))
            fids.append(pure_state_fidelity(rho_true, rho_mle))
            
        mean_min_eig_lin.append(np.mean(lin_eigs))
        mean_min_eig_mle.append(np.mean(mle_eigs))
        mean_fid_mle.append(np.mean(fids))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.axhline(0, color='black', linestyle='--', alpha=0.7, label="Physical Bound (λ ≥ 0)")
    ax1.plot(shots_grid, mean_min_eig_lin, color='crimson', label="Linear Inversion (MC Avg)", linewidth=1.5)
    ax1.plot(shots_grid, mean_min_eig_mle, color='teal', label="MLE (Cholesky)", linewidth=1.5)
    ax1.set_xscale('log')
    ax1.set_xlabel("Shots per Basis")
    ax1.set_ylabel("Minimum Eigenvalue")
    ax1.set_title(f"{n_qubits}-Qubit Physical Validity Check")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.plot(shots_grid, mean_fid_mle, color='purple', label="MLE Fidelity (MC Avg)", linewidth=1.5)
    ax2.set_xscale('log')
    ax2.set_xlabel("Shots per Basis")
    ax2.set_ylabel("Fidelity")
    ax2.set_title(f"{n_qubits}-Qubit Fidelity Convergence")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()
    plt.tight_layout()
    plt.savefig("plots/qst_physical_validity.png", dpi=300)
    plt.close()

    fid_raw_mle, fid_mit_mle = [], []

    for N_shots in shots_grid:
        f_raw, f_mit = [], []
        for _ in range(n_trials):
            counts = []
            for basis_projs in all_projectors:
                p_ideal = np.array([np.real(np.trace(rho_true @ P)) for P in basis_projs])
                p_obs = A_matrix @ p_ideal
                c = np.random.multinomial(N_shots, p_obs)
                counts.append(c)
            counts_matrix = np.array(counts)
            
            rho_raw = mle_qst_nqubit(counts_matrix, projs_tensor, np.eye(d), n_qubits)
            rho_mit = mle_qst_nqubit(counts_matrix, projs_tensor, A_matrix, n_qubits)
            
            f_raw.append(pure_state_fidelity(rho_true, rho_raw))
            f_mit.append(pure_state_fidelity(rho_true, rho_mit))
            
        fid_raw_mle.append(np.mean(f_raw))
        fid_mit_mle.append(np.mean(f_mit))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(shots_grid, fid_raw_mle, color='crimson', label="Raw MLE (Unmitigated)", linewidth=1.5)
    ax.plot(shots_grid, fid_mit_mle, color='teal', label="SPAM Mitigated MLE", linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel("Shots per Basis")
    ax.set_ylabel("Fidelity")
    ax.set_title(f"{n_qubits}-Qubit SPAM Readout Error Mitigation")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/qst_spam_mitigation.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    run_benchmarks(n_qubits=2, n_points=100, n_trials=25)