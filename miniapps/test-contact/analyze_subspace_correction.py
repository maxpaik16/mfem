#!/usr/bin/env python3
"""
Analyze subspace correction constants K0 and K1 for contact systems.

This script computes the constants from Xu's theory of subspace corrections:

K0: Stable decomposition constant
    For every v in V, there exists v_i in V_i such that
    v = sum_i v_i
    and
    sum_i (R_i^{-1} v_i, v_i) <= K0 (A v, v)

K1: Strengthened Cauchy-Schwarz constant
    For any S subset of index pairs and any u_i, v_i,
    sum_{(i,j) in S} (T_i u_i, T_j v_j)_A
    <= K1
       [sum_i (T_i u_i, u_i)_A]^(1/2)
       [sum_j (T_j v_j, v_j)_A]^(1/2)

    where T_i = R_i Q_i A = R_i A_i P_i

PSC result: kappa(B A) <= K0 K1
    where B = sum_i I_i R_i I_i^t
"""

import numpy as np
from pathlib import Path
import scipy.sparse as sp
import argparse


def read_mfem_matrix(filename):
    """
    Read a matrix from MFEM .mat format (supports both single and multi-rank files).

    MFEM parallel format:
    - First line: row_start row_end col_start nnz
    - Subsequent lines: row col value (space-separated)

    For multi-rank files, reads all files matching the pattern filename.XXXXX
    where XXXXX is the rank number (00000, 00001, etc.).

    Args:
        filename: Path to .mat file (can be with or without .00000 suffix)

    Returns:
        scipy sparse matrix
    """
    from pathlib import Path
    import glob

    filepath = Path(filename)

    # Check if this is a single-rank file or needs multi-rank handling
    if filepath.exists():
        # Single file case
        rank_files = [filepath]
    else:
        # Multi-rank case: find all rank files
        # Remove any existing rank suffix if present
        base_path = str(filepath)
        if base_path.endswith('.00000'):
            base_path = base_path[:-6]

        # Find all rank files matching pattern
        rank_files = sorted(glob.glob(f"{base_path}.*"))

        if not rank_files:
            raise FileNotFoundError(f"No matrix files found for pattern: {base_path}.*")

    # Collect data from all ranks
    all_rows = []
    all_cols = []
    all_data = []
    max_row = -1
    max_col = -1

    for rank_file in rank_files:
        with open(rank_file, 'r') as f:
            lines = f.readlines()

        # Skip empty lines
        data_lines = [line.strip() for line in lines if line.strip()]

        if not data_lines:
            continue

        # Parse header: row_start row_end col_start nnz
        header = data_lines[0].split()
        row_start = int(header[0])
        row_end = int(header[1])
        col_start = int(header[2])
        nnz = int(header[3])

        # Skip empty ranks (indicated by row_end = -1 or nnz = -1)
        if row_end == -1 or nnz == -1:
            continue

        # Parse entries: row col value
        for line in data_lines[1:]:
            parts = line.split()
            if len(parts) >= 3:
                # MFEM outputs global row/col indices
                row = int(parts[0])
                col = int(parts[1])
                val = float(parts[2])

                all_rows.append(row)
                all_cols.append(col)
                all_data.append(val)

                max_row = max(max_row, row)
                max_col = max(max_col, col)

    # Determine matrix dimensions
    if len(all_rows) > 0:
        nrows = max_row + 1
        ncols = max_col + 1
    else:
        # Empty matrix
        nrows = 0
        ncols = 0

    # Convert to scipy sparse matrix (CSR format)
    return sp.csr_matrix((all_data, (all_rows, all_cols)), shape=(nrows, ncols))


def build_block_subdomains(n, block_size):
    """
    Build block subdomains with given block size.

    Args:
        n: Total number of DOFs
        block_size: Size of each block

    Returns:
        list of arrays: Each array contains DOF indices for one block
    """
    subdomains = []
    for i in range(0, n, block_size):
        block_end = min(i + block_size, n)
        subdomains.append(np.array(range(i, block_end)))
    return subdomains


def build_subdomains_from_J(J, P, min_diag_value=0.0, D_diag=None, debug=False):
    """
    Build Schwarz subdomains from constraint Jacobian J.

    Each row of J corresponds to a constraint. The subdomain for that constraint
    includes all contact subspace DOFs (rows of P) that correspond to the
    full-space DOFs (columns) with non-zero entries in that J row.

    Args:
        J: Constraint Jacobian matrix (constraints x full DOFs)
        P: Transfer operator (contact subspace x full DOFs)
        min_diag_value: Minimal D diagonal value to include constraint
        D_diag: Diagonal of D matrix (for filtering)
        debug: Print debug information

    Returns:
        list of arrays: Each array contains contact subspace DOF indices for one subdomain
    """
    subdomains = []
    J_csr = sp.csr_matrix(J)
    P_csr = sp.csr_matrix(P)

    if debug:
        print(f"\nDEBUG: Building subdomains")
        print(f"  J shape: {J_csr.shape} (constraints x full DOFs)")
        print(f"  P shape: {P_csr.shape}")
        print(f"  P nnz: {P_csr.nnz}")

    # Build mapping: full DOF -> contact subspace DOFs
    # P is (full_dofs x contact_subspace), so P[i,j] != 0 means
    # full DOF i maps to contact subspace DOF j
    full_to_contact = {}
    for full_dof in range(P_csr.shape[0]):
        row_start = P_csr.indptr[full_dof]
        row_end = P_csr.indptr[full_dof + 1]
        contact_dofs = P_csr.indices[row_start:row_end]

        if len(contact_dofs) > 0:
            full_to_contact[full_dof] = list(contact_dofs)

    if debug:
        print(f"  Built mapping for {len(full_to_contact)} full DOFs -> contact DOFs")

    for i in range(J_csr.shape[0]):
        # Skip if D diagonal value is below threshold
        if D_diag is not None and min_diag_value > 0.0:
            if i < len(D_diag) and D_diag[i] < min_diag_value:
                continue

        # Get full-space DOF indices with non-zero entries in this constraint
        row_start = J_csr.indptr[i]
        row_end = J_csr.indptr[i + 1]
        full_dofs = J_csr.indices[row_start:row_end]

        # Map to contact subspace DOFs
        contact_dofs = set()
        for full_dof in full_dofs:
            if full_dof in full_to_contact:
                contact_dofs.update(full_to_contact[full_dof])

        if len(contact_dofs) > 0:
            subdomains.append(np.array(sorted(contact_dofs)))

    return subdomains


def build_partition_of_unity_weights(n, subdomains):
    """
    Build partition-of-unity weights for a valid decomposition.

    For each global DOF, compute weights w_i such that sum_i w_i = 1
    where the sum is over all subdomains containing that DOF.

    Args:
        n: Total number of DOFs
        subdomains: List of DOF index arrays for each subdomain

    Returns:
        weights: List of arrays, weights[i] contains the weight for each DOF in subdomain i
    """
    # Count how many subdomains contain each DOF
    dof_count = np.zeros(n)
    for dofs in subdomains:
        dof_count[dofs] += 1.0

    # Build weights: w_i(dof) = 1 / (number of subdomains containing dof)
    weights = []
    for dofs in subdomains:
        w_i = np.zeros(len(dofs))
        for local_idx, global_dof in enumerate(dofs):
            if dof_count[global_dof] > 0:
                w_i[local_idx] = 1.0 / dof_count[global_dof]
        weights.append(w_i)

    return weights


def compute_K0(A, subdomains, num_samples=100, seed=42):
    """
    Compute K0: stable decomposition constant (sampled estimate).

    For every v in V, there exists v_i in V_i such that
    v = sum_i I_i v_i
    and
    sum_i (R_i^{-1} v_i, v_i) <= K0 (A v, v)

    We assume exact local solves: R_i = A_i^{-1}, where A_i = I_i^T A I_i.

    For a valid decomposition, we use partition-of-unity weights:
    - v_i = W_i * (restricted v), where W_i is diagonal with weights
    - sum_i I_i v_i = v (exact reconstruction)

    Then we estimate K0 by sampling random vectors and computing:
    K0_estimate = max_v [sum_i (A_i v_i, v_i)] / (A v, v)

    Args:
        A: System matrix (n x n)
        subdomains: List of DOF index arrays for each subdomain
        num_samples: Number of random vectors to sample
        seed: Random seed

    Returns:
        K0_estimate: Sampled estimate of K0 constant
        subdomain_data: List of (dofs, A_sub, A_sub_inv) for use in kappa(BA) computation
    """
    np.random.seed(seed)
    n = A.shape[0]
    A_csr = sp.csr_matrix(A)

    print(f"\nComputing K0 (stable decomposition constant)...")
    print(f"  Using partition-of-unity decomposition with exact local solves")
    print(f"  Number of subdomains: {len(subdomains)}")
    print(f"  Number of samples: {num_samples}")

    # Build partition-of-unity weights
    print("  Building partition-of-unity weights...")
    pou_weights = build_partition_of_unity_weights(n, subdomains)

    # Precompute subdomain matrices A_i = I_i^T A I_i and verify SPD
    print("  Extracting subdomain matrices and verifying SPD...")
    subdomain_data = []
    subdomain_data_for_kappa = []
    num_spd = 0
    num_non_spd = 0

    for dofs, weights in zip(subdomains, pou_weights):
        # Extract subdomain matrix A_i = I_i^T A I_i
        A_sub = A_csr[np.ix_(dofs, dofs)].toarray()

        # Verify A_i is SPD via Cholesky
        try:
            L_sub = np.linalg.cholesky(A_sub)
            A_sub_inv = np.linalg.solve(L_sub.T, np.linalg.solve(L_sub, np.eye(len(dofs))))
            subdomain_data.append((dofs, weights, A_sub))
            subdomain_data_for_kappa.append((dofs, A_sub, A_sub_inv))
            num_spd += 1
        except np.linalg.LinAlgError:
            print(f"  ✗ WARNING: Subdomain (size {len(dofs)}) is NOT SPD!")
            print(f"            Xu theory assumes SPD local operators!")
            # Still add it for computation, but warn
            subdomain_data.append((dofs, weights, A_sub))
            try:
                A_sub_inv = np.linalg.inv(A_sub)
                subdomain_data_for_kappa.append((dofs, A_sub, A_sub_inv))
            except:
                pass
            num_non_spd += 1

    print(f"  SPD check: {num_spd} SPD, {num_non_spd} non-SPD")
    if num_non_spd > 0:
        print(f"  ✗ WARNING: {num_non_spd} subdomains are NOT SPD - Xu bound may not apply!")

    # Sample random vectors and compute ratio
    max_ratio = 0.0

    for sample in range(num_samples):
        # Generate random vector
        v = np.random.randn(n)
        v = v / np.linalg.norm(v)

        # Compute (A v, v)
        Av = A_csr @ v
        vAv = np.dot(v, Av)

        if vAv <= 1e-14:
            continue

        # Build valid decomposition v = sum_i I_i v_i using partition-of-unity
        # v_i = W_i * (restricted v)
        # Compute sum_i (R_i^{-1} v_i, v_i) = sum_i (A_i v_i, v_i)
        numerator = 0.0

        for dofs, weights, A_sub in subdomain_data:
            # Apply partition-of-unity weights: v_i = W_i * v[dofs]
            v_i = weights * v[dofs]

            # Compute A_i v_i (since R_i^{-1} = A_i)
            A_i_v_i = A_sub @ v_i

            # Add (A_i v_i, v_i)
            numerator += np.dot(A_i_v_i, v_i)

        ratio = numerator / vAv
        max_ratio = max(max_ratio, ratio)

        if (sample + 1) % 20 == 0:
            print(f"    Sample {sample+1}/{num_samples}: current max ratio = {max_ratio:.6f}")

    print(f"  K0 sampled estimate: {max_ratio:.6f}")
    return max_ratio, subdomain_data_for_kappa


def compute_K1(A, subdomains, num_samples=100, seed=42):
    """
    Compute K1: strengthened Cauchy-Schwarz constant (sampled estimate).

    For any subset S of index pairs and any u, v in V,
    sum_{(i,j) in S} (P_i u, P_j v)_A
    <= K1
       [sum_i (P_i u, u)_A]^(1/2)
       [sum_j (P_j v, v)_A]^(1/2)

    where P_i = I_i A_i^{-1} I_i^T A is the A-orthogonal projection onto subspace i.

    We assume exact local solves: R_i = A_i^{-1}, where A_i = I_i^T A I_i.

    For each sampled global vector u, we compute:
    - r_i = I_i^T A u (restrict residual-like quantity)
    - z_i = A_i^{-1} r_i (solve local problem)
    - P_i u = I_i z_i (prolong back to global space)

    Then we estimate K1 by sampling random vectors u, v and computing:
    K1_estimate = max [sum_{i,j} (P_i u, P_j v)_A] /
                      [sum_i (P_i u, u)_A]^(1/2) [sum_j (P_j v, v)_A]^(1/2)

    Args:
        A: System matrix (n x n)
        subdomains: List of DOF index arrays for each subdomain
        num_samples: Number of random samples
        seed: Random seed

    Returns:
        K1_estimate: Sampled estimate of K1 constant
    """
    np.random.seed(seed)
    n = A.shape[0]
    A_csr = sp.csr_matrix(A)
    m = len(subdomains)

    print(f"\nComputing K1 (strengthened Cauchy-Schwarz constant)...")
    print(f"  Using true A-orthogonal projections P_i = I_i A_i^{-1} I_i^T A")
    print(f"  Number of subdomains: {m}")
    print(f"  Number of samples: {num_samples}")

    # Precompute subdomain matrices and factorizations
    print("  Factorizing subdomain matrices...")
    subdomain_solvers = []
    num_cholesky = 0
    num_lu = 0
    num_failed = 0

    for dofs in subdomains:
        # Extract subdomain matrix A_i = I_i^T A I_i
        A_sub = A_csr[np.ix_(dofs, dofs)].toarray()

        # Factor the subdomain matrix for solving A_i z_i = r_i
        try:
            L_sub = np.linalg.cholesky(A_sub)
            subdomain_solvers.append(('cholesky', dofs, L_sub))
            num_cholesky += 1
        except np.linalg.LinAlgError:
            try:
                from scipy.linalg import lu_factor
                lu_sub = lu_factor(A_sub)
                subdomain_solvers.append(('lu', dofs, lu_sub))
                num_lu += 1
            except Exception:
                num_failed += 1
                continue

    print(f"  Factorized {num_cholesky} (Cholesky) + {num_lu} (LU) = {len(subdomain_solvers)} subdomains")
    if num_failed > 0:
        print(f"  Warning: {num_failed} subdomain factorizations failed")

    max_ratio = 0.0

    for sample in range(num_samples):
        # Generate random global vectors u and v
        u = np.random.randn(n)
        v = np.random.randn(n)
        u = u / np.linalg.norm(u)
        v = v / np.linalg.norm(v)

        # Compute A u and A v
        Au = A_csr @ u
        Av = A_csr @ v

        # Compute P_i u for each subdomain i
        # P_i u = I_i A_i^{-1} I_i^T A u = I_i A_i^{-1} r_i, where r_i = I_i^T A u
        P_u = []
        for solver_type, dofs, factor in subdomain_solvers:
            # Restrict: r_i = I_i^T A u
            r_i = Au[dofs]

            # Solve: z_i = A_i^{-1} r_i
            if solver_type == 'cholesky':
                z_i = np.linalg.solve(factor.T, np.linalg.solve(factor, r_i))
            else:  # lu
                from scipy.linalg import lu_solve
                z_i = lu_solve(factor, r_i)

            # Prolong: P_i u = I_i z_i (inject back to global space)
            Pi_u = np.zeros(n)
            Pi_u[dofs] = z_i
            P_u.append(Pi_u)

        # Compute P_j v for each subdomain j
        P_v = []
        for solver_type, dofs, factor in subdomain_solvers:
            # Restrict: r_j = I_j^T A v
            r_j = Av[dofs]

            # Solve: z_j = A_j^{-1} r_j
            if solver_type == 'cholesky':
                z_j = np.linalg.solve(factor.T, np.linalg.solve(factor, r_j))
            else:  # lu
                from scipy.linalg import lu_solve
                z_j = lu_solve(factor, r_j)

            # Prolong: P_j v = I_j z_j
            Pj_v = np.zeros(n)
            Pj_v[dofs] = z_j
            P_v.append(Pj_v)

        # Compute numerator: sum_{i,j} (P_i u, P_j v)_A = sum_{i,j} (A P_i u, P_j v)
        numerator = 0.0
        for i in range(len(P_u)):
            A_Pi_u = A_csr @ P_u[i]
            for j in range(len(P_v)):
                numerator += np.dot(A_Pi_u, P_v[j])

        # Compute denominator: [sum_i (P_i u, u)_A]^(1/2) [sum_j (P_j v, v)_A]^(1/2)
        sum_u = 0.0
        for i in range(len(P_u)):
            A_Pi_u = A_csr @ P_u[i]
            sum_u += np.dot(A_Pi_u, u)

        sum_v = 0.0
        for j in range(len(P_v)):
            A_Pj_v = A_csr @ P_v[j]
            sum_v += np.dot(A_Pj_v, v)

        if sum_u <= 1e-14 or sum_v <= 1e-14:
            continue

        denominator = np.sqrt(sum_u) * np.sqrt(sum_v)
        ratio = numerator / denominator
        max_ratio = max(max_ratio, ratio)

        if (sample + 1) % 20 == 0:
            print(f"    Sample {sample+1}/{num_samples}: current max ratio = {max_ratio:.6f}")

    print(f"  K1 sampled estimate: {max_ratio:.6f}")
    return max_ratio


def compute_preconditioned_condition_number(A, subdomains, subdomain_data=None, verbose=False):
    """
    Compute the condition number of the preconditioned system B*A.

    B = sum_i I_i A_i^{-1} I_i^T (additive Schwarz preconditioner)

    Note: BA is generally NOT symmetric in the Euclidean inner product.
    We compute eigenvalues of BA directly (which are real and positive for SPD A, B).

    Args:
        A: System matrix (n x n, assumed SPD)
        subdomains: List of DOF index arrays for each subdomain
        subdomain_data: Optional precomputed (dofs, A_sub, A_sub_inv) tuples
        verbose: Print detailed information

    Returns:
        kappa: Condition number of B*A
        lambda_min: Minimum eigenvalue
        lambda_max: Maximum eigenvalue
    """
    n = A.shape[0]
    A_csr = sp.csr_matrix(A)

    if verbose:
        print("  Verifying A is SPD...")

    # Check if A is SPD by attempting Cholesky on a sample
    if verbose:
        A_sample = A_csr[:min(100, n), :min(100, n)].toarray()
        try:
            np.linalg.cholesky(A_sample)
            print("  ✓ A appears to be SPD (sample Cholesky succeeded)")
        except np.linalg.LinAlgError:
            print("  ✗ WARNING: A may not be SPD (sample Cholesky failed)")

    if verbose:
        print("  Building preconditioner B = sum_i I_i A_i^{-1} I_i^T...")

    # Build preconditioner matrix B using the SAME A_i as in K0/K1
    B = sp.lil_matrix((n, n))
    num_cholesky = 0
    num_lu = 0
    num_failed = 0

    # If subdomain_data is provided, use it; otherwise compute it
    if subdomain_data is None:
        subdomain_data = []
        for dofs in subdomains:
            # Extract A_i = I_i^T A I_i
            A_sub = A_csr[np.ix_(dofs, dofs)].toarray()

            # Verify A_i is SPD and compute A_i^{-1}
            try:
                L_sub = np.linalg.cholesky(A_sub)
                # Compute inverse via Cholesky factor
                A_sub_inv = np.linalg.solve(L_sub.T, np.linalg.solve(L_sub, np.eye(len(dofs))))
                subdomain_data.append((dofs, A_sub, A_sub_inv))
                num_cholesky += 1
            except np.linalg.LinAlgError:
                # Fallback to LU
                try:
                    A_sub_inv = np.linalg.inv(A_sub)
                    subdomain_data.append((dofs, A_sub, A_sub_inv))
                    num_lu += 1
                    if verbose:
                        print(f"  ✗ WARNING: Subdomain {len(subdomain_data)} is NOT SPD (Cholesky failed, used LU)")
                        print(f"            Xu theory assumes SPD local operators!")
                except np.linalg.LinAlgError:
                    num_failed += 1
                    if verbose:
                        print(f"  ✗ ERROR: Failed to invert subdomain matrix (size {len(dofs)})")
                    continue

    # Assemble B
    for dofs, A_sub, A_sub_inv in subdomain_data:
        # Add contribution I_i A_i^{-1} I_i^T to B
        for local_i, global_i in enumerate(dofs):
            for local_j, global_j in enumerate(dofs):
                B[global_i, global_j] += A_sub_inv[local_i, local_j]

    if verbose:
        print(f"  Assembled B: {num_cholesky} Cholesky + {num_lu} LU + {num_failed} failed")
        if num_lu > 0:
            print(f"  ✗ WARNING: {num_lu} subdomains are NOT SPD - Xu bound may not apply!")

    B = B.tocsr()

    if verbose:
        print("  Computing B*A...")

    # Form preconditioned matrix B*A
    BA = B @ A_csr

    if verbose:
        print("  Computing eigenvalues of B*A...")
        print("  NOTE: BA is NOT symmetric in Euclidean inner product")

    # Convert to dense for eigenvalue computation
    BA_dense = BA.toarray()

    # BA is not symmetric, so use regular eigenvalue solver
    # For SPD A and B, eigenvalues should be real and positive
    eigvals_complex = np.linalg.eigvals(BA_dense)

    # Extract real parts (imaginary parts should be ~0 for SPD A, B)
    eigvals_real = np.real(eigvals_complex)
    eigvals_imag = np.imag(eigvals_complex)

    max_imag = np.max(np.abs(eigvals_imag))
    if verbose and max_imag > 1e-8:
        print(f"  ✗ WARNING: Eigenvalues have large imaginary parts (max |Im| = {max_imag:.6e})")
        print(f"            This suggests A or B is not SPD!")

    # Filter out near-zero or negative eigenvalues
    eigvals = eigvals_real # [eigvals_real > 1e-12]

    if len(eigvals) == 0:
        if verbose:
            print("  ✗ ERROR: No positive eigenvalues found")
        return np.inf, 0.0, 0.0

    lambda_min = eigvals.min()
    lambda_max = eigvals.max()
    kappa = lambda_max / lambda_min

    if verbose:
        print(f"  Eigenvalue range: [{lambda_min:.6e}, {lambda_max:.6e}]")
        print(f"  Condition number kappa(B*A) = {kappa:.3f}")

    return kappa, lambda_min, lambda_max


def print_comparison_table(results):
    """
    Print a nicely formatted comparison table of K0 and K1 results.

    Args:
        results: List of tuples (name, num_subdomains, mean_size, K0, K1, K0*K1, kappa, lambda_min, lambda_max)
    """
    print("\n" + "=" * 120)
    print("COMPARISON TABLE (Sampled Estimates)")
    print("=" * 120)

    # Header
    header = f"{'Decomposition':<30} {'#Subs':>8} {'Avg Size':>10} {'K0':>10} {'K1':>10} {'K0*K1':>10} {'κ(BA)':>10} {'λ_min':>12} {'λ_max':>12}"
    print(header)
    print("-" * 120)

    # Rows
    for name, num_subs, avg_size, K0, K1, K0K1, kappa, lmin, lmax in results:
        if kappa == np.inf:
            kappa_str = "inf"
            lmin_str = "N/A"
            lmax_str = "N/A"
        else:
            kappa_str = f"{kappa:.3f}"
            lmin_str = f"{lmin:.6e}"
            lmax_str = f"{lmax:.6e}"
        row = f"{name:<30} {num_subs:>8} {avg_size:>10.1f} {K0:>10.3f} {K1:>10.3f} {K0K1:>10.3f} {kappa_str:>10} {lmin_str:>12} {lmax_str:>12}"
        print(row)

    print("=" * 120)
    print("\nInterpretation:")
    print("  - K0, K1: SAMPLED ESTIMATES/SURROGATES, not exact Xu constants")
    print("  - κ(BA) = λ_max / λ_min: ACTUAL condition number of B*A (using eigenvalues, not SVD)")
    print("  - Xu's theory: κ(BA) <= K0*K1")
    print("  - K0*K1 may UNDERESTIMATE due to sampling, but large discrepancy suggests:")
    print("    * Non-SPD operators (check warnings above)")
    print("    * Inconsistent A_i definitions")
    print("    * Need more samples")
    print("  - All computations use the SAME A_i = I_i^T A I_i throughout")
    print("=" * 120)


def compute_unpreconditioned_condition_number(A, verbose=False):
    """
    Compute the condition number of unpreconditioned matrix A.

    Args:
        A: System matrix (n x n, assumed SPD)
        verbose: Print detailed information

    Returns:
        kappa: Condition number of A
        lambda_min: Minimum eigenvalue
        lambda_max: Maximum eigenvalue
    """
    A_csr = sp.csr_matrix(A)

    if verbose:
        print("  Computing eigenvalues of A...")

    A_dense = A_csr.toarray()

    # For SPD matrices, use eigvalsh (symmetric eigenvalue solver)
    eigvals = np.linalg.eigvalsh(A_dense)

    # Filter out near-zero or negative eigenvalues
    eigvals = eigvals[eigvals > 1e-12]

    if len(eigvals) == 0:
        if verbose:
            print("  ✗ ERROR: No positive eigenvalues found")
        return np.inf, 0.0, 0.0

    lambda_min = eigvals[0]
    lambda_max = eigvals[-1]
    kappa = lambda_max / lambda_min

    if verbose:
        print(f"  Eigenvalue range: [{lambda_min:.6e}, {lambda_max:.6e}]")
        print(f"  Condition number kappa(A) = {kappa:.3f}")

    return kappa, lambda_min, lambda_max


def main():
    parser = argparse.ArgumentParser(
        description='Analyze subspace correction constants K0 and K1',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mats-dir', type=str, default='mats',
                        help='Directory containing matrix files')
    parser.add_argument('--iteration', type=int, default=1,
                        help='Iteration number to analyze')
    parser.add_argument('--min-diag', type=float, default=0.0,
                        help='Minimum D diagonal value to include')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of random samples for estimating K0 and K1')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--block-sizes', type=int, nargs='+', default=[3, 27],
                        help='Block sizes to test (in addition to constraint-based)')
    parser.add_argument('--subsystem-analysis', action='store_true',
                        help='Compute condition numbers on PTAP subsystem (unpreconditioned and with additive Schwarz)')
    args = parser.parse_args()

    mats_dir = Path(args.mats_dir)

    # Construct filenames
    j_file = mats_dir / f"J_matrix_iter_{args.iteration}.mat"
    d_file = mats_dir / f"D_matrix_iter_{args.iteration}.mat"
    p_file = mats_dir / f"P_matrix_iter_{args.iteration}.mat"
    ptap_file = mats_dir / f"PTAP_matrix_iter_{args.iteration}.mat"
    areduced_file = mats_dir / f"Areduced_matrix_iter_{args.iteration}.mat"

    print("=" * 70)
    print("Subspace Correction Analysis")
    print("=" * 70)
    print(f"Matrix directory: {mats_dir}")
    print(f"Iteration: {args.iteration}")
    print(f"Min D diagonal: {args.min_diag}")
    print()

    # Load matrices
    print("Loading matrices...")

    def check_matrix_exists(base_file):
        """Check if matrix file exists (single or multi-rank)."""
        import glob
        if base_file.exists():
            return True
        pattern = str(base_file) + ".*"
        return len(glob.glob(pattern)) > 0

    if not check_matrix_exists(j_file):
        print(f"Error: J matrix file not found: {j_file}")
        return
    if not check_matrix_exists(d_file):
        print(f"Error: D matrix file not found: {d_file}")
        return
    if not check_matrix_exists(p_file):
        print(f"Error: P matrix file not found: {p_file}")
        return
    if not check_matrix_exists(ptap_file):
        print(f"Error: PTAP matrix file not found: {ptap_file}")
        return

    J = read_mfem_matrix(j_file)
    D = read_mfem_matrix(d_file)
    P = read_mfem_matrix(p_file)
    PTAP = read_mfem_matrix(ptap_file)

    # Load Areduced if doing subsystem analysis
    Areduced = None
    if args.subsystem_analysis:
        if check_matrix_exists(areduced_file):
            print(f"Loading Areduced for subsystem analysis...")
            Areduced = read_mfem_matrix(areduced_file)
            print(f"Areduced shape: {Areduced.shape}")
        else:
            print(f"Warning: Areduced matrix file not found: {areduced_file}")
            print(f"         Subsystem analysis will be skipped")
            args.subsystem_analysis = False

    print(f"J shape: {J.shape} (constraints x DOFs)")
    print(f"D shape: {D.shape}")
    print(f"P shape: {P.shape} (contact subspace x DOFs)")
    print(f"PTAP shape: {PTAP.shape} (projected system)")
    print()

    # Extract D diagonal
    D_csr = sp.csr_matrix(D)
    D_diag = np.array(D_csr.diagonal())

    # Use PTAP as the system matrix
    print("Using PTAP as system matrix (P^T * (Huu + J^T D J) * P)...")
    A = sp.csr_matrix(PTAP)

    print(f"A shape: {A.shape}")
    print(f"A nnz: {A.nnz}")
    print()

    # Build subdomains
    print("Building Schwarz subdomains...")
    subdomains = build_subdomains_from_J(J, P, args.min_diag, D_diag, debug=False)

    num_subdomains = len(subdomains)
    subdomain_sizes = [len(s) for s in subdomains]

    print(f"Number of subdomains: {num_subdomains}")
    if num_subdomains > 0:
        print(f"Subdomain sizes: min={min(subdomain_sizes)}, "
              f"max={max(subdomain_sizes)}, mean={np.mean(subdomain_sizes):.1f}")
    print()

    if num_subdomains == 0:
        print("Error: No subdomains created. Check min_diag threshold.")
        return

    # Store results for comparison
    results = []

    # 1. Constraint-based subdomains
    print("\n" + "=" * 70)
    print("TEST 1: Constraint-based subdomains (from J matrix)")
    print("=" * 70)
    K0_constraint, subdomain_data_constraint = compute_K0(A, subdomains, num_samples=args.num_samples, seed=args.seed)
    K1_constraint = compute_K1(A, subdomains, num_samples=args.num_samples, seed=args.seed)

    print("\nComputing actual condition number of B*A...")
    print("  Using the SAME A_i matrices as in K0 and K1 computation")
    kappa_constraint, lmin_constraint, lmax_constraint = compute_preconditioned_condition_number(
        A, subdomains, subdomain_data=subdomain_data_constraint, verbose=True
    )

    results.append((
        "Constraint-based",
        num_subdomains,
        np.mean(subdomain_sizes),
        K0_constraint,
        K1_constraint,
        K0_constraint * K1_constraint,
        kappa_constraint,
        lmin_constraint,
        lmax_constraint
    ))

    # 2. Block subdomains with different sizes
    for block_size in args.block_sizes:
        print("\n" + "=" * 70)
        print(f"TEST: Block subdomains (block size = {block_size})")
        print("=" * 70)

        block_subdomains = build_block_subdomains(A.shape[0], block_size)
        block_num_subdomains = len(block_subdomains)
        block_sizes = [len(s) for s in block_subdomains]

        print(f"Number of subdomains: {block_num_subdomains}")
        if block_num_subdomains > 0:
            print(f"Subdomain sizes: min={min(block_sizes)}, "
                  f"max={max(block_sizes)}, mean={np.mean(block_sizes):.1f}")

        K0_block, subdomain_data_block = compute_K0(A, block_subdomains, num_samples=args.num_samples, seed=args.seed)
        K1_block = compute_K1(A, block_subdomains, num_samples=args.num_samples, seed=args.seed)

        print("\nComputing actual condition number of B*A...")
        print("  Using the SAME A_i matrices as in K0 and K1 computation")
        kappa_block, lmin_block, lmax_block = compute_preconditioned_condition_number(
            A, block_subdomains, subdomain_data=subdomain_data_block, verbose=True
        )

        results.append((
            f"Block (size={block_size})",
            block_num_subdomains,
            np.mean(block_sizes),
            K0_block,
            K1_block,
            K0_block * K1_block,
            kappa_block,
            lmin_block,
            lmax_block
        ))

    # Print comparison table
    print_comparison_table(results)

    # Print theory reminder
    print("\nAccording to Xu's theory:")
    print("  kappa(B A) <= K0 * K1")
    print("  where B = sum_i I_i R_i I_i^T (additive Schwarz preconditioner)")
    print()

    # Optional: Subsystem analysis on Areduced
    if args.subsystem_analysis and Areduced is not None:
        print("\n" + "=" * 70)
        print("SUBSYSTEM ANALYSIS: Areduced Matrix")
        print("=" * 70)
        print("Analyzing the full reduced system Areduced = Huu + J^T D J")
        print()

        # Build subdomains on Areduced space (use P to map from full space)
        print("Building subdomains on Areduced space...")
        areduced_subdomains = build_subdomains_from_J(J, P, args.min_diag, D_diag, debug=False)

        # Build block subdomains on Areduced
        areduced_block_subdomains = {}
        for block_size in args.block_sizes:
            areduced_block_subdomains[block_size] = build_block_subdomains(Areduced.shape[0], block_size)

        # Add full direct solve (single subdomain containing all DOFs)
        full_subdomain = [np.array(range(Areduced.shape[0]))]

        subsystem_results = []

        # 1. Unpreconditioned Areduced
        print("\n" + "=" * 70)
        print("Unpreconditioned Areduced:")
        kappa_unprecond, lmin_unprecond, lmax_unprecond = compute_unpreconditioned_condition_number(Areduced, verbose=True)

        # 2. Full direct solve (single subdomain = exact inverse)
        print("\n" + "=" * 70)
        print("Full direct solve (single subdomain = full size):")
        _, subdomain_data_full = compute_K0(Areduced, full_subdomain, num_samples=1, seed=args.seed)
        kappa_full, lmin_full, lmax_full = compute_preconditioned_condition_number(
            Areduced, full_subdomain, subdomain_data=subdomain_data_full, verbose=True
        )
        subsystem_results.append((
            "Full direct solve",
            kappa_full, lmin_full, lmax_full
        ))

        # 3. Constraint-based decomposition
        print("\n" + "=" * 70)
        print("Constraint-based additive Schwarz:")
        _, subdomain_data_constraint_ar = compute_K0(Areduced, areduced_subdomains, num_samples=1, seed=args.seed)
        kappa_as_constraint, lmin_as_constraint, lmax_as_constraint = compute_preconditioned_condition_number(
            Areduced, areduced_subdomains, subdomain_data=subdomain_data_constraint_ar, verbose=True
        )
        subsystem_results.append((
            "Constraint-based AS",
            kappa_as_constraint, lmin_as_constraint, lmax_as_constraint
        ))

        # 4. Block subdomains
        for block_size in args.block_sizes:
            print("\n" + "=" * 70)
            print(f"Block (size={block_size}) additive Schwarz:")

            block_subs = areduced_block_subdomains[block_size]
            _, subdomain_data_block_ar = compute_K0(Areduced, block_subs, num_samples=1, seed=args.seed)
            kappa_as_block, lmin_as_block, lmax_as_block = compute_preconditioned_condition_number(
                Areduced, block_subs, subdomain_data=subdomain_data_block_ar, verbose=True
            )
            subsystem_results.append((
                f"Block (size={block_size}) AS",
                kappa_as_block, lmin_as_block, lmax_as_block
            ))

        # Print subsystem comparison table
        print("\n" + "=" * 90)
        print("SUBSYSTEM COMPARISON: Areduced Matrix")
        print("=" * 90)
        print(f"Unpreconditioned κ(Areduced) = {kappa_unprecond:.3f}")
        print("-" * 90)
        header = f"{'Preconditioner':<30} {'κ(B*Areduced)':>18} {'λ_min':>18} {'λ_max':>18}"
        print(header)
        print("-" * 90)

        for name, k, lmin, lmax in subsystem_results:
            if k == np.inf:
                k_str, lmin_str, lmax_str = "inf", "N/A", "N/A"
            else:
                k_str = f"{k:.3f}"
                lmin_str = f"{lmin:.6e}"
                lmax_str = f"{lmax:.6e}"

            row = f"{name:<30} {k_str:>18} {lmin_str:>18} {lmax_str:>18}"
            print(row)

        print("=" * 90)
        print("\nInterpretation:")
        print("  - κ(Areduced): Unpreconditioned condition number of full reduced system")
        print("  - Full direct solve: B = Areduced^{-1} (exact inverse, κ(B*Areduced) = 1)")
        print("  - Additive Schwarz (AS): B = sum_i I_i A_i^{-1} I_i^T")
        print("  - Lower κ(B*Areduced) means better preconditioner effectiveness")
        print("=" * 90)
        print()


if __name__ == "__main__":
    main()
