#!/usr/bin/env python3
"""
Analyze eigenvectors of J^T D J with varying D.

This script:
1. Loads J and D matrices from files
2. Computes J^T D J
3. Extracts the 5 largest eigenvectors
4. Creates heatmaps showing how eigenvectors change as D values are pulled toward their mean
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
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


def compute_jtdj(J, D):
    """
    Compute J^T D J.

    Args:
        J: Constraint Jacobian matrix (constraints x DOFs)
        D: Diagonal matrix (constraints x constraints)

    Returns:
        J^T D J as a sparse matrix
    """
    J_csr = sp.csr_matrix(J)
    D_csr = sp.csr_matrix(D)

    # Compute J^T D J
    DJ = D_csr @ J_csr
    JtDJ = J_csr.T @ DJ

    return JtDJ


def compute_all_eigenvectors(A):
    """
    Compute all eigenvectors and eigenvalues of a symmetric matrix using full eigendecomposition.

    Args:
        A: Symmetric sparse or dense matrix

    Returns:
        eigenvalues: Array of all eigenvalues
        eigenvectors: Matrix of all eigenvectors (columns)
    """
    # Ensure the matrix is square and symmetric
    assert A.shape[0] == A.shape[1], "Matrix must be square"

    try:
        # Convert to dense if sparse
        if sp.issparse(A):
            A_dense = A.toarray()
        else:
            A_dense = A

        # Use full eigendecomposition for symmetric matrices
        eigenvalues, eigenvectors = np.linalg.eigh(A_dense)

        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors
    except Exception as e:
        print(f"Error computing eigenvectors: {e}")
        raise


def create_interpolated_D_values(D_diag, num_steps):
    """
    Create a sequence of D diagonal values interpolating from original to uniform mean.

    Args:
        D_diag: Original D diagonal values
        num_steps: Number of interpolation steps (including original and uniform)

    Returns:
        List of D diagonal arrays, from original to uniform mean
    """
    mean_val = np.mean(D_diag)
    D_sequence = []

    for i in range(num_steps):
        # Linear interpolation parameter: 0 (original) to 1 (uniform)
        alpha = i / (num_steps - 1)

        # Interpolate: D_new = (1 - alpha) * D_orig + alpha * mean
        D_new = (1 - alpha) * D_diag + alpha * mean_val
        D_sequence.append(D_new)

    return D_sequence


def build_schwarz_subdomains(J):
    """
    Build Schwarz subdomains from constraint Jacobian J.
    Each row of J (constraint) defines a subdomain containing all DOFs with nonzero entries.

    Args:
        J: Constraint Jacobian matrix (constraints x DOFs)

    Returns:
        List of arrays, each containing DOF indices for one subdomain
    """
    J_csr = sp.csr_matrix(J)
    subdomains = []

    for i in range(J_csr.shape[0]):
        # Get DOF indices with nonzero entries in this constraint
        row_start = J_csr.indptr[i]
        row_end = J_csr.indptr[i + 1]
        dofs = J_csr.indices[row_start:row_end]

        if len(dofs) > 0:
            subdomains.append(dofs)

    return subdomains


def build_additive_schwarz_preconditioner(A, subdomains, weight=0.1):
    """
    Build additive Schwarz preconditioner as a dense matrix.

    M^{-1} = weight * sum_i R_i^T A_ii^{-1} R_i

    Args:
        A: System matrix (sparse)
        subdomains: List of DOF index arrays for each subdomain
        weight: Weight applied to each correction (default: 0.1)

    Returns:
        Preconditioner matrix M^{-1} as dense array
    """
    A_csr = sp.csr_matrix(A)
    n = A.shape[0]
    M_inv = np.zeros((n, n))

    for subdomain_dofs in subdomains:
        # Extract subdomain matrix
        A_sub = A_csr[np.ix_(subdomain_dofs, subdomain_dofs)].toarray()

        # Invert subdomain matrix
        try:
            A_sub_inv = np.linalg.inv(A_sub)
        except np.linalg.LinAlgError:
            # Skip singular subdomains
            assert False
            continue

        # Add weighted contribution: R_i^T A_ii^{-1} R_i
        for i, dof_i in enumerate(subdomain_dofs):
            for j, dof_j in enumerate(subdomain_dofs):
                M_inv[dof_i, dof_j] += weight * A_sub_inv[i, j]

    return M_inv


def compute_condition_number(A, M_inv):
    """
    Compute condition number of preconditioned system M^{-1} A using sparse eigenvalue estimation.

    Args:
        A: System matrix (sparse or dense)
        M_inv: Preconditioner matrix (dense)

    Returns:
        Condition number (ratio of largest to smallest eigenvalue)
    """
    if sp.issparse(A):
        A = A.toarray()

    # Compute preconditioned system
    M_inv_A = M_inv @ A
    
    eigvals = np.linalg.eigvalsh(M_inv_A)
    print(np.sort(np.abs(eigvals)))
    return np.max(np.abs(eigvals))

def main():
    parser = argparse.ArgumentParser(
        description='Analyze eigenvectors of A + J^T D J with varying D',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mats-dir', type=str, default='mats',
                        help='Directory containing matrix files')
    parser.add_argument('--iteration', type=int, required=True,
                        help='Iteration number to analyze')
    parser.add_argument('--num-eigenvectors', type=int, default=None,
                        help='Number of top eigenvectors to display (default: all)')
    parser.add_argument('--num-steps', type=int, default=5,
                        help='Number of interpolation steps from original to uniform D')
    parser.add_argument('--output-prefix', type=str, default='eigenvector_analysis',
                        help='Prefix for output files')
    parser.add_argument('--compute-condition-numbers', action='store_true',
                        help='Compute condition numbers (can be slow, off by default)')
    args = parser.parse_args()

    mats_dir = Path(args.mats_dir)

    # Construct filenames
    j_file = mats_dir / f"J_matrix_iter_{args.iteration}.mat"
    d_file = mats_dir / f"D_matrix_iter_{args.iteration}.mat"
    p_file = mats_dir / f"P_matrix_iter_{args.iteration}.mat"
    ptap_file = mats_dir / f"PTAP_matrix_iter_{args.iteration}.mat"

    print("=" * 70)
    print("Eigenvector Analysis of A + J^T D J")
    print("=" * 70)
    print(f"Matrix directory: {mats_dir}")
    print(f"Iteration: {args.iteration}")
    print(f"Number of steps: {args.num_steps}")
    print()

    # Load matrices
    print("Loading matrices...")
    J = read_mfem_matrix(j_file)
    D = read_mfem_matrix(d_file)
    P = read_mfem_matrix(p_file)
    A = read_mfem_matrix(ptap_file)  # This is PTAP = P^T (H + J^T D J) P

    print(f"J shape: {J.shape} (constraints x DOFs)")
    print(f"D shape: {D.shape}")
    print(f"P shape: {P.shape} (full DOFs x contact subspace)")
    print(f"A (PTAP) shape: {A.shape}")
    print()

    # Extract D diagonal
    D_csr = sp.csr_matrix(D)
    D_diag = np.array(D_csr.diagonal())

    print(f"D diagonal statistics:")
    print(f"  Min: {D_diag.min():.6e}")
    print(f"  Max: {D_diag.max():.6e}")
    print(f"  Mean: {D_diag.mean():.6e}")
    print(f"  Std: {D_diag.std():.6e}")
    print()

    # Note: A is PTAP = P^T (H + J^T D J) P in the contact subspace
    # To modify D, we need to project the change: P^T (JtDJ_new - JtDJ_orig) P

    # Create sequence of D values
    print(f"Creating {args.num_steps} interpolation steps...")
    D_sequence = create_interpolated_D_values(D_diag, args.num_steps)
    print()

    # Compute original J^T D J for reference
    JtDJ_orig = compute_jtdj(J, D)
    # Project to contact subspace: P^T JtDJ_orig P
    P_csr = sp.csr_matrix(P)
    PtJtDJ_orig_P = P_csr.T @ JtDJ_orig @ P_csr

    print(f"P^T J^T D J P shape: {PtJtDJ_orig_P.shape} (should match A: {A.shape})")
    print()

    # Compute eigenvectors for each D on A + P^T (JtDJ - JtDJ_orig) P
    print("Computing all eigenvectors for each D value on A + P^T (J^T D J) P...")
    eigenvector_sequence = []
    eigenvalue_sequence = []

    for i, D_diag_modified in enumerate(D_sequence):
        alpha = i / (args.num_steps - 1)
        print(f"  Step {i+1}/{args.num_steps} (alpha={alpha:.2f})...", end=" ")

        # Create diagonal matrix with modified values
        D_modified = sp.diags(D_diag_modified, format='csr')

        # Compute J^T D J with modified D
        JtDJ = compute_jtdj(J, D_modified)

        # Project to contact subspace: P^T JtDJ P
        PtJtDJ_P = P_csr.T @ JtDJ @ P_csr

        # Compute full system: A + (P^T JtDJ P - P^T JtDJ_orig P)
        A_full = A + (PtJtDJ_P - PtJtDJ_orig_P)

        # Compute all eigenvectors
        eigenvalues, eigenvectors = compute_all_eigenvectors(A_full)

        eigenvector_sequence.append(eigenvectors)
        eigenvalue_sequence.append(eigenvalues)

        print(f"Matrix size: {A_full.shape[0]} -> {len(eigenvalues)} eigenvalues")

    print()

    # Determine how many eigenvectors to display
    total_eigenvectors = len(eigenvalue_sequence[0])
    if args.num_eigenvectors is None:
        num_to_display = total_eigenvectors
    else:
        num_to_display = min(args.num_eigenvectors, total_eigenvectors)

    print(f"Total eigenvectors: {total_eigenvectors}")
    print(f"Displaying: {num_to_display}")
    print()

    # Identify DOFs in the subspace (nonzero columns of J)
    print("Identifying DOFs in constraint subspace...")
    J_csr = sp.csr_matrix(J)

    # Find columns with nonzero entries
    col_nnz = np.array(np.abs(J_csr).sum(axis=0)).flatten()
    subspace_dofs = np.where(col_nnz > 0)[0]

    print(f"  Total DOFs: {J.shape[1]}")
    print(f"  DOFs in subspace: {len(subspace_dofs)}")
    print(f"  Ratio: {len(subspace_dofs)/J.shape[1]:.2%}")
    print()

    if args.compute_condition_numbers:
        # Build Schwarz subdomains from J (in full DOF space)
        print("Building Schwarz subdomains from J...")
        subdomains_full = build_schwarz_subdomains(J)
        print(f"  Number of subdomains (full space): {len(subdomains_full)}")
        if len(subdomains_full) > 0:
            subdomain_sizes = [len(s) for s in subdomains_full]
            print(f"  Subdomain sizes: min={min(subdomain_sizes)}, "
                  f"max={max(subdomain_sizes)}, mean={np.mean(subdomain_sizes):.1f}")

        # Map subdomains to contact subspace using P
        # P is (full_DOFs x contact_subspace), so P[i, j] != 0 means full DOF i maps to contact DOF j
        print("  Mapping subdomains to contact subspace...")

        # Build mapping: full DOF -> contact subspace DOFs
        full_to_contact = {}
        for full_dof in range(P_csr.shape[0]):
            row_start = P_csr.indptr[full_dof]
            row_end = P_csr.indptr[full_dof + 1]
            contact_dofs = P_csr.indices[row_start:row_end]

            if len(contact_dofs) > 0:
                full_to_contact[full_dof] = list(contact_dofs)

        print(f"  Full DOFs with contact mapping: {len(full_to_contact)}")
        print(f"  Contact subspace size (P rows): {P.shape[0]}")
        print(f"  Full DOF space size (P cols): {P.shape[1]}")

        # Debug: check what contact DOF indices we're getting
        if len(full_to_contact) > 0:
            all_contact_indices = []
            for dofs in full_to_contact.values():
                all_contact_indices.extend(dofs)
            if len(all_contact_indices) > 0:
                print(f"  Contact DOF range from mapping: [{min(all_contact_indices)}, {max(all_contact_indices)}]")

        # Map each subdomain to contact subspace
        subdomains = []
        for subdomain_full in subdomains_full:
            contact_dofs = set()
            for full_dof in subdomain_full:
                if full_dof in full_to_contact:
                    contact_dofs.update(full_to_contact[full_dof])

            if len(contact_dofs) > 0:
                contact_dofs_array = np.array(sorted(contact_dofs))
                # Verify indices are valid
                if np.max(contact_dofs_array) >= A.shape[0]:
                    print(f"  WARNING: Invalid contact DOF index max={np.max(contact_dofs_array)} >= A.shape[0]={A.shape[0]}")
                    continue
                subdomains.append(contact_dofs_array)

        print(f"  Number of subdomains (contact space): {len(subdomains)}")
        if len(subdomains) > 0:
            subdomain_sizes_contact = [len(s) for s in subdomains]
            print(f"  Contact subspace subdomain sizes: min={min(subdomain_sizes_contact)}, "
                  f"max={max(subdomain_sizes_contact)}, mean={np.mean(subdomain_sizes_contact):.1f}")
            print(f"  Max subdomain index: {max(np.max(s) for s in subdomains)}")
            print(f"  A_full will have shape: {A.shape}")
        print()

        # Compute condition numbers for each alpha
        print("Computing preconditioned condition numbers...")
        condition_numbers = []
        unpreconditioned_condition_numbers = []

        for i, (D_diag_modified, eigenvalues, eigenvectors) in enumerate(zip(D_sequence, eigenvalue_sequence, eigenvector_sequence)):
            alpha = i / (args.num_steps - 1)
            print(f"  Step {i+1}/{args.num_steps} (alpha={alpha:.2f})...", end=" ")

            # Create diagonal matrix with modified values
            D_modified = sp.diags(D_diag_modified, format='csr')

            # Compute J^T D J with modified D
            JtDJ = compute_jtdj(J, D_modified)

            # Project to contact subspace: P^T JtDJ P
            PtJtDJ_P = P_csr.T @ JtDJ @ P_csr

            # Compute full system: A + (P^T JtDJ P - P^T JtDJ_orig P)
            A_full = A + (PtJtDJ_P - PtJtDJ_orig_P)

            # Build Schwarz preconditioner with weight 1/10
            M_inv = build_additive_schwarz_preconditioner(A_full, subdomains, weight=0.1)

            # Compute condition numbers
            cond_precond = compute_condition_number(A_full, M_inv)

            # Compute unpreconditioned condition number using eigenvalues
            A_full_dense = A_full.toarray()
            eigvals = np.linalg.eigvalsh(A_full_dense)
            eigvals_positive = eigvals[eigvals > 1e-12]
            if len(eigvals_positive) > 0:
                cond_unprecond = np.max(eigvals_positive) / np.min(eigvals_positive)
            else:
                cond_unprecond = np.inf

            condition_numbers.append(cond_precond)
            unpreconditioned_condition_numbers.append(cond_unprecond)

            print(f"κ(M⁻¹A) = {cond_precond:.3e}, κ(A) = {cond_unprecond:.3e}")

        print()

    # Create heatmaps
    print("Creating heatmaps...")

    # Determine how many eigenvectors to plot per figure (max 5 for readability)
    max_per_figure = 5
    num_figures = (num_to_display + max_per_figure - 1) // max_per_figure

    # First, compute global min/max across all eigenvectors for consistent colorbar
    global_max = 0.0
    eigenvector_data = []
    for k in range(num_to_display):
        # Collect k-th eigenvector across all steps
        eigenvector_evolution = np.column_stack([evecs[:, k] for evecs in eigenvector_sequence])
        eigenvector_data.append(eigenvector_evolution)
        global_max = max(global_max, np.abs(eigenvector_evolution).max())

    # Make colorbar symmetric around 0
    vmin = -global_max
    vmax = global_max

    print(f"Colorbar range: [{vmin:.3e}, {vmax:.3e}]")
    print(f"Creating {num_figures} figure(s) to display {num_to_display} eigenvectors")

    # Create figures with up to max_per_figure eigenvectors each
    for fig_idx in range(num_figures):
        start_k = fig_idx * max_per_figure
        end_k = min(start_k + max_per_figure, num_to_display)
        num_in_fig = end_k - start_k

        # Create a figure with subplots for each eigenvector
        fig, axes = plt.subplots(1, num_in_fig,
                                 figsize=(8*num_in_fig, 12),
                                 constrained_layout=True)

        if num_in_fig == 1:
            axes = [axes]

        for i, k in enumerate(range(start_k, end_k)):
            eigenvector_evolution = eigenvector_data[k]

            # Create heatmap with consistent colorbar range
            im = axes[i].imshow(eigenvector_evolution, aspect='auto', cmap='seismic',
                                interpolation='nearest', vmin=vmin, vmax=vmax)

            # Get eigenvalue for this eigenvector (from the original, alpha=0 case)
            eigenval = eigenvalue_sequence[0][k]
            axes[i].set_title(f'Eigenvector {k+1}\nλ = {eigenval:.3e}',
                             fontsize=18, fontweight='bold', pad=10)

            # Only add axis labels on the edges
            if i == 0:
                axes[i].set_ylabel('DOF Index', fontsize=15)
            if i == num_in_fig // 2:
                axes[i].set_xlabel('Interpolation Step (α)', fontsize=15)

            # Set x-ticks to show alpha values
            x_ticks = np.linspace(0, args.num_steps-1, min(args.num_steps, 6), dtype=int)
            x_labels = [f'{j/(args.num_steps-1):.1f}' for j in x_ticks]
            axes[i].set_xticks(x_ticks)
            axes[i].set_xticklabels(x_labels, fontsize=13)

            # Adjust y-tick labels
            axes[i].tick_params(axis='y', labelsize=11)

            # Add colorbar with larger font and adjusted size to prevent overlap
            cbar = plt.colorbar(im, ax=axes[i], label='Eigenvector Component',
                               fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('Eigenvector Component', fontsize=14)

        # Display the figure (but don't save unless it's the first one)
        if fig_idx == 0:
            # Save only the first figure
            output_file = f'{args.output_prefix}_heatmaps.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Heatmaps (eigenvectors 1-{end_k}) saved to: {output_file}")
        else:
            print(f"Displaying eigenvectors {start_k+1}-{end_k} (not saved)")

        plt.show(block=False)

    # Create a second figure showing eigenvalue evolution
    fig2, ax2 = plt.subplots(figsize=(14, 8))

    eigenvalue_array = np.column_stack(eigenvalue_sequence)
    alphas = np.linspace(0, 1, args.num_steps)

    for k in range(num_to_display):
        ax2.plot(alphas, eigenvalue_array[k, :], marker='o', linewidth=3,
                label=f'Eigenvalue {k+1}', markersize=10)

    ax2.set_xlabel('Interpolation Parameter α (0=original, 1=uniform D)', fontsize=16)
    ax2.set_ylabel('Eigenvalue', fontsize=16)
    ax2.set_title('Eigenvalue Evolution as D Becomes Uniform', fontsize=18, fontweight='bold', pad=15)
    ax2.legend(fontsize=14, loc='best', ncol=max(1, num_to_display // 10))
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=14)
    plt.tight_layout()

    # Save eigenvalue plot
    eigenvalue_output_file = f'{args.output_prefix}_eigenvalues.png'
    plt.savefig(eigenvalue_output_file, dpi=300, bbox_inches='tight')
    print(f"Eigenvalue plot saved to: {eigenvalue_output_file}")

    if args.compute_condition_numbers:
        # Create a third figure showing condition numbers
        fig3, ax3 = plt.subplots(figsize=(14, 8))

        alphas = np.linspace(0, 1, args.num_steps)

        ax3.semilogy(alphas, condition_numbers, marker='o', linewidth=3,
                     label='Preconditioned κ(M⁻¹A)', color='blue', markersize=10)
        ax3.semilogy(alphas, unpreconditioned_condition_numbers, marker='s', linewidth=3,
                     label='Unpreconditioned κ(A)', color='red', markersize=10)

        ax3.set_xlabel('Interpolation Parameter α (0=original, 1=uniform D)', fontsize=16)
        ax3.set_ylabel('Condition Number', fontsize=16)
        ax3.set_title('Condition Number Evolution (Additive Schwarz, weight=0.1)', fontsize=18, fontweight='bold', pad=15)
        ax3.legend(fontsize=14, loc='best')
        ax3.grid(True, alpha=0.3, which='both')
        ax3.tick_params(axis='both', labelsize=14)
        plt.tight_layout()

        # Save condition number plot
        cond_output_file = f'{args.output_prefix}_condition_numbers.png'
        plt.savefig(cond_output_file, dpi=300, bbox_inches='tight')
        print(f"Condition number plot saved to: {cond_output_file}")

    print()
    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    # Keep all figures displayed
    plt.show()


if __name__ == "__main__":
    main()
