#!/usr/bin/env python3
"""
Test if the Additive Schwarz operator is SPD.

This script:
1. Loads J matrix from files
2. Constructs the Additive Schwarz operator M where subdomains are defined by rows of J
3. Checks if M is symmetric and positive definite
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


def identify_subspace_dofs(J):
    """
    Identify DOFs that are touched by J (have nonzero entries in any constraint).

    Args:
        J: Constraint Jacobian matrix (constraints x DOFs)

    Returns:
        subspace_dofs: Array of DOF indices in the subspace
        dof_to_subspace: Dictionary mapping full DOF index to subspace index
    """
    J_csr = sp.csr_matrix(J)

    # Find columns with nonzero entries
    col_nnz = np.array(np.abs(J_csr).sum(axis=0)).flatten()
    subspace_dofs = np.where(col_nnz > 0)[0]

    # Create mapping from full DOF space to subspace
    dof_to_subspace = {dof: i for i, dof in enumerate(subspace_dofs)}

    return subspace_dofs, dof_to_subspace


def restrict_to_subspace(A, subspace_dofs):
    """
    Restrict matrix A to the subspace defined by subspace_dofs.

    Args:
        A: System matrix (sparse)
        subspace_dofs: Array of DOF indices to keep

    Returns:
        A_sub: Restricted matrix
    """
    A_csr = sp.csr_matrix(A)
    A_sub = A_csr[np.ix_(subspace_dofs, subspace_dofs)]
    return A_sub


def map_subdomains_to_subspace(subdomains, dof_to_subspace):
    """
    Map subdomain DOF indices from full space to subspace.

    Args:
        subdomains: List of arrays with full-space DOF indices
        dof_to_subspace: Dictionary mapping full DOF to subspace index

    Returns:
        subdomains_subspace: List of arrays with subspace DOF indices
    """
    subdomains_subspace = []

    for subdomain in subdomains:
        # Map each DOF to subspace index
        subdomain_sub = []
        for dof in subdomain:
            if dof in dof_to_subspace:
                subdomain_sub.append(dof_to_subspace[dof])

        if len(subdomain_sub) > 0:
            subdomains_subspace.append(np.array(subdomain_sub))

    return subdomains_subspace


def build_additive_schwarz_operator(A, subdomains, check_eigenvalues=True):
    """
    Build the Additive Schwarz operator M applied once:

    M = sum_i R_i^T A_ii^{-1} R_i

    where:
    - R_i is the restriction operator to subdomain i
    - A_ii is the subdomain matrix (A restricted to subdomain i)

    Args:
        A: System matrix (sparse or dense, already restricted to subspace)
        subdomains: List of DOF index arrays for each subdomain (in subspace indices)
        check_eigenvalues: If True, check eigenvalues of each subdomain matrix

    Returns:
        M: Additive Schwarz operator as dense array
        subdomain_info: Dictionary with information about subdomains
    """
    if sp.issparse(A):
        A_csr = sp.csr_matrix(A)
    else:
        A_csr = sp.csr_matrix(A)

    n = A.shape[0]
    M = np.zeros((n, n))

    print(f"Building Additive Schwarz operator for {len(subdomains)} subdomains...")

    singular_count = 0
    negative_eigenvalue_count = 0
    subdomain_eigenvalue_info = []

    for idx, subdomain_dofs in enumerate(subdomains):
        if idx % 100 == 0:
            print(f"  Processing subdomain {idx+1}/{len(subdomains)}")

        # Extract subdomain matrix A_ii
        A_sub = A_csr[np.ix_(subdomain_dofs, subdomain_dofs)].toarray()

        # Check eigenvalues if requested
        if check_eigenvalues:
            eigvals = np.linalg.eigvalsh(A_sub)
            min_eigval = eigvals[0]
            max_eigval = eigvals[-1]
            num_negative = np.sum(eigvals < -1e-10)
            num_zero = np.sum(np.abs(eigvals) <= 1e-10)

            if num_negative > 0:
                negative_eigenvalue_count += 1
                subdomain_eigenvalue_info.append({
                    'index': idx,
                    'size': len(subdomain_dofs),
                    'min_eigval': min_eigval,
                    'max_eigval': max_eigval,
                    'num_negative': num_negative,
                    'num_zero': num_zero
                })

        # Invert subdomain matrix
        try:
            A_sub_inv = np.linalg.inv(A_sub)
        except np.linalg.LinAlgError:
            singular_count += 1
            if singular_count <= 5:
                print(f"  WARNING: Subdomain {idx} is singular, skipping")
            continue

        # Add contribution: R_i^T A_ii^{-1} R_i
        # This is equivalent to placing A_sub_inv in the appropriate block of M
        for i, dof_i in enumerate(subdomain_dofs):
            for j, dof_j in enumerate(subdomain_dofs):
                M[dof_i, dof_j] += A_sub_inv[i, j]

    if singular_count > 5:
        print(f"  ... and {singular_count - 5} more singular subdomains")

    subdomain_info = {
        'singular_count': singular_count,
        'negative_eigenvalue_count': negative_eigenvalue_count,
        'negative_eigenvalue_subdomains': subdomain_eigenvalue_info
    }

    return M, subdomain_info


def check_symmetry(M, tol=1e-10):
    """
    Check if matrix M is symmetric.

    Args:
        M: Matrix to check
        tol: Tolerance for symmetry check

    Returns:
        is_symmetric: Boolean
        max_asymmetry: Maximum asymmetry |M - M^T|
    """
    asymmetry = M - M.T
    max_asymmetry = np.max(np.abs(asymmetry))
    is_symmetric = max_asymmetry < tol

    return is_symmetric, max_asymmetry


def check_positive_definite(M, tol=1e-10):
    """
    Check if matrix M is positive definite by computing its eigenvalues.

    Args:
        M: Matrix to check (should be symmetric)
        tol: Tolerance for considering an eigenvalue as zero

    Returns:
        is_pd: Boolean indicating if M is positive definite
        eigenvalues: Array of eigenvalues (sorted)
        num_positive: Number of positive eigenvalues
        num_zero: Number of near-zero eigenvalues
        num_negative: Number of negative eigenvalues
    """
    eigenvalues = np.linalg.eigvalsh(M)
    eigenvalues_sorted = np.sort(eigenvalues)

    num_positive = np.sum(eigenvalues > tol)
    num_zero = np.sum(np.abs(eigenvalues) <= tol)
    num_negative = np.sum(eigenvalues < -tol)

    # Positive definite means ALL eigenvalues are positive
    is_pd = num_negative == 0 and num_zero == 0

    return is_pd, eigenvalues_sorted, num_positive, num_zero, num_negative


def main():
    parser = argparse.ArgumentParser(
        description='Test if Additive Schwarz operator is SPD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--mats-dir', type=str, default='mats',
                        help='Directory containing matrix files')
    parser.add_argument('--iteration', type=int, required=True,
                        help='Iteration number to analyze')
    parser.add_argument('--use-ptap', action='store_true',
                        help='Use PTAP matrix as the system matrix instead of J^T D J')
    parser.add_argument('--output-prefix', type=str, default='schwarz_spd',
                        help='Prefix for output files')
    args = parser.parse_args()

    mats_dir = Path(args.mats_dir)

    # Construct filenames
    j_file = mats_dir / f"J_matrix_iter_{args.iteration}.mat"
    d_file = mats_dir / f"D_matrix_iter_{args.iteration}.mat"
    ptap_file = mats_dir / f"PTAP_matrix_iter_{args.iteration}.mat"

    print("=" * 70)
    print("Additive Schwarz SPD Test")
    print("=" * 70)
    print(f"Matrix directory: {mats_dir}")
    print(f"Iteration: {args.iteration}")
    print()

    # Load matrices
    print("Loading matrices...")
    J = read_mfem_matrix(j_file)
    print(f"J shape: {J.shape} (constraints x DOFs)")

    # Identify subspace touched by J (nonzero columns)
    print("\nIdentifying subspace touched by J (nonzero columns)...")
    subspace_dofs, dof_to_subspace = identify_subspace_dofs(J)
    print(f"  Total DOFs: {J.shape[1]}")
    print(f"  DOFs in subspace (nonzero columns of J): {len(subspace_dofs)}")
    print(f"  Reduction: {J.shape[1]} -> {len(subspace_dofs)} ({100 * len(subspace_dofs) / J.shape[1]:.1f}%)")

    if args.use_ptap:
        print("\nUsing PTAP matrix (already in reduced space)")
        A = read_mfem_matrix(ptap_file)
        print(f"PTAP shape: {A.shape}")

        # Verify that PTAP size matches nonzero columns of J
        if A.shape[0] == len(subspace_dofs):
            print(f"✓ PTAP size ({A.shape[0]}) matches nonzero columns of J ({len(subspace_dofs)})")
        else:
            print(f"✗ WARNING: PTAP size ({A.shape[0]}) does NOT match nonzero columns of J ({len(subspace_dofs)})")
            print(f"  Using PTAP size as the reduced space dimension")
    else:
        print("\nUsing J^T D J as system matrix")
        D = read_mfem_matrix(d_file)
        print(f"D shape: {D.shape}")

        # Compute J^T D J in full space
        print("Computing J^T D J...")
        J_csr = sp.csr_matrix(J)
        D_csr = sp.csr_matrix(D)
        DJ = D_csr @ J_csr
        A_full = J_csr.T @ DJ
        print(f"J^T D J shape: {A_full.shape}")

        # Restrict to subspace
        print("Restricting J^T D J to subspace (nonzero columns of J)...")
        A = restrict_to_subspace(A_full, subspace_dofs)
        print(f"Restricted matrix shape: {A.shape}")
    print()

    # Build Schwarz subdomains from J (full space indices)
    print("Building Schwarz subdomains from J (full space indices)...")
    subdomains_full = build_schwarz_subdomains(J)
    print(f"Number of subdomains: {len(subdomains_full)}")

    if len(subdomains_full) > 0:
        subdomain_sizes_full = [len(s) for s in subdomains_full]
        print(f"Subdomain sizes (full space):")
        print(f"  Min: {min(subdomain_sizes_full)}")
        print(f"  Max: {max(subdomain_sizes_full)}")
        print(f"  Mean: {np.mean(subdomain_sizes_full):.1f}")
        print(f"  Median: {np.median(subdomain_sizes_full):.1f}")

    # Map subdomains to reduced space (0-indexed in the subspace)
    print("\nMapping subdomains to reduced space...")
    subdomains = map_subdomains_to_subspace(subdomains_full, dof_to_subspace)
    print(f"Number of subdomains in reduced space: {len(subdomains)}")

    if len(subdomains) > 0:
        subdomain_sizes = [len(s) for s in subdomains]
        print(f"Subdomain sizes (reduced space):")
        print(f"  Min: {min(subdomain_sizes)}")
        print(f"  Max: {max(subdomain_sizes)}")
        print(f"  Mean: {np.mean(subdomain_sizes):.1f}")
        print(f"  Median: {np.median(subdomain_sizes):.1f}")

        # Check that subdomain indices are valid
        max_subdomain_idx = max(np.max(s) for s in subdomains)
        print(f"  Max subdomain index: {max_subdomain_idx}")
        print(f"  Reduced space size: {A.shape[0]}")
        if max_subdomain_idx >= A.shape[0]:
            print(f"  ✗ ERROR: Max subdomain index ({max_subdomain_idx}) >= reduced space size ({A.shape[0]})")
        else:
            print(f"  ✓ All subdomain indices are valid")
    print()

    # Build Additive Schwarz operator in subspace
    print("Building Additive Schwarz operator M in subspace...")
    M, subdomain_info = build_additive_schwarz_operator(A, subdomains, check_eigenvalues=True)
    print(f"M shape: {M.shape}")
    print()

    # Report subdomain eigenvalue information
    print("Subdomain eigenvalue analysis:")
    print(f"  Singular subdomains: {subdomain_info['singular_count']}")
    print(f"  Subdomains with negative eigenvalues: {subdomain_info['negative_eigenvalue_count']}")

    if subdomain_info['negative_eigenvalue_count'] > 0:
        print()
        print("Details of subdomains with negative eigenvalues:")
        for info in subdomain_info['negative_eigenvalue_subdomains'][:10]:  # Show first 10
            print(f"  Subdomain {info['index']}: size={info['size']}, "
                  f"min_eigval={info['min_eigval']:.6e}, max_eigval={info['max_eigval']:.6e}, "
                  f"num_negative={info['num_negative']}, num_zero={info['num_zero']}")

        if subdomain_info['negative_eigenvalue_count'] > 10:
            print(f"  ... and {subdomain_info['negative_eigenvalue_count'] - 10} more subdomains with negative eigenvalues")
    print()

    # Check symmetry
    print("Checking symmetry...")
    is_symmetric, max_asymmetry = check_symmetry(M)
    print(f"  Is symmetric: {is_symmetric}")
    print(f"  Max asymmetry |M - M^T|: {max_asymmetry:.6e}")
    print()

    # Check positive definiteness
    print("Checking positive definiteness...")
    is_pd, eigenvalues, num_positive, num_zero, num_negative = check_positive_definite(M)

    print(f"  Is positive definite: {is_pd}")
    print(f"  Number of positive eigenvalues: {num_positive}")
    print(f"  Number of near-zero eigenvalues: {num_zero}")
    print(f"  Number of negative eigenvalues: {num_negative}")
    print()

    print("Eigenvalue statistics:")
    print(f"  Minimum: {eigenvalues[0]:.6e}")
    print(f"  Maximum: {eigenvalues[-1]:.6e}")
    print(f"  Condition number: {eigenvalues[-1] / max(eigenvalues[0], 1e-15):.6e}")
    print()

    # Show smallest and largest eigenvalues
    num_to_show = min(5, len(eigenvalues))
    print(f"Smallest {num_to_show} eigenvalues:")
    for i in range(num_to_show):
        print(f"  λ_{i+1} = {eigenvalues[i]:.6e}")
    print()

    print(f"Largest {num_to_show} eigenvalues:")
    for i in range(num_to_show):
        idx = len(eigenvalues) - num_to_show + i
        print(f"  λ_{idx+1} = {eigenvalues[idx]:.6e}")
    print()

    # Create visualization of eigenvalue spectrum
    print("Creating eigenvalue spectrum plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: All eigenvalues
    ax1.plot(eigenvalues, 'o-', markersize=3, linewidth=1)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, label='y=0')
    ax1.set_xlabel('Index', fontsize=14)
    ax1.set_ylabel('Eigenvalue', fontsize=14)
    ax1.set_title('Eigenvalue Spectrum of Additive Schwarz Operator M', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', labelsize=12)

    # Plot 2: Eigenvalues in log scale (absolute value)
    eigenvalues_abs = np.abs(eigenvalues)
    eigenvalues_abs_nonzero = eigenvalues_abs[eigenvalues_abs > 1e-15]

    if len(eigenvalues_abs_nonzero) > 0:
        ax2.semilogy(eigenvalues_abs, 'o-', markersize=3, linewidth=1)
        ax2.set_xlabel('Index', fontsize=14)
        ax2.set_ylabel('|Eigenvalue| (log scale)', fontsize=14)
        ax2.set_title('Absolute Eigenvalue Spectrum (log scale)', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both')
        ax2.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    # Save plot
    output_file = f'{args.output_prefix}_eigenvalues.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Eigenvalue plot saved to: {output_file}")

    plt.show()

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Matrix size: {M.shape[0]} x {M.shape[1]}")
    print(f"Symmetric: {is_symmetric} (max asymmetry: {max_asymmetry:.6e})")
    print(f"Positive Definite: {is_pd}")

    if is_symmetric and is_pd:
        print()
        print("✓ The Additive Schwarz operator M is SPD!")
    elif is_symmetric and not is_pd:
        print()
        print("✗ The operator is symmetric but NOT positive definite")
        print(f"  ({num_negative} negative, {num_zero} zero eigenvalues)")
    elif not is_symmetric:
        print()
        print("✗ The operator is NOT symmetric (cannot be SPD)")

    print("=" * 70)


if __name__ == "__main__":
    main()
