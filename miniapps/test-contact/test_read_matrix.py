#!/usr/bin/env python3
"""Test script to verify multi-rank matrix reading."""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analyze_schwarz_convergence import read_mfem_matrix

def test_multi_rank_read():
    """Test reading multi-rank matrix files."""

    # Test with multi-rank files from subsystems/pr0_sr0
    print("Testing multi-rank matrix reading...")
    print("=" * 60)

    test_dir = Path("subsystems/pr0_sr0")

    if not test_dir.exists():
        print(f"Error: Test directory not found: {test_dir}")
        return

    # Test reading D matrix
    print("\n1. Reading D matrix (diagonal matrix)...")
    d_file = test_dir / "D_matrix_iter_1.mat"
    D = read_mfem_matrix(d_file)
    print(f"   Shape: {D.shape}")
    print(f"   NNZ: {D.nnz}")
    print(f"   Diagonal (first 10): {D.diagonal()[:10]}")

    # Test reading J matrix
    print("\n2. Reading J matrix (constraint Jacobian)...")
    j_file = test_dir / "J_matrix_iter_1.mat"
    J = read_mfem_matrix(j_file)
    print(f"   Shape: {J.shape}")
    print(f"   NNZ: {J.nnz}")

    # Test reading P matrix
    print("\n3. Reading P matrix (transfer operator)...")
    p_file = test_dir / "P_matrix_iter_1.mat"
    P = read_mfem_matrix(p_file)
    print(f"   Shape: {P.shape}")
    print(f"   NNZ: {P.nnz}")

    print("\n" + "=" * 60)
    print("All tests passed!")

if __name__ == "__main__":
    test_multi_rank_read()
