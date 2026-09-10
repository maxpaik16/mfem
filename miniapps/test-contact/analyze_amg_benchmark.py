#!/usr/bin/env python3
"""
Analyze AMG benchmark results from amg-benchmark executable.

This script reads CSV output from amg-benchmark and generates comparison plots
showing setup time, solve time, iterations, and convergence behavior across
different solvers (BoomerAMG, AMGF, Schwarz-AMGF) and D-interpolation steps.

Usage:
    python analyze_amg_benchmark.py --csv results.csv
    python analyze_amg_benchmark.py --csv results.csv --output analysis_
    python analyze_amg_benchmark.py --csv results.csv --plot-type alpha  # Plot vs alpha
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def read_benchmark_csv(csv_file):
    """
    Read benchmark results from CSV file.

    Expected columns:
    Iteration, InterpStep, Alpha, Solver, SetupTime_s, SolveTime_s, Iterations, FinalResidual, Converged
    """
    # Try reading with header first
    df = pd.read_csv(csv_file)

    # If the first column is not named 'Iteration' and contains numeric data,
    # assume no header and provide column names
    if 'Iteration' not in df.columns:
        # Read again without header and assign column names
        df = pd.read_csv(csv_file, header=None,
                        names=['Iteration', 'InterpStep', 'Alpha', 'Solver',
                               'SetupTime_s', 'SolveTime_s', 'Iterations',
                               'FinalResidual', 'Converged'])

    # Verify required columns
    required_cols = ['Iteration', 'Solver', 'SetupTime_s', 'SolveTime_s',
                     'Iterations', 'FinalResidual', 'Converged']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Check if interpolation columns exist (new format)
    has_interpolation = 'InterpStep' in df.columns and 'Alpha' in df.columns

    return df, has_interpolation


def plot_solver_comparison(df, output_prefix='benchmark_'):
    """
    Create comparison plots for different solvers across iterations.
    """
    solvers = df['Solver'].unique()
    iterations = sorted(df['Iteration'].unique())

    # If multiple alpha values, use alpha=0 for comparison
    if 'Alpha' in df.columns:
        df = df[df['Alpha'] == 0.0]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('AMG Solver Benchmark Comparison', fontsize=16, fontweight='bold')

    # Define colors for each solver
    colors = {'BoomerAMG': 'blue', 'AMGF': 'green', 'Schwarz-AMGF': 'red'}
    markers = {'BoomerAMG': 'o', 'AMGF': 's', 'Schwarz-AMGF': '^'}

    # Plot 1: Setup Time
    ax1 = axes[0, 0]
    for solver in solvers:
        solver_data = df[df['Solver'] == solver]
        ax1.plot(solver_data['Iteration'], solver_data['SetupTime_s'],
                marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                label=solver, color=colors.get(solver, 'black'))
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Setup Time (s)', fontsize=12)
    ax1.set_title('Preconditioner Setup Time', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Solve Time
    ax2 = axes[0, 1]
    for solver in solvers:
        solver_data = df[df['Solver'] == solver]
        ax2.plot(solver_data['Iteration'], solver_data['SolveTime_s'],
                marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                label=solver, color=colors.get(solver, 'black'))
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Solve Time (s)', fontsize=12)
    ax2.set_title('PCG Solve Time', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: PCG Iterations
    ax3 = axes[1, 0]
    for solver in solvers:
        solver_data = df[df['Solver'] == solver]
        ax3.plot(solver_data['Iteration'], solver_data['Iterations'],
                marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                label=solver, color=colors.get(solver, 'black'))
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('PCG Iterations', fontsize=12)
    ax3.set_title('PCG Iteration Count', fontsize=13, fontweight='bold')
    ax3.set_yscale('log')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Total Time (Setup + Solve)
    ax4 = axes[1, 1]
    for solver in solvers:
        solver_data = df[df['Solver'] == solver]
        total_time = solver_data['SetupTime_s'] + solver_data['SolveTime_s']
        ax4.plot(solver_data['Iteration'], total_time,
                marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                label=solver, color=colors.get(solver, 'black'))
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Total Time (s)', fontsize=12)
    ax4.set_title('Total Time (Setup + Solve)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = f'{output_prefix}comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")

    return fig


def plot_alpha_sensitivity(df, output_prefix='benchmark_'):
    """
    Create plots showing solver performance vs alpha (D interpolation parameter).
    """
    if 'Alpha' not in df.columns:
        print("Warning: No Alpha column found. Skipping alpha sensitivity plots.")
        return None

    solvers = df['Solver'].unique()
    iterations = sorted(df['Iteration'].unique())

    # Define colors for each solver
    colors = {'BoomerAMG': 'blue', 'AMGF': 'green', 'Schwarz-AMGF': 'red'}
    markers = {'BoomerAMG': 'o', 'AMGF': 's', 'Schwarz-AMGF': '^'}

    # Create a separate figure for each iteration
    for iteration in iterations:
        iter_df = df[df['Iteration'] == iteration]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'D-Interpolation Sensitivity (Iteration {iteration})',
                     fontsize=16, fontweight='bold')

        # Plot 1: Solve Time vs Alpha
        ax1 = axes[0, 0]
        for solver in solvers:
            solver_data = iter_df[iter_df['Solver'] == solver].sort_values('Alpha')
            if len(solver_data) > 0:
                ax1.plot(solver_data['Alpha'], solver_data['SolveTime_s'],
                        marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                        label=solver, color=colors.get(solver, 'black'))
        ax1.set_xlabel('α (0=original D, 1=uniform D)', fontsize=12)
        ax1.set_ylabel('Solve Time (s)', fontsize=12)
        ax1.set_title('PCG Solve Time vs D Uniformity', fontsize=13, fontweight='bold')
        ax1.set_yscale('log')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, which='both')

        # Plot 2: PCG Iterations vs Alpha
        ax2 = axes[0, 1]
        for solver in solvers:
            solver_data = iter_df[iter_df['Solver'] == solver].sort_values('Alpha')
            if len(solver_data) > 0:
                ax2.plot(solver_data['Alpha'], solver_data['Iterations'],
                        marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                        label=solver, color=colors.get(solver, 'black'))
        ax2.set_xlabel('α (0=original D, 1=uniform D)', fontsize=12)
        ax2.set_ylabel('PCG Iterations', fontsize=12)
        ax2.set_title('PCG Iteration Count vs D Uniformity', fontsize=13, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, which='both')

        # Plot 3: Setup Time vs Alpha
        ax3 = axes[1, 0]
        for solver in solvers:
            solver_data = iter_df[iter_df['Solver'] == solver].sort_values('Alpha')
            if len(solver_data) > 0:
                ax3.plot(solver_data['Alpha'], solver_data['SetupTime_s'],
                        marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                        label=solver, color=colors.get(solver, 'black'))
        ax3.set_xlabel('α (0=original D, 1=uniform D)', fontsize=12)
        ax3.set_ylabel('Setup Time (s)', fontsize=12)
        ax3.set_title('Preconditioner Setup Time vs D Uniformity', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Convergence Status
        ax4 = axes[1, 1]
        for solver in solvers:
            solver_data = iter_df[iter_df['Solver'] == solver].sort_values('Alpha')
            if len(solver_data) > 0:
                # Convert Converged boolean to int for plotting
                converged = solver_data['Converged'].map({'True': 1, True: 1, 'False': 0, False: 0})
                ax4.plot(solver_data['Alpha'], converged,
                        marker=markers.get(solver, 'o'), linewidth=2, markersize=10,
                        label=solver, color=colors.get(solver, 'black'))
        ax4.set_xlabel('α (0=original D, 1=uniform D)', fontsize=12)
        ax4.set_ylabel('Converged (1=Yes, 0=No)', fontsize=12)
        ax4.set_title('Convergence Status vs D Uniformity', fontsize=13, fontweight='bold')
        ax4.set_ylim([-0.1, 1.1])
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        output_file = f'{output_prefix}alpha_iter{iteration}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Alpha sensitivity plot (iteration {iteration}) saved to: {output_file}")

    return fig


def plot_speedup_analysis(df, baseline_solver='BoomerAMG', output_prefix='benchmark_'):
    """
    Create speedup analysis plots comparing solvers to baseline.
    """
    # If multiple alpha values, use alpha=0 for comparison
    if 'Alpha' in df.columns:
        df = df[df['Alpha'] == 0.0]

    iterations = sorted(df['Iteration'].unique())
    solvers = [s for s in df['Solver'].unique() if s != baseline_solver]

    if baseline_solver not in df['Solver'].values:
        print(f"Warning: Baseline solver '{baseline_solver}' not found in data.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Speedup Analysis (Baseline: {baseline_solver})',
                 fontsize=16, fontweight='bold')

    # Get baseline data
    baseline_data = df[df['Solver'] == baseline_solver].set_index('Iteration')

    colors = {'AMGF': 'green', 'Schwarz-AMGF': 'red'}
    markers = {'AMGF': 's', 'Schwarz-AMGF': '^'}

    # Plot 1: Solve Time Speedup
    ax1 = axes[0]
    for solver in solvers:
        solver_data = df[df['Solver'] == solver].set_index('Iteration')
        speedup = baseline_data['SolveTime_s'] / solver_data['SolveTime_s']
        ax1.plot(speedup.index, speedup.values,
                marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                label=solver, color=colors.get(solver, 'black'))
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Speedup (×)', fontsize=12)
    ax1.set_title('Solve Time Speedup', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Iteration Reduction
    ax2 = axes[1]
    for solver in solvers:
        solver_data = df[df['Solver'] == solver].set_index('Iteration')
        reduction = baseline_data['Iterations'] / solver_data['Iterations']
        ax2.plot(reduction.index, reduction.values,
                marker=markers.get(solver, 'o'), linewidth=2, markersize=8,
                label=solver, color=colors.get(solver, 'black'))
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, label='Baseline')
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Iteration Reduction (×)', fontsize=12)
    ax2.set_title('PCG Iteration Reduction', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_file = f'{output_prefix}speedup.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Speedup plot saved to: {output_file}")

    return fig


def print_summary_statistics(df, has_interpolation=False):
    """
    Print summary statistics for all solvers.
    """
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)

    # If interpolation data exists, show statistics for alpha=0 and alpha=1
    if has_interpolation and 'Alpha' in df.columns:
        alpha_values = sorted(df['Alpha'].unique())
        print(f"\nInterpolation steps: {len(alpha_values)} (α = {alpha_values})")

        for alpha in [alpha_values[0], alpha_values[-1]]:
            alpha_df = df[df['Alpha'] == alpha]
            print(f"\n--- α = {alpha:.2f} {'(original D)' if alpha == 0 else '(uniform D)'} ---")

            for solver in alpha_df['Solver'].unique():
                solver_data = alpha_df[alpha_df['Solver'] == solver]

                print(f"\n{solver}:")
                print(f"  Setup Time:    mean={solver_data['SetupTime_s'].mean():.4e} s, "
                      f"std={solver_data['SetupTime_s'].std():.4e} s")
                print(f"  Solve Time:    mean={solver_data['SolveTime_s'].mean():.4e} s, "
                      f"std={solver_data['SolveTime_s'].std():.4e} s")
                print(f"  Total Time:    mean={(solver_data['SetupTime_s'] + solver_data['SolveTime_s']).mean():.4e} s")
                print(f"  Iterations:    mean={solver_data['Iterations'].mean():.2f}, "
                      f"std={solver_data['Iterations'].std():.2f}")
                converged_count = solver_data['Converged'].map({'True': 1, True: 1, 'False': 0, False: 0}).sum()
                print(f"  Converged:     {converged_count}/{len(solver_data)} iterations")
    else:
        for solver in df['Solver'].unique():
            solver_data = df[df['Solver'] == solver]

            print(f"\n{solver}:")
            print(f"  Setup Time:    mean={solver_data['SetupTime_s'].mean():.4e} s, "
                  f"std={solver_data['SetupTime_s'].std():.4e} s")
            print(f"  Solve Time:    mean={solver_data['SolveTime_s'].mean():.4e} s, "
                  f"std={solver_data['SolveTime_s'].std():.4e} s")
            print(f"  Total Time:    mean={(solver_data['SetupTime_s'] + solver_data['SolveTime_s']).mean():.4e} s")
            print(f"  Iterations:    mean={solver_data['Iterations'].mean():.2f}, "
                  f"std={solver_data['Iterations'].std():.2f}")
            converged_count = solver_data['Converged'].map({'True': 1, True: 1, 'False': 0, False: 0}).sum()
            print(f"  Converged:     {converged_count}/{len(solver_data)} iterations")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze AMG benchmark results with D-interpolation support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--csv', type=str, required=True,
                        help='CSV file with benchmark results')
    parser.add_argument('--output-prefix', type=str, default='benchmark_',
                        help='Prefix for output files')
    parser.add_argument('--baseline', type=str, default='BoomerAMG',
                        help='Baseline solver for speedup analysis')
    parser.add_argument('--plot-type', type=str, default='both',
                        choices=['iteration', 'alpha', 'both'],
                        help='Type of plots to generate: iteration (vs iteration), alpha (vs alpha), or both')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    args = parser.parse_args()

    # Read data
    print(f"Reading benchmark results from: {args.csv}")
    df, has_interpolation = read_benchmark_csv(args.csv)

    print(f"Loaded {len(df)} benchmark results")
    print(f"Solvers: {', '.join(df['Solver'].unique())}")
    print(f"Iterations: {sorted(df['Iteration'].unique())}")
    if has_interpolation:
        print(f"Interpolation steps: {len(df['InterpStep'].unique())}")
        print(f"Alpha values: {sorted(df['Alpha'].unique())}")

    # Print summary statistics
    print_summary_statistics(df, has_interpolation)

    if not args.no_plots:
        if args.plot_type in ['iteration', 'both']:
            # Generate comparison plots (vs iteration)
            print("\nGenerating comparison plots (vs iteration)...")
            plot_solver_comparison(df, output_prefix=args.output_prefix)

            # Generate speedup analysis
            print("Generating speedup analysis...")
            plot_speedup_analysis(df, baseline_solver=args.baseline,
                                 output_prefix=args.output_prefix)

        if args.plot_type in ['alpha', 'both'] and has_interpolation:
            # Generate alpha sensitivity plots
            print("\nGenerating D-interpolation sensitivity plots...")
            plot_alpha_sensitivity(df, output_prefix=args.output_prefix)

        print("\nAnalysis complete!")
        plt.show()


if __name__ == "__main__":
    main()
