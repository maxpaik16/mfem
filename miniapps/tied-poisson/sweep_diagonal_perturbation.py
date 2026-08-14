#!/usr/bin/env python3
"""
Sweep diagonal perturbation parameters for tied-poisson miniapp.
Plots PCG iterations vs width for different centers and solver types.
"""

import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict

# Solver configurations
SOLVERS = {
    'AMG': [],
    'AMGF': ['-amgf'],
    'AMGF+Schwarz': ['-amgf', '-schwarz-filter']
}

# Plotting styles
STYLES = {
    'AMG': {'linestyle': '-', 'marker': 'o'},
    'AMGF': {'linestyle': '--', 'marker': 's'},
    'AMGF+Schwarz': {'linestyle': '-.', 'marker': '^'}
}

# Color palette for different centers
COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


def parse_pcg_iterations(output):
    """
    Parse PCG iteration count from tied-poisson output.
    Looks for lines like: "Iteration :  10  (B r, r) = 1.68152e-26"
    Returns the last iteration number found.
    """
    iterations = []
    pattern = r'Iteration\s*:\s*(\d+)\s+\(B r, r\)\s*=\s*[\d.e+-]+'

    for line in output.split('\n'):
        match = re.search(pattern, line)
        if match:
            iterations.append(int(match.group(1)))

    if iterations:
        return iterations[-1]
    else:
        return None


def run_tied_poisson(executable, base_args, width, center, solver_args, timeout=300):
    """
    Run the tied-poisson executable with given parameters.

    Args:
        executable: Path to tied-poisson executable
        base_args: Base arguments (mesh, refinement, alpha, etc.)
        width: Diagonal perturbation width
        center: Diagonal perturbation center
        solver_args: Solver-specific arguments
        timeout: Timeout in seconds

    Returns:
        Number of PCG iterations or None if failed
    """
    cmd = [executable] + base_args + [
        '-dp-width', str(width),
        '-dp-center', str(center)
    ] + solver_args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        iterations = parse_pcg_iterations(result.stdout)

        if iterations is None:
            print(f"  Warning: Could not parse iterations for width={width}, center={center}")
            print(f"  Command: {' '.join(cmd)}")

        return iterations

    except subprocess.TimeoutExpired:
        print(f"  Timeout for width={width}, center={center}")
        return None
    except Exception as e:
        print(f"  Error running width={width}, center={center}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Sweep diagonal perturbation parameters for tied-poisson'
    )
    parser.add_argument(
        '--executable',
        type=str,
        default='./tied-poisson',
        help='Path to tied-poisson executable'
    )
    parser.add_argument(
        '--mesh',
        type=str,
        default='beam-tet.mesh',
        help='Mesh file'
    )
    parser.add_argument(
        '--refine',
        type=int,
        default=4,
        help='Refinement levels'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1e6,
        help='Penalty parameter'
    )
    parser.add_argument(
        '--widths',
        type=str,
        default='0,50,100,200,500,1000,10000,100000,500000',
        help='Comma-separated list of widths to sweep'
    )
    parser.add_argument(
        '--centers',
        type=str,
        default='0,500,1000,100000,1000000',
        help='Comma-separated list of centers to sweep'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='perturbation_sweep.png',
        help='Output plot filename'
    )
    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--max-iters',
        type=int,
        default=10000,
        help='Maximum PCG iterations'
    )

    args = parser.parse_args()

    # Parse sweep parameters
    widths = [float(w) for w in args.widths.split(',')]
    centers = [float(c) for c in args.centers.split(',')]

    print(f"Sweeping {len(widths)} widths and {len(centers)} centers")
    print(f"Widths: {widths}")
    print(f"Centers: {centers}")
    print(f"Solvers: {list(SOLVERS.keys())}")

    # Base arguments for all runs
    base_args = [
        '-m', args.mesh,
        '-r', str(args.refine),
        '-a', str(args.alpha),
        '-p', str(args.max_iters),
        '-no-vis' if args.no_vis else '-vis'
    ]

    # Storage for results: results[solver][center][width] = iterations
    results = {solver: defaultdict(dict) for solver in SOLVERS.keys()}

    # Run sweep
    total_runs = len(widths) * len(centers) * len(SOLVERS)
    run_count = 0

    for solver_name, solver_args in SOLVERS.items():
        print(f"\n{'='*60}")
        print(f"Solver: {solver_name}")
        print(f"{'='*60}")

        for center in centers:
            print(f"\n  Center = {center}")

            for width in widths:
                run_count += 1
                print(f"    [{run_count}/{total_runs}] Width = {width}...", end=' ', flush=True)

                iterations = run_tied_poisson(
                    args.executable,
                    base_args,
                    width,
                    center,
                    solver_args
                )

                results[solver_name][center][width] = iterations

                if iterations is not None:
                    print(f"{iterations} iterations")
                else:
                    print("FAILED")

    # Create plot
    print(f"\n{'='*60}")
    print("Creating plot...")
    print(f"{'='*60}")

    fig, ax = plt.subplots(figsize=(12, 8))

    for solver_idx, (solver_name, solver_data) in enumerate(results.items()):
        for center_idx, (center, center_data) in enumerate(sorted(solver_data.items())):
            # Extract data points
            plot_widths = []
            plot_iters = []

            for width in sorted(center_data.keys()):
                iterations = center_data[width]
                if iterations is not None:
                    plot_widths.append(width)
                    plot_iters.append(iterations)

            if plot_widths:
                # Plot with unique color per center, style per solver
                color = COLORS[center_idx % len(COLORS)]
                style = STYLES[solver_name]

                label = f"{solver_name}, center={center:.0f}"

                ax.plot(
                    plot_widths,
                    plot_iters,
                    color=color,
                    linestyle=style['linestyle'],
                    marker=style['marker'],
                    markersize=8,
                    linewidth=2,
                    label=label,
                    alpha=0.8
                )

    ax.set_xlabel('Diagonal Perturbation Width', fontsize=14)
    ax.set_ylabel('PCG Iterations', fontsize=14)
    ax.set_title(
        f'PCG Convergence vs Diagonal Perturbation\n'
        f'α={args.alpha}, Refinement={args.refine}',
        fontsize=16
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', fontsize=10, ncol=2)

    # Use log scale if range is large
    if max(widths) / min([w for w in widths if w > 0] or [1]) > 10:
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {args.output}")

    # Print summary table
    print(f"\n{'='*60}")
    print("Summary Table")
    print(f"{'='*60}")
    print(f"{'Solver':<20} {'Center':<10} {'Width':<10} {'Iterations':<12}")
    print('-' * 60)

    for solver_name in SOLVERS.keys():
        for center in sorted(results[solver_name].keys()):
            for width in sorted(results[solver_name][center].keys()):
                iters = results[solver_name][center][width]
                iters_str = str(iters) if iters is not None else 'FAILED'
                print(f"{solver_name:<20} {center:<10.1f} {width:<10.1f} {iters_str:<12}")


if __name__ == '__main__':
    main()
