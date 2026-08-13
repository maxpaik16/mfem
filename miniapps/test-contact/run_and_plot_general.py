#!/usr/bin/env python3
"""
General experiment runner for the updated test-contact miniapp.

Features:
- Runs arbitrary parameter sweeps
- Parses PCG iterations and linear solve times
- Saves raw output and structured metadata
- Plots both per-solve behavior and aggregate summaries
- Supports grouping runs into curves by arbitrary parameters

Example uses:
- Compare solver variants
- Sweep refinement levels
- Sweep MPI ranks
- Sweep Schwarz weight / variant / cg iterations
- Compare problems or nonlinear vs linear modes
"""

import argparse
import hashlib
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Professional academic presentation style
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['figure.titlesize'] = 20
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams['lines.markersize'] = 8
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['grid.linewidth'] = 0.8
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.grid'] = True
mpl.rcParams['axes.axisbelow'] = True
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.facecolor'] = 'white'

# Professional color palette (colorblind-friendly)
COLORS = ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#D55E00', '#949494', '#ECE133', '#56B4E9']


# =============================================================================
# USER CONFIGURATION
# =============================================================================

BASE_CONFIG = {
    "np": 4,
    "prob": 0,
    "model": "linear",          # "linear" or "nonlinear"
    "sr": 1,
    "pr": 0,
    "nsteps": 4,
    "msteps": 0,
    "tr": 2.0,
    "vis": False,
    "paraview": False,
    "amgf": True,
    "amgf_reversed": False,
    "amgf_fsolver": "auto",
    "schwarz": True,
    "schwarz_expand": False,
    "schwarz_cg_iters": 0,
    "schwarz_variant": 2,
    "schwarz_weight": 1.0,
    "schwarz_min_diag": 0.0,
    "schwarz_uniform_weight": 0.1,
    "hybrid_amg": False,
    "subspace_pl": 0
}

# Arbitrary parameter sweep.
# Each key maps to a list of values to test.
SWEEP_PARAMETERS = {
    # Example solver sweep:
    #"amgf": [True, False],
    #"schwarz": [True, False],
    #"schwarz_variant": [2],
    #"schwarz_uniform_weight": [0.05],
    #"amgf_reversed": [True, False],
    #"hybrid_amg": [True, False]

    "schwarz_min_diag": [0, 1, 1e1, 1e2, 1e12]

    # Uncomment for other experiments:
    #"np": [1, 2, 4, 8],
    #"sr": [0, 1, 2],
    # "pr": [0, 1],
    # "prob": [0, 1, 2],
}

# Which parameters define distinct curves in plots
CURVE_KEYS = [
    #"amgf",
    #"schwarz",
    #sr
    "schwarz_min_diag"
]

# Which parameter should be used for x-axis in summary plots.
# Options:
#   - a parameter name from configs, such as "sr", "np", "pr", "schwarz_weight"
#   - "run_index", meaning just one point per experiment in run order
X_AXIS_MODE = "sr"

# Output directory prefix
OUTPUT_PREFIX = "contact_experiments"

# Executable path
EXECUTABLE = "./test-contact"

# Timeout per run
TIMEOUT_SECONDS = 3600

PROBLEM_NAMES = {
    0: "two-block",
    1: "ironing",
    2: "beam-sphere",
}


# =============================================================================
# UTILS
# =============================================================================

def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def short_hash(obj, n=10):
    return hashlib.md5(canonical_json(obj).encode()).hexdigest()[:n]


def sanitize_filename(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def build_output_dir(base_config, sweep_parameters, user_output_dir=None):
    if user_output_dir:
        return Path(user_output_dir)
    signature = {"base": base_config}
    return Path(f"{OUTPUT_PREFIX}_{short_hash(signature)}")


def expand_sweep(base_config, sweep_parameters):
    if not sweep_parameters:
        return [base_config.copy()]

    keys = list(sweep_parameters.keys())
    values_product = itertools.product(*(sweep_parameters[k] for k in keys))

    configs = []
    for values in values_product:
        cfg = base_config.copy()
        for k, v in zip(keys, values):
            cfg[k] = v
        configs.append(cfg)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for cfg in configs:
        key = canonical_json(cfg)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    return unique


def format_solver_label(config):
    if not config.get("amgf", False):
        return "AMG"

    base_label = ""
    if config.get("schwarz", False):
        cg_iters = config.get("schwarz_cg_iters", 0)
        if cg_iters and cg_iters > 0:
            base_label = f"AMGF + Schwarz-CG({cg_iters})"
        else:
            base_label = "AMGF + Schwarz"

        if config.get("hybrid_amg", False):
            base_label += " + Hybrid AMG"
    else:
        fsolver = config.get("amgf_fsolver", "auto")
        if fsolver == "auto":
            base_label = "AMGF + direct subspace"
        else:
            base_label = f"AMGF + {fsolver}"

        # Add "Hybrid AMG" even when using direct subspace solver
        if config.get("hybrid_amg", False):
            base_label += " + Hybrid AMG"

    if config.get("amgf_reversed", False):
        base_label += " (reversed)"

    return base_label


def format_label_component(key, config):
    value = config.get(key)

    if key == "amgf":
        return None
    if key == "amgf_reversed":
        return "reversed order" if value else None
    if key == "schwarz":
        return None
    if key == "hybrid_amg":
        return "hybrid AMG smoother" if value else None
    if key == "amgf_fsolver":
        return None if config.get("schwarz", False) else f"subspace={value}"
    if key == "np":
        return f"{value} ranks"
    if key == "prob":
        return f"problem={PROBLEM_NAMES.get(value, value)}"
    if key == "model":
        return "nonlinear" if value == "nonlinear" else "linear"
    if key == "sr":
        return f"sr={value}"
    if key == "pr":
        return f"pr={value}"
    if key == "nsteps":
        return f"nsteps={value}"
    if key == "msteps":
        return f"msteps={value}"
    if key == "tr":
        return f"tr={value:g}" if isinstance(value, float) else f"tr={value}"
    if key == "schwarz_expand":
        return "expanded Schwarz" if value else "unexpanded Schwarz"
    if key == "schwarz_cg_iters":
        return None if not config.get("schwarz", False) else (
            f"Schwarz-CG iters={value}" if value > 0 else "direct Schwarz"
        )
    if key == "schwarz_variant":
        return f"variant={value}"
    if key == "schwarz_weight":
        return f"weight={value:g}" if isinstance(value, float) else f"weight={value}"
    if key == "schwarz_min_diag":
        return f"min diag={value:g}" if isinstance(value, float) else f"min diag={value}"
    if key == "schwarz_uniform_weight":
        return f"uniform w={value:g}" if isinstance(value, float) else f"uniform w={value}"
    if key == "subspace_pl":
        return f"print level={value}"
    if key == "vis":
        return "vis" if value else "no vis"
    if key == "paraview":
        return "paraview" if value else "no paraview"

    if isinstance(value, bool):
        return key if value else f"no {key}"
    return f"{key}={value}"


def format_dofs(dofs):
    """Format degrees of freedom with K/M abbreviation."""
    if dofs is None:
        return None
    if dofs >= 1_000_000:
        return f"{dofs / 1_000_000:.1f}M"
    elif dofs >= 1_000:
        return f"{dofs / 1_000:.0f}K"
    else:
        return str(dofs)


def config_to_label(config, keys, dofs=None):
    parts = []

    if any(k in keys for k in ("amgf", "schwarz", "amgf_fsolver", "schwarz_cg_iters")):
        parts.append(format_solver_label(config))

    for key in keys:
        component = format_label_component(key, config)
        if component and component not in parts:
            parts.append(component)

    label = ", ".join(parts) if parts else format_solver_label(config)

    # Append dofs if provided
    if dofs is not None:
        dofs_str = format_dofs(dofs)
        if dofs_str:
            label += f" ({dofs_str} dofs)"

    return label


def config_to_command(config):
    cmd = [
        "mpirun", "-np", str(config["np"]),
        EXECUTABLE,
        "-prob", str(config["prob"]),
        "--nonlinear" if config["model"] == "nonlinear" else "--linear",
        "-sr", str(config["sr"]),
        "-pr", str(config["pr"]),
        "-nsteps", str(config["nsteps"]),
        "-msteps", str(config["msteps"]),
        "-tr", str(config["tr"]),
        "--visualization" if config["vis"] else "--no-visualization",
        "--paraview" if config["paraview"] else "--no-paraview",
        "--amgf" if config["amgf"] else "--no-amgf",
        "--amgf-reversed" if config["amgf_reversed"] else "--no-amgf-reversed",
        "-amgf-fsolver", str(config["amgf_fsolver"]),
        "--schwarz" if config["schwarz"] else "--no-schwarz",
        "--schwarz-expand" if config["schwarz_expand"] else "--no-schwarz-expand",
        "-schwarz-cg-iters", str(config["schwarz_cg_iters"]),
        "-schwarz-variant", str(config["schwarz_variant"]),
        "-schwarz-weight", str(config["schwarz_weight"]),
        "-schwarz-min-diag", str(config["schwarz_min_diag"]),
        "-schwarz-uniform-weight", str(config["schwarz_uniform_weight"]),
        "-hybrid-amg" if config["hybrid_amg"] else "-no-hybrid-amg",
        "-subspace-pl", str(config["subspace_pl"])
    ]
    return cmd


def run_command(cmd, timeout=TIMEOUT_SECONDS):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "combined_output": result.stdout + result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -999,
            "stdout": "",
            "stderr": f"Timeout after {timeout} seconds",
            "combined_output": "",
        }
    except Exception as e:
        return {
            "returncode": -998,
            "stdout": "",
            "stderr": str(e),
            "combined_output": "",
        }


# =============================================================================
# PARSING
# =============================================================================

def parse_int_list_line(text, label):
    pattern = rf"{re.escape(label)}\s*=\s*([0-9\s]+)"
    matches = re.findall(pattern, text)
    values = []
    for match in matches:
        values.extend(int(x) for x in match.split())
    return values


def parse_float_list_line(text, label):
    pattern = rf"{re.escape(label)}\s*=\s*([0-9eE+.\-\s]+)"
    matches = re.findall(pattern, text)
    values = []
    for match in matches:
        values.extend(float(x) for x in match.split())
    return values


def parse_scalar_int(text, label):
    pattern = rf"{re.escape(label)}\s*=\s*(\d+)"
    match = re.search(pattern, text)
    return int(match.group(1)) if match else None


def parse_scalar_float(text, label):
    pattern = rf"{re.escape(label)}\s*=\s*([0-9eE+.\-]+)"
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def parse_scalar_int_with_colon(text, label):
    """Parse all occurrences of 'label: value' and return the average."""
    pattern = rf"{re.escape(label)}\s*:\s*(\d+)"
    matches = re.findall(pattern, text)
    if matches:
        values = [int(m) for m in matches]
        return sum(values) / len(values)
    return None


def parse_amgf_setup_breakdown(text):
    pattern = re.compile(
        r"AMGF filtered setup time \[s\]: total=([0-9eE+.\-]+), "
        r"AMG=([0-9eE+.\-]+), subspace=([0-9eE+.\-]+) "
        r"\(PtAP=([0-9eE+.\-]+), solver=([0-9eE+.\-]+)\)"
    )
    breakdown = []
    for match in pattern.finditer(text):
        breakdown.append({
            "total": float(match.group(1)),
            "amg": float(match.group(2)),
            "subspace": float(match.group(3)),
            "ptap": float(match.group(4)),
            "solver": float(match.group(5)),
        })
    return breakdown


def parse_amgf_solve_breakdown(text):
    pattern = re.compile(
        r"AMGF solve time split \[s\]: total=([0-9eE+.\-]+), "
        r"AMG=([0-9eE+.\-]+) \([0-9eE+.\-]+%\), "
        r"subspace=([0-9eE+.\-]+) \([0-9eE+.\-]+%\), "
        r"other=([0-9eE+.\-]+) \([0-9eE+.\-]+%\), applies=(\d+)"
    )
    breakdown = []
    for match in pattern.finditer(text):
        breakdown.append({
            "total": float(match.group(1)),
            "amg": float(match.group(2)),
            "subspace": float(match.group(3)),
            "other": float(match.group(4)),
            "applies": int(match.group(5)),
        })
    return breakdown


def sum_breakdown_entries(entries, keys):
    return {key: float(sum(entry[key] for entry in entries)) for key in keys}


def parse_run_output(output_text):
    amgf_setup_breakdown = parse_amgf_setup_breakdown(output_text)
    amgf_solve_breakdown = parse_amgf_solve_breakdown(output_text)

    data = {
        "optimizer_iterations": parse_scalar_int(output_text, "Optimizer number of iterations"),
        "initial_energy": parse_scalar_float(output_text, "Initial Energy objective"),
        "final_energy": parse_scalar_float(output_text, "Final Energy objective"),
        "pcg_iterations": parse_int_list_line(output_text, "PCG number of iterations"),
        "linear_solve_times": parse_float_list_line(output_text, "Linear solve times [s]"),
        "amgf_setup_breakdown": amgf_setup_breakdown,
        "amgf_solve_breakdown": amgf_solve_breakdown,
        "global_dofs": parse_scalar_int(output_text, "Global number of dofs"),
        "schwarz_subdomains_avg": parse_scalar_int_with_colon(output_text, "Total number of subdomains"),
    }

    data["num_linear_solves_from_iters"] = len(data["pcg_iterations"])
    data["num_linear_solves_from_times"] = len(data["linear_solve_times"])

    if data["pcg_iterations"]:
        data["pcg_total"] = int(sum(data["pcg_iterations"]))
        data["pcg_mean"] = float(np.mean(data["pcg_iterations"]))
        data["pcg_median"] = float(np.median(data["pcg_iterations"]))
        data["pcg_min"] = int(min(data["pcg_iterations"]))
        data["pcg_max"] = int(max(data["pcg_iterations"]))
    else:
        data["pcg_total"] = None
        data["pcg_mean"] = None
        data["pcg_median"] = None
        data["pcg_min"] = None
        data["pcg_max"] = None

    if data["linear_solve_times"]:
        data["time_total"] = float(sum(data["linear_solve_times"]))
        data["time_mean"] = float(np.mean(data["linear_solve_times"]))
        data["time_median"] = float(np.median(data["linear_solve_times"]))
        data["time_min"] = float(min(data["linear_solve_times"]))
        data["time_max"] = float(max(data["linear_solve_times"]))
    else:
        data["time_total"] = None
        data["time_mean"] = None
        data["time_median"] = None
        data["time_min"] = None
        data["time_max"] = None

    if data["pcg_iterations"] and data["linear_solve_times"]:
        n = min(len(data["pcg_iterations"]), len(data["linear_solve_times"]))
        if n > 0:
            ratios = [data["linear_solve_times"][i] / data["pcg_iterations"][i]
                      for i in range(n) if data["pcg_iterations"][i] > 0]
            data["time_per_iter_mean"] = float(np.mean(ratios)) if ratios else None
        else:
            data["time_per_iter_mean"] = None
    else:
        data["time_per_iter_mean"] = None

    data["num_amgf_setup_entries"] = len(amgf_setup_breakdown)
    data["num_amgf_solve_entries"] = len(amgf_solve_breakdown)

    if amgf_setup_breakdown:
        totals = sum_breakdown_entries(amgf_setup_breakdown, ["total", "amg", "subspace", "ptap", "solver"])
        data["amgf_setup_total_sum"] = totals["total"]
        data["amgf_setup_amg_sum"] = totals["amg"]
        data["amgf_setup_subspace_sum"] = totals["subspace"]
        data["amgf_setup_ptap_sum"] = totals["ptap"]
        data["amgf_setup_solver_sum"] = totals["solver"]
        data["amgf_setup_total_mean"] = totals["total"] / len(amgf_setup_breakdown)
    else:
        data["amgf_setup_total_sum"] = None
        data["amgf_setup_amg_sum"] = None
        data["amgf_setup_subspace_sum"] = None
        data["amgf_setup_ptap_sum"] = None
        data["amgf_setup_solver_sum"] = None
        data["amgf_setup_total_mean"] = None

    if amgf_solve_breakdown:
        totals = sum_breakdown_entries(amgf_solve_breakdown, ["total", "amg", "subspace", "other"])
        data["amgf_solve_total_sum"] = totals["total"]
        data["amgf_solve_amg_sum"] = totals["amg"]
        data["amgf_solve_subspace_sum"] = totals["subspace"]
        data["amgf_solve_other_sum"] = totals["other"]
        data["amgf_solve_total_mean"] = totals["total"] / len(amgf_solve_breakdown)
        if totals["total"] > 0.0:
            data["amgf_solve_amg_frac"] = totals["amg"] / totals["total"]
            data["amgf_solve_subspace_frac"] = totals["subspace"] / totals["total"]
            data["amgf_solve_other_frac"] = totals["other"] / totals["total"]
        else:
            data["amgf_solve_amg_frac"] = None
            data["amgf_solve_subspace_frac"] = None
            data["amgf_solve_other_frac"] = None
    else:
        data["amgf_solve_total_sum"] = None
        data["amgf_solve_amg_sum"] = None
        data["amgf_solve_subspace_sum"] = None
        data["amgf_solve_other_sum"] = None
        data["amgf_solve_total_mean"] = None
        data["amgf_solve_amg_frac"] = None
        data["amgf_solve_subspace_frac"] = None
        data["amgf_solve_other_frac"] = None

    return data


# =============================================================================
# DATA STORAGE
# =============================================================================

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_run_artifacts(output_dir, run_id, config, result, parsed):
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "config.json", config)
    save_json(run_dir / "parsed.json", parsed)

    (run_dir / "stdout.txt").write_text(result["stdout"])
    (run_dir / "stderr.txt").write_text(result["stderr"])
    (run_dir / "combined_output.txt").write_text(result["combined_output"])

    metadata = {
        "run_id": run_id,
        "returncode": result["returncode"],
    }
    save_json(run_dir / "run_metadata.json", metadata)


def load_existing_runs(output_dir, base_config=None, sweep_parameters=None):
    runs_root = output_dir / "runs"
    records = []

    if not runs_root.exists():
        return records

    # Generate expected run IDs based on current sweep parameters
    expected_run_ids = None
    if base_config is not None and sweep_parameters is not None:
        configs = expand_sweep(base_config, sweep_parameters)
        expected_run_ids = {
            f"run_{i:03d}_{short_hash(config)}"
            for i, config in enumerate(configs)
        }

    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir():
            continue

        # If we have expected run IDs, only load those
        if expected_run_ids is not None and run_dir.name not in expected_run_ids:
            continue

        config_file = run_dir / "config.json"
        parsed_file = run_dir / "parsed.json"
        meta_file = run_dir / "run_metadata.json"

        if config_file.exists() and parsed_file.exists() and meta_file.exists():
            parsed = load_json(parsed_file)

            # If global_dofs or schwarz_subdomains_avg is missing or None, try to parse from stdout
            if parsed.get("global_dofs") is None or parsed.get("schwarz_subdomains_avg") is None:
                stdout_file = run_dir / "stdout.txt"
                if stdout_file.exists():
                    stdout_text = stdout_file.read_text()
                    if parsed.get("global_dofs") is None:
                        parsed["global_dofs"] = parse_scalar_int(stdout_text, "Global number of dofs")
                        if parsed["global_dofs"] is not None:
                            print(f"  Re-parsed global_dofs={parsed['global_dofs']} for {run_dir.name}")
                    if parsed.get("schwarz_subdomains_avg") is None:
                        parsed["schwarz_subdomains_avg"] = parse_scalar_int_with_colon(stdout_text, "Total number of subdomains")

            records.append({
                "run_id": run_dir.name,
                "config": load_json(config_file),
                "parsed": parsed,
                "metadata": load_json(meta_file),
            })

    return records


# =============================================================================
# PLOTTING
# =============================================================================

def group_records_by_curve(records, curve_keys, include_dofs=False):
    grouped = {}
    for record in records:
        dofs = record["parsed"].get("global_dofs") if include_dofs else None
        label = config_to_label(record["config"], curve_keys, dofs)
        grouped.setdefault(label, []).append(record)
    return grouped


def get_x_value(record, x_axis_mode, fallback_index):
    if x_axis_mode == "run_index":
        return fallback_index
    # If x_axis_mode is "sr", use DOFs instead
    if x_axis_mode == "sr":
        dofs = record["parsed"].get("global_dofs")
        return dofs if dofs is not None else fallback_index
    return record["config"].get(x_axis_mode, fallback_index)


def sort_curve_records(records, x_axis_mode):
    decorated = []
    for i, rec in enumerate(records):
        x = get_x_value(rec, x_axis_mode, i)
        decorated.append((x, rec))
    decorated.sort(key=lambda t: t[0])
    return [rec for _, rec in decorated]


def plot_per_solve_curves(records, output_dir):
    valid = [r for r in records if r["parsed"]["pcg_iterations"] or r["parsed"]["linear_solve_times"]]
    if not valid:
        print("No per-solve data to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, rec in enumerate(valid):
        dofs = rec["parsed"].get("global_dofs")
        label = config_to_label(rec["config"], CURVE_KEYS, dofs)
        color = COLORS[idx % len(COLORS)]
        pcg = rec["parsed"]["pcg_iterations"]
        times = rec["parsed"]["linear_solve_times"]

        if pcg:
            x = np.arange(1, len(pcg) + 1)
            axes[0].plot(x, pcg, marker="o", color=color, label=label, markeredgewidth=0.5, markeredgecolor='white')

        if times:
            x = np.arange(1, len(times) + 1)
            axes[1].plot(x, times, marker="o", color=color, label=label, markeredgewidth=0.5, markeredgecolor='white')

    axes[0].set_title("PCG Iterations per Linear Solve", fontweight='bold', pad=12)
    axes[0].set_xlabel("Linear Solve Index", fontweight='semibold')
    axes[0].set_ylabel("PCG Iterations", fontweight='semibold')
    axes[0].set_yscale("log")
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.6)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    axes[1].set_title("Linear Solve Time per Solve", fontweight='bold', pad=12)
    axes[1].set_xlabel("Linear Solve Index", fontweight='semibold')
    axes[1].set_ylabel("Time (s)", fontweight='semibold')
    axes[1].set_yscale("log")
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.6)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    for ax in axes:
        ax.legend(frameon=True, fancybox=False, edgecolor='gray', framealpha=0.95)

    plt.tight_layout()
    out = output_dir / "per_solve_curves.png"
    plt.savefig(out)
    print(f"Saved {out}")
    plt.show()


def plot_summary_curves(records, output_dir, x_axis_mode):
    valid = [r for r in records if r["metadata"]["returncode"] == 0]
    if not valid:
        print("No successful runs to summarize.")
        return

    # Don't include dofs in grouping/labels if x-axis is sr (which will be plotted as dofs)
    include_dofs_in_label = (x_axis_mode != "sr")
    grouped = group_records_by_curve(valid, CURVE_KEYS, include_dofs=include_dofs_in_label)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    metrics = [
        ("pcg_total", "Total PCG Iterations"),
        ("pcg_mean", "Mean PCG Iterations per Solve"),
        ("time_total", "Total Linear Solve Time (s)"),
        ("time_mean", "Mean Linear Solve Time (s)"),
    ]

    for ax, (metric_key, metric_title) in zip(axes, metrics):
        for idx, (label, recs) in enumerate(grouped.items()):
            color = COLORS[idx % len(COLORS)]
            recs_sorted = sort_curve_records(recs, x_axis_mode)

            xvals = []
            yvals = []

            for i, rec in enumerate(recs_sorted):
                x = get_x_value(rec, x_axis_mode, i)
                y = rec["parsed"].get(metric_key)
                if y is not None:
                    xvals.append(x)
                    yvals.append(y)

            if xvals and yvals:
                ax.plot(xvals, yvals, marker="o", color=color, label=label,
                       markeredgewidth=0.5, markeredgecolor='white')

        ax.set_title(metric_title, fontweight='bold', pad=12)
        # If x_axis_mode is sr, label x-axis as DOFs and use log scale
        x_label = "DOFs" if x_axis_mode == "sr" else x_axis_mode.replace('_', ' ').title()
        ax.set_xlabel(x_label, fontweight='semibold')
        ax.set_ylabel(metric_title.split('(')[0].strip(), fontweight='semibold')
        if x_axis_mode == "sr":
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which='both', linestyle='--', linewidth=0.6)
        ax.minorticks_on()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=True, fancybox=False, edgecolor='gray', framealpha=0.95)

    plt.tight_layout()
    out = output_dir / "summary_curves.png"
    plt.savefig(out)
    print(f"Saved {out}")
    plt.show()


def plot_scatter_time_vs_iterations(records, output_dir):
    valid = [r for r in records if r["parsed"]["pcg_total"] is not None and r["parsed"]["time_total"] is not None]
    if not valid:
        print("No aggregate timing vs iteration data to plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Don't include dofs in labels for scatter plot
    grouped = group_records_by_curve(valid, CURVE_KEYS, include_dofs=False)
    for idx, (label, recs) in enumerate(grouped.items()):
        color = COLORS[idx % len(COLORS)]
        x = [r["parsed"]["pcg_total"] for r in recs]
        y = [r["parsed"]["time_total"] for r in recs]
        ax.scatter(x, y, s=120, alpha=0.85, label=label, color=color,
                  edgecolors='white', linewidth=0.5)

    ax.set_xlabel("Total PCG Iterations", fontweight='semibold')
    ax.set_ylabel("Total Linear Solve Time (s)", fontweight='semibold')
    ax.set_title("Total Time vs Total Iterations", fontweight='bold', pad=12)
    ax.grid(True, linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, fancybox=False, edgecolor='gray', framealpha=0.95)

    plt.tight_layout()
    out = output_dir / "time_vs_iterations.png"
    plt.savefig(out)
    print(f"Saved {out}")
    plt.show()


def build_run_tick_label(record, x_axis_mode):
    keys = list(CURVE_KEYS)
    if x_axis_mode != "run_index" and x_axis_mode not in keys:
        keys.append(x_axis_mode)
    return config_to_label(record["config"], keys)


def plot_timing_breakdown(records, output_dir, x_axis_mode):
    valid = [r for r in records if r["parsed"]["amgf_solve_total_sum"] is not None]
    if not valid:
        print("No AMGF timing breakdown data to plot.")
        return

    ordered = sort_curve_records(valid, x_axis_mode)
    labels = [build_run_tick_label(rec, x_axis_mode) for rec in ordered]
    x = np.arange(len(ordered))

    solve_amg = np.array([rec["parsed"]["amgf_solve_amg_sum"] for rec in ordered], dtype=float)
    solve_subspace = np.array([rec["parsed"]["amgf_solve_subspace_sum"] for rec in ordered], dtype=float)
    solve_other = np.array([rec["parsed"]["amgf_solve_other_sum"] for rec in ordered], dtype=float)

    setup_ptap = np.array([
        rec["parsed"]["amgf_setup_ptap_sum"] if rec["parsed"]["amgf_setup_ptap_sum"] is not None else 0.0
        for rec in ordered
    ], dtype=float)
    setup_solver = np.array([
        rec["parsed"]["amgf_setup_solver_sum"] if rec["parsed"]["amgf_setup_solver_sum"] is not None else 0.0
        for rec in ordered
    ], dtype=float)
    setup_amg = np.array([
        rec["parsed"]["amgf_setup_amg_sum"] if rec["parsed"]["amgf_setup_amg_sum"] is not None else 0.0
        for rec in ordered
    ], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(max(10, 1.4 * len(ordered)), 10), sharex=True)

    bar_width = 0.7
    axes[0].bar(x, solve_amg, label="AMG", color=COLORS[0], width=bar_width, edgecolor='white', linewidth=0.5)
    axes[0].bar(x, solve_subspace, bottom=solve_amg, label="Subspace", color=COLORS[1], width=bar_width, edgecolor='white', linewidth=0.5)
    axes[0].bar(x, solve_other, bottom=solve_amg + solve_subspace, label="Other", color=COLORS[2], width=bar_width, edgecolor='white', linewidth=0.5)
    axes[0].set_title("AMGF Solve Timing Breakdown", fontweight='bold', pad=12)
    axes[0].set_ylabel("Accumulated Time (s)", fontweight='semibold')
    axes[0].grid(True, axis="y", linestyle='--', linewidth=0.6)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].legend(frameon=True, fancybox=False, edgecolor='gray', framealpha=0.95)

    axes[1].bar(x, setup_amg, label="AMG Setup", color=COLORS[3], width=bar_width, edgecolor='white', linewidth=0.5)
    axes[1].bar(x, setup_ptap, bottom=setup_amg, label="PtAP", color=COLORS[4], width=bar_width, edgecolor='white', linewidth=0.5)
    axes[1].bar(x, setup_solver, bottom=setup_amg + setup_ptap, label="Subspace Solver", color=COLORS[5], width=bar_width, edgecolor='white', linewidth=0.5)
    axes[1].set_title("AMGF Filtered-Subspace Setup Breakdown", fontweight='bold', pad=12)
    axes[1].set_ylabel("Accumulated Time (s)", fontweight='semibold')
    axes[1].grid(True, axis="y", linestyle='--', linewidth=0.6)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].legend(frameon=True, fancybox=False, edgecolor='gray', framealpha=0.95)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_xlabel("Run Configuration", fontweight='semibold')

    plt.tight_layout()
    out = output_dir / "timing_breakdown.png"
    plt.savefig(out)
    print(f"Saved {out}")
    plt.show()


# =============================================================================
# REPORTING
# =============================================================================

def print_summary_table(records):
    headers = [
        "run_id", "returncode", "curve_label", "dofs", "avg_subdomains", "pcg_total", "pcg_mean",
        "time_total", "time_mean", "n_iters", "n_times"
    ]
    rows = []

    for rec in records:
        parsed = rec["parsed"]
        dofs = parsed.get("global_dofs")
        subdomains_avg = parsed.get("schwarz_subdomains_avg")
        rows.append([
            rec["run_id"],
            rec["metadata"]["returncode"],
            config_to_label(rec["config"], CURVE_KEYS),
            format_dofs(dofs) if dofs else "N/A",
            f"{subdomains_avg:.1f}" if subdomains_avg is not None else "N/A",
            parsed["pcg_total"],
            parsed["pcg_mean"],
            parsed["time_total"],
            parsed["time_mean"],
            parsed["num_linear_solves_from_iters"],
            parsed["num_linear_solves_from_times"],
        ])

    widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(headers)]

    def fmt_row(r):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))

    print()
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run and plot parameter sweeps for test-contact."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not execute experiments, load existing run artifacts only.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Explicit output directory.",
    )
    args = parser.parse_args()

    output_dir = build_output_dir(BASE_CONFIG, SWEEP_PARAMETERS, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(output_dir / "experiment_definition.json", {
        "base_config": BASE_CONFIG,
        "sweep_parameters": SWEEP_PARAMETERS,
        "curve_keys": CURVE_KEYS,
        "x_axis_mode": X_AXIS_MODE,
        "executable": EXECUTABLE,
    })

    records = []

    if args.skip_run:
        print(f"Loading existing runs from {output_dir}")
        records = load_existing_runs(output_dir, BASE_CONFIG, SWEEP_PARAMETERS)
        print(f"Loaded {len(records)} runs matching SWEEP_PARAMETERS")
    else:
        if not Path(EXECUTABLE).exists():
            print(f"Error: executable not found: {EXECUTABLE}")
            sys.exit(1)

        configs = expand_sweep(BASE_CONFIG, SWEEP_PARAMETERS)
        print(f"Running {len(configs)} experiment(s)")
        print(f"Output directory: {output_dir}")

        for i, config in enumerate(configs):
            label = config_to_label(config, CURVE_KEYS)
            run_id = f"run_{i:03d}_{short_hash(config)}"
            cmd = config_to_command(config)

            print()
            print("=" * 80)
            print(f"Run {i + 1}/{len(configs)}")
            print(f"Run ID: {run_id}")
            print(f"Label: {label}")
            print("Command:")
            print(" ".join(cmd))

            result = run_command(cmd)
            parsed = parse_run_output(result["combined_output"])

            save_run_artifacts(output_dir, run_id, config, result, parsed)

            records.append({
                "run_id": run_id,
                "config": config,
                "parsed": parsed,
                "metadata": {"returncode": result["returncode"]},
            })

            print(f"Return code: {result['returncode']}")
            print(f"PCG solves parsed: {parsed['num_linear_solves_from_iters']}")
            print(f"Timing entries parsed: {parsed['num_linear_solves_from_times']}")
            print(f"Total PCG iterations: {parsed['pcg_total']}")
            print(f"Total linear solve time [s]: {parsed['time_total']}")

    if not records:
        print("No records found.")
        sys.exit(1)

    print_summary_table(records)

    plot_per_solve_curves(records, output_dir)
    plot_summary_curves(records, output_dir, X_AXIS_MODE)
    plot_scatter_time_vs_iterations(records, output_dir)
    plot_timing_breakdown(records, output_dir, X_AXIS_MODE)

    print("Done.")


if __name__ == "__main__":
    main()
