# AMG Benchmark Tool

This tool benchmarks different AMG preconditioners (BoomerAMG, AMGF, and Additive-Schwarz AMGF) on the full contact system `A + J^T D J`.

## Overview

The `amg-benchmark` executable:
- Loads matrices from disk (J, D, P, PTAP)
- Constructs the full system matrix
- Tests multiple AMG solver configurations
- Reports detailed timing and iteration information
- Outputs results in CSV format for analysis

## Building

```bash
cd mfem/miniapps/test-contact
make amg-benchmark
```

Requirements:
- MFEM built with MPI support
- For AMGF with direct solver: MUMPS, MKL CPardiso, SuperLU, or STRUMPACK
- For Schwarz-AMGF: HYPRE with Schwarz support

## Usage

### Basic Usage

```bash
# Run all three solvers on iteration 0
mpirun -np 4 ./amg-benchmark -mats-dir mats -iteration 0

# Run only BoomerAMG and AMGF
mpirun -np 4 ./amg-benchmark -mats-dir mats -iteration 0 -no-schwarz-amgf

# Save results to CSV file
mpirun -np 4 ./amg-benchmark -mats-dir mats -iteration 0 -output results.csv
```

### Controlling AMG Parameters

#### Interpolation Type
```bash
# Extended+i interpolation (default)
mpirun -np 4 ./amg-benchmark -iteration 0 -interp ext+i

# Standard interpolation
mpirun -np 4 ./amg-benchmark -iteration 0 -interp standard

# Extended interpolation
mpirun -np 4 ./amg-benchmark -iteration 0 -interp ext
```

#### Relaxation Type
```bash
# Symmetric Gauss-Seidel (default: 8)
mpirun -np 4 ./amg-benchmark -iteration 0 -relax-type 8

# L1-Jacobi
mpirun -np 4 ./amg-benchmark -iteration 0 -relax-type 18

# L1-hybrid symmetric Gauss-Seidel (HYPRE >= 2.30.0)
mpirun -np 4 ./amg-benchmark -iteration 0 -relax-type 88
```

#### Coarsening Type
```bash
# HMIS coarsening (default: 10)
mpirun -np 4 ./amg-benchmark -iteration 0 -coarsen-type 10

# PMIS coarsening
mpirun -np 4 ./amg-benchmark -iteration 0 -coarsen-type 8

# Falgout coarsening
mpirun -np 4 ./amg-benchmark -iteration 0 -coarsen-type 6
```

### Schwarz Solver Options

```bash
# Adjust Schwarz relaxation weight
mpirun -np 4 ./amg-benchmark -iteration 0 -schwarz-weight 0.5

# Use CG with Schwarz preconditioner (instead of direct)
mpirun -np 4 ./amg-benchmark -iteration 0 -schwarz-cg-iters 10

# Expand subdomains with graph neighbors
mpirun -np 4 ./amg-benchmark -iteration 0 -schwarz-expand

# Unweighted additive Schwarz
mpirun -np 4 ./amg-benchmark -iteration 0 -schwarz-unweighted

# Set uniform weight for all DOFs
mpirun -np 4 ./amg-benchmark -iteration 0 -schwarz-uniform-weight 0.1
```

### Running Multiple Iterations

```bash
# Benchmark all iterations and collect results
for i in {0..10}; do
    mpirun -np 4 ./amg-benchmark -iteration $i -output results.csv
done
```

## Output Format

The tool outputs results in CSV format with the following columns:

- `Iteration`: Iteration number
- `Solver`: Solver name (BoomerAMG, AMGF, Schwarz-AMGF)
- `SetupTime_s`: Preconditioner setup time in seconds
- `SolveTime_s`: PCG solve time in seconds
- `Iterations`: Number of PCG iterations
- `FinalResidual`: Final residual norm
- `Converged`: Whether PCG converged (True/False)

Example output:
```
Iteration,Solver,SetupTime_s,SolveTime_s,Iterations,FinalResidual,Converged
0,BoomerAMG,1.234567e+00,2.345678e+00,150,9.876543e-11,True
0,AMGF,1.456789e+00,1.234567e+00,80,9.123456e-11,True
0,Schwarz-AMGF,1.567890e+00,8.901234e-01,60,9.456789e-11,True
```

## Analysis

Use the provided Python script to analyze and visualize results:

```bash
# Generate comparison plots
python analyze_amg_benchmark.py --csv results.csv

# Save plots with custom prefix
python analyze_amg_benchmark.py --csv results.csv --output-prefix my_analysis_

# Use AMGF as baseline for speedup analysis
python analyze_amg_benchmark.py --csv results.csv --baseline AMGF
```

The analysis script generates:
1. **Comparison plots**: Setup time, solve time, iterations, and total time
2. **Speedup analysis**: Solve time speedup and iteration reduction vs baseline
3. **Summary statistics**: Mean and standard deviation for all metrics

## Example Workflow

```bash
# 1. Run test-contact to generate matrix files
mpirun -np 4 ./test-contact -prob 0 -sr 1 -pr 0 -nsteps 5

# 2. Benchmark different interpolation schemes
for interp in "standard" "ext" "ext+i"; do
    for iter in {0..4}; do
        mpirun -np 4 ./amg-benchmark \
            -iteration $iter \
            -interp $interp \
            -output results_${interp}.csv
    done
done

# 3. Analyze results
python analyze_amg_benchmark.py --csv results_standard.csv --output-prefix standard_
python analyze_amg_benchmark.py --csv results_ext.csv --output-prefix ext_
python analyze_amg_benchmark.py --csv results_ext+i.csv --output-prefix extplusi_

# 4. Compare different interpolation schemes
python compare_interpolations.py results_standard.csv results_ext.csv results_ext+i.csv
```

## Command-Line Options Reference

| Option | Default | Description |
|--------|---------|-------------|
| `-mats-dir` | `mats` | Directory containing matrix files |
| `-iteration` | `0` | Iteration number to load |
| `-interp` | `ext+i` | HYPRE interpolation type |
| `-relax-type` | `8` | HYPRE relaxation type |
| `-coarsen-type` | `10` | HYPRE coarsening type |
| `-boomeramg` / `-no-boomeramg` | `true` | Run BoomerAMG benchmark |
| `-amgf` / `-no-amgf` | `true` | Run AMGF benchmark |
| `-schwarz-amgf` / `-no-schwarz-amgf` | `true` | Run Schwarz-AMGF benchmark |
| `-schwarz-weight` | `1.0` | Schwarz relaxation weight |
| `-schwarz-variant` | `2` | HYPRE Schwarz variant |
| `-schwarz-cg-iters` | `0` | CG iterations (0=direct) |
| `-schwarz-expand` | `false` | Expand subdomains with neighbors |
| `-schwarz-unweighted` | `false` | Disable per-DOF scaling |
| `-schwarz-uniform-weight` | `-1` | Uniform weight (-1=disabled) |
| `-output` | `` | Output CSV file (empty=stdout) |

## Notes

1. **Matrix Files**: The tool expects matrix files in MFEM parallel format:
   - `J_matrix_iter_<N>.mat.<RANK>`
   - `D_matrix_iter_<N>.mat.<RANK>`
   - `P_matrix_iter_<N>.mat.<RANK>`
   - `PTAP_matrix_iter_<N>.mat.<RANK>`

2. **AMGF Requirements**: AMGF with direct subspace solver requires a parallel direct solver. The tool will select the first available solver (MUMPS, CPardiso, SuperLU, STRUMPACK).

3. **Schwarz Subdomains**: Schwarz-AMGF constructs subdomains from the constraint Jacobian J, where each constraint row defines a subdomain.

4. **Performance**: Setup time includes both AMG hierarchy construction and subspace solver factorization. Solve time is the PCG iteration time only.

## Troubleshooting

**Error: "Cannot open matrix file"**
- Ensure matrix files exist in the specified directory
- Check that files have the correct rank suffix

**Error: "No parallel direct solver available"**
- AMGF requires MUMPS, MKL CPardiso, SuperLU, or STRUMPACK
- Rebuild MFEM with one of these libraries enabled
- Use `-no-amgf` to skip AMGF benchmark

**Warning: High iteration counts**
- Try different interpolation schemes (`-interp`)
- Adjust relaxation type (`-relax-type`)
- Tune Schwarz parameters (`-schwarz-weight`, `-schwarz-uniform-weight`)

## References

- MFEM documentation: https://mfem.org
- HYPRE documentation: https://hypre.readthedocs.io
- test-contact miniapp: `test-contact.cpp`
