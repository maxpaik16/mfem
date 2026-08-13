// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

//                  AMG Solver Benchmark for Contact Systems
//
// Description:
// This miniapp benchmarks different AMG preconditioners (BoomerAMG, AMGF,
// and Additive-Schwarz AMGF) on the reduced contact system Areduced which
// includes J^T D J. It loads matrices from disk and solves the system
// using PCG with different preconditioners, reporting detailed timing and
// iteration information for post-processing analysis.
//
// The benchmark supports D-interpolation experiments where the diagonal D
// matrix is interpolated from its original values toward a uniform mean:
// D_interp = (1 - alpha) * D_orig + alpha * mean(D), where alpha ∈ [0, 1].
// This allows studying how AMG performance changes as D becomes more uniform.
//
// Compile with: make amg-benchmark
//
// Sample runs:
// mpirun -np 4 ./amg-benchmark -mats-dir mats -iteration 0
// mpirun -np 4 ./amg-benchmark -mats-dir mats -iteration 0 -num-steps 5
// mpirun -np 4 ./amg-benchmark -mats-dir mats -iteration 0 -num-steps 10 -output results.csv

#include "mfem.hpp"
#include "solver_utils.hpp"
#include "../common/schwarz_solver.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace mfem;

// Solver configuration
struct SolverConfig
{
   string name;
   bool use_amgf;
   bool use_schwarz;
   real_t schwarz_weight;
   bool schwarz_unweighted;
   real_t schwarz_uniform_weight;
};

// Timing and iteration results
struct SolverResults
{
   string solver_name;
   int interp_step;
   real_t alpha;
   double setup_time;
   double solve_time;
   int num_iterations;
   double final_residual;
   bool converged;
};

// Load MFEM matrix from file using built-in HypreParMatrix::Read
HypreParMatrix* LoadHypreMatrix(const string& filename, MPI_Comm comm)
{
   int myid;
   MPI_Comm_rank(comm, &myid);

   if (myid == 0)
   {
      cout << "Loading matrix from: " << filename << endl;
   }

   HypreParMatrix* mat = new HypreParMatrix();

   try
   {
      // Try Read_IJMatrix instead of Read
      mat->Read_IJMatrix(comm, filename);
   }
   catch (const exception& e)
   {
      if (myid == 0)
      {
         cerr << "Error reading matrix: " << e.what() << endl;
      }
      throw;
   }

   if (myid == 0)
   {
      cout << "Matrix loaded: " << mat->M() << " x " << mat->N() << endl;
   }

   return mat;
}

// Construct J^T D J
HypreParMatrix* ComputeJtDJ(HypreParMatrix* J, HypreParMatrix* D)
{
   // Compute D*J
   HypreParMatrix* DJ = ParMult(D, J);

   // Compute J^T * (D*J)
   HypreParMatrix* JT = J->Transpose();
   HypreParMatrix* JtDJ = ParMult(JT, DJ);

   delete DJ;
   delete JT;
   return JtDJ;
}

// Create interpolated D matrix: D_new = (1 - alpha) * D_orig + alpha * mean(D)
HypreParMatrix* InterpolateD(HypreParMatrix* D_orig, real_t alpha)
{
   // Extract diagonal values from D_orig
   hypre_ParCSRMatrix* parcsr_D = *D_orig;
   hypre_CSRMatrix* diag_D = hypre_ParCSRMatrixDiag(parcsr_D);
   HYPRE_Real* diag_data = hypre_CSRMatrixData(diag_D);
   HYPRE_Int D_size = hypre_CSRMatrixNumRows(diag_D);

   // Compute local sum for mean calculation
   real_t local_sum = 0.0;
   for (int i = 0; i < D_size; i++)
   {
      local_sum += diag_data[i];
   }

   // Compute global mean
   real_t global_sum = 0.0;
   HYPRE_Int global_size = 0;
   MPI_Allreduce(&local_sum, &global_sum, 1, HYPRE_MPI_REAL, MPI_SUM, D_orig->GetComm());
   MPI_Allreduce(&D_size, &global_size, 1, HYPRE_MPI_INT, MPI_SUM, D_orig->GetComm());
   real_t mean_val = global_sum / global_size;

   // Create new diagonal matrix by modifying hypre data directly
   // Clone the original D matrix structure
   hypre_ParCSRMatrix* D_new_hypre = hypre_ParCSRMatrixClone(*D_orig, 1);

   // Get diagonal block and modify its data
   hypre_CSRMatrix* diag_new = hypre_ParCSRMatrixDiag(D_new_hypre);
   HYPRE_Real* diag_new_data = hypre_CSRMatrixData(diag_new);

   // Update diagonal values: (1 - alpha) * D_orig + alpha * mean
   for (int i = 0; i < D_size; i++)
   {
      diag_new_data[i] = (1.0 - alpha) * diag_data[i] + alpha * mean_val;
   }

   // Wrap in HypreParMatrix
   HypreParMatrix* D_new = new HypreParMatrix(D_new_hypre);

   return D_new;
}

// Build Schwarz subdomains from J matrix mapped to contact subspace via P
// EXACT COPY from ip.cpp IPSolver::BuildSchwarzSubdomains
void BuildSchwarzSubdomainsFromJ(HypreParMatrix* J, HypreSchwarz* schwarz,
                                  HypreParMatrix* PTAP,  // Matrix in contact subspace
                                  HypreParMatrix* P,      // Transfer operator (full DOFs -> contact subspace)
                                  bool schwarz_expand, real_t schwarz_min_diag_value,
                                  real_t schwarz_relax_weight, bool schwarz_unweighted,
                                  real_t schwarz_uniform_weight,
                                  HypreParMatrix* D)
{
   int myid;
   MPI_Comm_rank(J->GetComm(), &myid);
   MPI_Comm comm = J->GetComm();

   // Extract D diagonal values
   Vector D_diag;
   hypre_ParCSRMatrix* parcsr_D = *D;
   hypre_CSRMatrix* diag_D = hypre_ParCSRMatrixDiag(parcsr_D);
   HYPRE_Real* diag_data = hypre_CSRMatrixData(diag_D);
   HYPRE_Int D_size = hypre_CSRMatrixNumRows(diag_D);
   D_diag.SetSize(D_size);
   for (int i = 0; i < D_size; i++)
   {
      D_diag(i) = diag_data[i];
   }

   // Merge J matrix to access local rows
   SparseMatrix merged_J;
   J->MergeDiagAndOffd(merged_J);

   // Access PTAP diagonal for graph queries (if expanding subdomains)
   hypre_ParCSRMatrix* parPTAP = (hypre_ParCSRMatrix*)(*PTAP);
   hypre_CSRMatrix* PTAP_diag = hypre_ParCSRMatrixDiag(parPTAP);
   HYPRE_Int* PTAP_Ai = hypre_CSRMatrixI(PTAP_diag);
   HYPRE_Int* PTAP_Aj = hypre_CSRMatrixJ(PTAP_diag);
   HYPRE_Int num_dofs_subspace = hypre_CSRMatrixNumRows(PTAP_diag);

   int nranks;
   MPI_Comm_size(comm, &nranks);

   const HYPRE_BigInt local_ptap_row_start = PTAP->GetRowStarts()[0];
   const HYPRE_BigInt local_ptap_row_end = PTAP->GetRowStarts()[1];
   std::vector<HYPRE_BigInt> all_ptap_row_starts(nranks);
   std::vector<HYPRE_BigInt> all_ptap_row_ends(nranks);
   MPI_Allgather(&local_ptap_row_start, 1, HYPRE_MPI_BIG_INT,
                 all_ptap_row_starts.data(), 1, HYPRE_MPI_BIG_INT, comm);
   MPI_Allgather(&local_ptap_row_end, 1, HYPRE_MPI_BIG_INT,
                 all_ptap_row_ends.data(), 1, HYPRE_MPI_BIG_INT, comm);

   struct DofMapping
   {
      HYPRE_BigInt global_dof;
      HYPRE_BigInt subspace_dof;
   };

   std::vector<DofMapping> local_mappings;

   // Get P's row/column partitions to convert local indices to global indices.
   const HYPRE_BigInt *p_row_starts = P->GetRowStarts();
   const HYPRE_BigInt p_local_row_start = p_row_starts[0];
   const HYPRE_BigInt p_local_col_start = P->GetColStarts()[0];
   const HYPRE_Int *P_diag_I = P->GetDiagMemoryI();
   const HYPRE_Int *P_diag_J = P->GetDiagMemoryJ();
   const HYPRE_Int *P_offd_I = P->GetOffdMemoryI();
   const HYPRE_Int *P_offd_J = P->GetOffdMemoryJ();
   HYPRE_BigInt *P_offd_cmap = nullptr;
   HYPRE_Int P_num_offd_cols = 0;
   P->GetOffdColMap(P_offd_cmap, P_num_offd_cols);

   for (int i = 0; i < P->GetNumRows(); i++)
   {
      const HYPRE_BigInt global_dof = p_local_row_start + i;
      for (int j = P_diag_I[i]; j < P_diag_I[i + 1]; ++j)
      {
         local_mappings.push_back({global_dof, p_local_col_start + P_diag_J[j]});
      }
      for (int j = P_offd_I[i]; j < P_offd_I[i + 1]; ++j)
      {
         local_mappings.push_back({global_dof, P_offd_cmap[P_offd_J[j]]});
      }
   }

   int local_count = local_mappings.size();
   std::vector<int> counts(nranks), displs(nranks + 1, 0);

   MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);

   for (int r = 0; r < nranks; r++)
   {
      displs[r + 1] = displs[r] + counts[r];
   }

   int total_mappings = displs[nranks];
   std::vector<DofMapping> all_mappings(total_mappings);

   std::vector<int> byte_counts(nranks), byte_displs(nranks);
   for (int r = 0; r < nranks; r++)
   {
      byte_counts[r] = counts[r] * sizeof(DofMapping);
      byte_displs[r] = displs[r] * sizeof(DofMapping);
   }

   MPI_Allgatherv(local_mappings.data(), local_count * sizeof(DofMapping), MPI_BYTE,
                  all_mappings.data(), byte_counts.data(), byte_displs.data(),
                  MPI_BYTE, comm);

   std::map<HYPRE_BigInt, HYPRE_BigInt> global_to_subspace;
   for (const auto &mapping : all_mappings)
   {
      global_to_subspace[mapping.global_dof] = mapping.subspace_dof;
   }

   std::vector<std::vector<HYPRE_BigInt>> local_subdomains;
   {
      int local_J_rows = J->Height();
      for (int i = 0; i < local_J_rows; i++)
      {
         const int* cols = merged_J.GetRowColumns(i);
         int row_size = merged_J.RowSize(i);

         if (row_size == 0) { continue; }

         // Skip this constraint if D diagonal value is below threshold.
         if (schwarz_min_diag_value > 0.0 && i < D_diag.Size())
         {
            if (D_diag(i) < schwarz_min_diag_value)
            {
               continue;
            }
         }

         std::set<HYPRE_BigInt> subdomain_set;

         // Add all DoFs in this contact constraint.
         for (int j = 0; j < row_size; j++)
         {
            const HYPRE_BigInt global_dof = cols[j];
            auto it = global_to_subspace.find(global_dof);
            if (it != global_to_subspace.end())
            {
               subdomain_set.insert(it->second);
            }
         }

         if (schwarz_expand)
         {
            std::vector<HYPRE_BigInt> original_dofs(subdomain_set.begin(), subdomain_set.end());
            for (HYPRE_BigInt dof : original_dofs)
            {
               if (dof >= local_ptap_row_start &&
                   dof < local_ptap_row_start + num_dofs_subspace)
               {
                  const HYPRE_Int local_dof = static_cast<HYPRE_Int>(dof - local_ptap_row_start);
                  for (HYPRE_Int jj = PTAP_Ai[local_dof]; jj < PTAP_Ai[local_dof + 1]; jj++)
                  {
                     const HYPRE_BigInt neighbor = local_ptap_row_start + PTAP_Aj[jj];
                     if (neighbor != dof)
                     {
                        subdomain_set.insert(neighbor);
                     }
                  }
               }
            }
         }

         if (!subdomain_set.empty())
         {
            local_subdomains.emplace_back(subdomain_set.begin(), subdomain_set.end());
         }
      }
   } // end row-based subdomain selection

   // J rows may be unevenly distributed across ranks, so rebalance the
   // generated subdomains explicitly before building the Schwarz factors.
   const int local_num_subdomains = static_cast<int>(local_subdomains.size());
   std::vector<int> subdomain_counts(nranks, 0);
   MPI_Allgather(&local_num_subdomains, 1, MPI_INT,
                 subdomain_counts.data(), 1, MPI_INT, comm);

   std::vector<int> subdomain_displs(nranks, 0);
   int total_subdomains = 0;
   for (int r = 0; r < nranks; ++r)
   {
      subdomain_displs[r] = total_subdomains;
      total_subdomains += subdomain_counts[r];
   }

   std::vector<int> local_subdomain_sizes(local_num_subdomains, 0);
   int local_total_memberships = 0;
   for (int i = 0; i < local_num_subdomains; ++i)
   {
      local_subdomain_sizes[i] = static_cast<int>(local_subdomains[i].size());
      local_total_memberships += local_subdomain_sizes[i];
   }

   std::vector<int> all_subdomain_sizes(total_subdomains, 0);
   MPI_Allgatherv(local_subdomain_sizes.empty() ? nullptr : local_subdomain_sizes.data(),
                  local_num_subdomains, MPI_INT,
                  all_subdomain_sizes.empty() ? nullptr : all_subdomain_sizes.data(),
                  subdomain_counts.data(), subdomain_displs.data(), MPI_INT, comm);

   std::vector<int> membership_counts(nranks, 0);
   MPI_Allgather(&local_total_memberships, 1, MPI_INT,
                 membership_counts.data(), 1, MPI_INT, comm);

   std::vector<int> membership_displs(nranks, 0);
   int total_memberships = 0;
   for (int r = 0; r < nranks; ++r)
   {
      membership_displs[r] = total_memberships;
      total_memberships += membership_counts[r];
   }

   std::vector<HYPRE_BigInt> local_flat_subdomains;
   local_flat_subdomains.reserve(local_total_memberships);
   for (const auto &subdomain : local_subdomains)
   {
      local_flat_subdomains.insert(local_flat_subdomains.end(),
                                   subdomain.begin(), subdomain.end());
   }

   std::vector<HYPRE_BigInt> all_flat_subdomains(total_memberships);
   MPI_Allgatherv(local_flat_subdomains.empty() ? nullptr : local_flat_subdomains.data(),
                  local_total_memberships, HYPRE_MPI_BIG_INT,
                  all_flat_subdomains.empty() ? nullptr : all_flat_subdomains.data(),
                  membership_counts.data(), membership_displs.data(),
                  HYPRE_MPI_BIG_INT, comm);

   std::vector<int> subdomain_value_offsets(total_subdomains + 1, 0);
   for (int i = 0; i < total_subdomains; ++i)
   {
      subdomain_value_offsets[i + 1] = subdomain_value_offsets[i] + all_subdomain_sizes[i];
   }

   std::vector<int> assigned_subdomain_counts(nranks, 0);
   std::vector<int> assigned_subdomain_sizes(nranks, 0);
   std::vector<int> subdomain_owner(total_subdomains, -1);
   const int target_count_low = total_subdomains / nranks;
   const int target_count_high = (total_subdomains + nranks - 1) / nranks;
   const int target_size_low = total_memberships / nranks;
   const int target_size_high = (total_memberships + nranks - 1) / nranks;

   auto owner_rank = [&](HYPRE_BigInt gdof) -> int
   {
      for (int r = 0; r < nranks; ++r)
      {
         if (gdof >= all_ptap_row_starts[r] && gdof < all_ptap_row_ends[r])
         {
            return r;
         }
      }
      return -1;
   };

   std::vector<int> assignment_order(total_subdomains);
   for (int i = 0; i < total_subdomains; ++i)
   {
      assignment_order[i] = i;
   }
   std::sort(assignment_order.begin(), assignment_order.end(),
             [&](int lhs, int rhs)
   {
      if (all_subdomain_sizes[lhs] != all_subdomain_sizes[rhs])
      {
         return all_subdomain_sizes[lhs] > all_subdomain_sizes[rhs];
      }
      return lhs < rhs;
   });

   for (int idx = 0; idx < total_subdomains; ++idx)
   {
      const int i = assignment_order[idx];
      const int subdomain_size = all_subdomain_sizes[i];
      std::vector<int> owned_dofs_per_rank(nranks, 0);
      for (int j = subdomain_value_offsets[i]; j < subdomain_value_offsets[i + 1]; ++j)
      {
         const int rank = owner_rank(all_flat_subdomains[j]);
         MFEM_VERIFY(rank >= 0, "Failed to determine owner of Schwarz subdomain DOF.");
         owned_dofs_per_rank[rank]++;
      }

      int best_rank = 0;
      for (int r = 1; r < nranks; ++r)
      {
         const int new_count_r = assigned_subdomain_counts[r] + 1;
         const int new_count_best = assigned_subdomain_counts[best_rank] + 1;
         const int new_size_r = assigned_subdomain_sizes[r] + subdomain_size;
         const int new_size_best = assigned_subdomain_sizes[best_rank] + subdomain_size;
         const int remote_dofs_r = subdomain_size - owned_dofs_per_rank[r];
         const int remote_dofs_best = subdomain_size - owned_dofs_per_rank[best_rank];

         const bool r_exceeds_count = new_count_r > target_count_high;
         const bool best_exceeds_count = new_count_best > target_count_high;
         const bool r_exceeds_size = new_size_r > target_size_high;
         const bool best_exceeds_size = new_size_best > target_size_high;

         const int count_gap_r = std::abs(new_count_r - target_count_low);
         const int count_gap_best = std::abs(new_count_best - target_count_low);
         const int size_gap_r = std::abs(new_size_r - target_size_low);
         const int size_gap_best = std::abs(new_size_best - target_size_low);

         if ((!r_exceeds_count && best_exceeds_count) ||
             ((!r_exceeds_count == !best_exceeds_count) &&
              (!r_exceeds_size && best_exceeds_size)) ||
             ((!r_exceeds_count == !best_exceeds_count) &&
              (!r_exceeds_size == !best_exceeds_size) &&
              count_gap_r < count_gap_best) ||
             ((!r_exceeds_count == !best_exceeds_count) &&
              (!r_exceeds_size == !best_exceeds_size) &&
              count_gap_r == count_gap_best &&
              size_gap_r < size_gap_best) ||
             ((!r_exceeds_count == !best_exceeds_count) &&
              (!r_exceeds_size == !best_exceeds_size) &&
              count_gap_r == count_gap_best &&
              size_gap_r == size_gap_best &&
              remote_dofs_r < remote_dofs_best) ||
             ((!r_exceeds_count == !best_exceeds_count) &&
              (!r_exceeds_size == !best_exceeds_size) &&
              count_gap_r == count_gap_best &&
              size_gap_r == size_gap_best &&
              remote_dofs_r == remote_dofs_best &&
              owned_dofs_per_rank[r] > owned_dofs_per_rank[best_rank]) ||
             ((!r_exceeds_count == !best_exceeds_count) &&
              (!r_exceeds_size == !best_exceeds_size) &&
              count_gap_r == count_gap_best &&
              size_gap_r == size_gap_best &&
              remote_dofs_r == remote_dofs_best &&
              owned_dofs_per_rank[r] == owned_dofs_per_rank[best_rank] &&
              r < best_rank))
         {
            best_rank = r;
         }
      }

      subdomain_owner[i] = best_rank;
      assigned_subdomain_counts[best_rank]++;
      assigned_subdomain_sizes[best_rank] += subdomain_size;
   }

   std::vector<std::vector<HYPRE_BigInt>> subdomains;
   subdomains.reserve(assigned_subdomain_counts[myid]);
   for (int i = 0; i < total_subdomains; ++i)
   {
      if (subdomain_owner[i] != myid)
      {
         continue;
      }

      const int begin = subdomain_value_offsets[i];
      const int end = subdomain_value_offsets[i + 1];
      subdomains.emplace_back(all_flat_subdomains.begin() + begin,
                              all_flat_subdomains.begin() + end);
   }

   // Print diagnostic information about subdomains
   if (myid == 0)
   {
      // Compute subdomain size statistics on the globally balanced set.
      int num_subdomains = total_subdomains;
      int total_size = 0;
      int min_size = num_subdomains > 0 ? all_subdomain_sizes[0] : 0;
      int max_size = 0;

      // Track covered DOFs
      std::set<HYPRE_BigInt> covered_dofs;
      for (int i = 0; i < num_subdomains; ++i)
      {
         int size = all_subdomain_sizes[i];
         total_size += size;
         if (size < min_size) min_size = size;
         if (size > max_size) max_size = size;

         for (int j = subdomain_value_offsets[i]; j < subdomain_value_offsets[i + 1]; ++j)
         {
            covered_dofs.insert(all_flat_subdomains[j]);
         }
      }

      double avg_size = num_subdomains > 0 ? (double)total_size / num_subdomains : 0.0;
      int num_covered = covered_dofs.size();
      int num_uncovered = PTAP->N() - num_covered;
      double overlap_ratio = num_covered > 0 ? (double)(total_size - num_covered) / num_covered : 0.0;

      cout << "\nSchwarz subdomain statistics:" << endl;
      cout << "  Total number of subdomains: " << num_subdomains << endl;
      cout << "  Subdomain size range: min = " << min_size
           << ", max = " << max_size << endl;
      cout << "  Total aggregate subdomain size: " << total_size << endl;
      cout << "  Subdomain partition across ranks: ";
      for (int r = 0; r < nranks; ++r)
      {
         cout << assigned_subdomain_counts[r];
         if (r + 1 < nranks)
         {
            cout << " ";
         }
      }
      cout << endl;
      cout << "  Total subspace size: " << PTAP->N() << endl;
      if (schwarz_min_diag_value > 0.0)
      {
         int total_constraints = J->Height();
         int num_filtered = total_constraints - num_subdomains;
         cout << "  Total constraints: " << total_constraints << endl;
         cout << "  Filtered (D < " << schwarz_min_diag_value << "): " << num_filtered << endl;
      }
      if (num_subdomains > 0)
      {
         cout << "  Average subdomain size: " << avg_size << endl;
         cout << "  DOF coverage: " << num_covered << "/" << PTAP->N()
              << " (" << (100.0 * num_covered / PTAP->N()) << "%)" << endl;
         cout << "  Uncovered DOFs: " << num_uncovered << endl;
         cout << "  Overlap ratio: " << overlap_ratio << endl;

         if (num_uncovered > 0)
         {
            cout << "  WARNING: " << num_uncovered << " DOFs are not covered by any subdomain!" << endl;
            cout << "           This may affect Schwarz preconditioner effectiveness." << endl;
         }
      }
      cout << endl;
   }

   // Initialize HYPRE Schwarz if not already done
   if (!schwarz->schwarz_solver)
   {
      HYPRE_SchwarzCreate(&schwarz->schwarz_solver);
      HYPRE_SchwarzSetVariant(schwarz->schwarz_solver, HypreSchwarz::RequiredVariant);
   }

   // Configure Schwarz with custom subdomains
   // use_nonsymm = 0 → Cholesky factorization (symmetric, appropriate for contact)
   HYPRE_Int use_nonsymm = 0;
   schwarz->SetCustomSubdomains(subdomains, *PTAP, schwarz_relax_weight, use_nonsymm, schwarz_unweighted, schwarz_uniform_weight);
}

// Run solver benchmark
SolverResults BenchmarkSolver(HypreParMatrix* A, const Vector& b, const Vector& x0,
                               const SolverConfig& config, HypreParMatrix* J,
                               HypreParMatrix* D, const Operator* P,
                               const Operator* PTAP)
{
   int myid;
   MPI_Comm_rank(A->GetComm(), &myid);

   SolverResults results;
   results.solver_name = config.name;

   // Create preconditioner
   Solver* prec = nullptr;
   Solver* subspace_solver = nullptr;
   HypreSchwarz* schwarz_ptr = nullptr;

   double setup_start = MPI_Wtime();

   // Fixed AMG parameters
   const int relax_type = 8;  // Symmetric Gauss-Seidel
   const int coarsen_type = 10;  // HMIS coarsening

   if (config.use_amgf)
   {
      AMGFSolver* amgf = new AMGFSolver();

      // Configure AMG component
      amgf->GetAMG().SetSystemsOptions(3);
      amgf->GetAMG().SetPrintLevel(0);
      amgf->GetAMG().SetRelaxType(relax_type);
      amgf->GetAMG().SetCoarsening(coarsen_type);

      amgf->SetPrintLevel(0);

      // Configure subspace solver
      if (config.use_schwarz)
      {
         schwarz_ptr = new HypreSchwarz();
         schwarz_ptr->SetPrintLevel(0);

         // Initialize HYPRE Schwarz solver
         HYPRE_SchwarzCreate(&(schwarz_ptr->schwarz_solver));
         HYPRE_SchwarzSetVariant(schwarz_ptr->schwarz_solver, HypreSchwarz::RequiredVariant);

         // Build Schwarz subdomains BEFORE setting up AMGF
         // Use the PTAP matrix passed from the outer scope (already computed/loaded)
         // BuildSchwarzSubdomainsFromJ will call SetCustomSubdomains on the Schwarz solver
         BuildSchwarzSubdomainsFromJ(J, schwarz_ptr, (HypreParMatrix*)PTAP, (HypreParMatrix*)P,
                                     false, 0.0,
                                     config.schwarz_weight, config.schwarz_unweighted,
                                     config.schwarz_uniform_weight, D);

         // Always use direct Schwarz (no CG)
         subspace_solver = schwarz_ptr;
      }
      else
      {
         auto* direct_solver = new ParallelDirectSolver(A->GetComm(), "auto");
         direct_solver->SetPrintLevel(0);
         subspace_solver = direct_solver;
      }

      amgf->SetFilteredSubspaceSolver(*subspace_solver);
      amgf->SetFilteredSubspaceTransferOperator(*(HypreParMatrix*)P);

      prec = amgf;
   }
   else
   {
      HypreBoomerAMG* amg = new HypreBoomerAMG();
      amg->SetSystemsOptions(3);
      amg->SetPrintLevel(0);
      amg->SetRelaxType(relax_type);
      amg->SetCoarsening(coarsen_type);

      prec = amg;
   }

   prec->SetOperator(*A);

   double setup_end = MPI_Wtime();
   results.setup_time = setup_end - setup_start;

   // Create PCG solver
   CGSolver pcg(A->GetComm());
   pcg.SetRelTol(1e-10);
   pcg.SetMaxIter(1000);
   pcg.SetPrintLevel(0);
   pcg.SetOperator(*A);
   pcg.SetPreconditioner(*prec);

   // Solve
   Vector x(x0);
   double solve_start = MPI_Wtime();
   pcg.Mult(b, x);
   double solve_end = MPI_Wtime();

   results.solve_time = solve_end - solve_start;
   results.num_iterations = pcg.GetNumIterations();
   results.final_residual = pcg.GetFinalNorm();
   results.converged = pcg.GetConverged();

   // Cleanup
   delete prec;
   if (subspace_solver) { delete subspace_solver; }

   return results;
}

// Print results in CSV format for easy parsing
void PrintResultsCSV(const vector<SolverResults>& results, int iteration, int myid)
{
   if (myid != 0) return;

   cout << "\n=== BENCHMARK RESULTS (CSV Format) ===" << endl;
   cout << "Iteration,InterpStep,Alpha,Solver,SetupTime_s,SolveTime_s,Iterations,FinalResidual,Converged" << endl;

   for (const auto& r : results)
   {
      cout << iteration << ","
           << r.interp_step << ","
           << scientific << setprecision(6) << r.alpha << ","
           << r.solver_name << ","
           << scientific << setprecision(6) << r.setup_time << ","
           << scientific << setprecision(6) << r.solve_time << ","
           << r.num_iterations << ","
           << scientific << setprecision(6) << r.final_residual << ","
           << (r.converged ? "True" : "False") << endl;
   }
   cout << endl;
}

int main(int argc, char* argv[])
{
   // Initialize MPI
   Mpi::Init();
   int myid = Mpi::WorldRank();
   int num_procs = Mpi::WorldSize();
   Hypre::Init();

   // Parse command-line options
   const char* mats_dir = "mats";
   int iteration = 0;
   bool run_boomeramg = true;
   bool run_amgf = true;
   bool run_schwarz_amgf = true;
   real_t schwarz_weight = 1.0;
   bool schwarz_unweighted = false;
   real_t schwarz_uniform_weight = 0.1;
   const char* output_file = "";
   int num_interp_steps = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mats_dir, "-mats-dir", "--matrices-directory",
                  "Directory containing matrix files.");
   args.AddOption(&iteration, "-iteration", "--iteration",
                  "Iteration number to load.");
   args.AddOption(&run_boomeramg, "-boomeramg", "--run-boomeramg",
                  "-no-boomeramg", "--no-run-boomeramg",
                  "Run BoomerAMG benchmark.");
   args.AddOption(&run_amgf, "-amgf", "--run-amgf",
                  "-no-amgf", "--no-run-amgf",
                  "Run AMGF benchmark.");
   args.AddOption(&run_schwarz_amgf, "-schwarz-amgf", "--run-schwarz-amgf",
                  "-no-schwarz-amgf", "--no-run-schwarz-amgf",
                  "Run Schwarz-AMGF benchmark.");
   args.AddOption(&schwarz_weight, "-schwarz-weight", "--schwarz-weight",
                  "Schwarz relaxation weight.");
   args.AddOption(&schwarz_unweighted, "-schwarz-unweighted", "--schwarz-unweighted",
                  "-no-schwarz-unweighted", "--no-schwarz-unweighted",
                  "Disable per-DOF scaling in Schwarz preconditioner.");
   args.AddOption(&schwarz_uniform_weight, "-schwarz-uniform-weight",
                  "--schwarz-uniform-weight",
                  "Set uniform weight for Schwarz (default 0.1, -1 to disable).");
   args.AddOption(&output_file, "-output", "--output-file",
                  "Output CSV file (empty for stdout).");
   args.AddOption(&num_interp_steps, "-num-steps", "--num-interpolation-steps",
                  "Number of interpolation steps from original to uniform D (1=no interpolation).");

   args.Parse();
   if (!args.Good())
   {
      if (myid == 0)
      {
         args.PrintUsage(cout);
      }
      return 1;
   }
   if (myid == 0)
   {
      args.PrintOptions(cout);
   }

   // Load matrices
   if (myid == 0)
   {
      cout << "\n=== Loading matrices from iteration " << iteration << " ===" << endl;
   }

   stringstream ss_j, ss_d, ss_p, ss_ptap, ss_areduced;
   ss_j << mats_dir << "/J_matrix_iter_" << iteration << ".mat";
   ss_d << mats_dir << "/D_matrix_iter_" << iteration << ".mat";
   ss_p << mats_dir << "/P_matrix_iter_" << iteration << ".mat";
   ss_ptap << mats_dir << "/PTAP_matrix_iter_" << iteration << ".mat";
   ss_areduced << mats_dir << "/Areduced_matrix_iter_" << iteration << ".mat";

   HypreParMatrix* J = LoadHypreMatrix(ss_j.str(), MPI_COMM_WORLD);
   HypreParMatrix* D = LoadHypreMatrix(ss_d.str(), MPI_COMM_WORLD);
   HypreParMatrix* P = LoadHypreMatrix(ss_p.str(), MPI_COMM_WORLD);
   HypreParMatrix* PTAP_orig = LoadHypreMatrix(ss_ptap.str(), MPI_COMM_WORLD);
   HypreParMatrix* Areduced = LoadHypreMatrix(ss_areduced.str(), MPI_COMM_WORLD);

   if (myid == 0)
   {
      cout << "J:        " << J->M() << " x " << J->N() << endl;
      cout << "D:        " << D->M() << " x " << D->N() << endl;
      cout << "P:        " << P->M() << " x " << P->N() << endl;
      cout << "PTAP:     " << PTAP_orig->M() << " x " << PTAP_orig->N() << endl;
      cout << "Areduced: " << Areduced->M() << " x " << Areduced->N() << endl;
   }

   // Compute original J^T D J for reference
   if (myid == 0)
   {
      cout << "\n=== Computing original J^T D J ===" << endl;
   }
   HypreParMatrix* JtDJ_orig = ComputeJtDJ(J, D);

   // Configure solvers
   vector<SolverConfig> configs;

   if (run_boomeramg)
   {
      SolverConfig amg;
      amg.name = "BoomerAMG";
      amg.use_amgf = false;
      amg.use_schwarz = false;
      configs.push_back(amg);
   }

   if (run_amgf)
   {
      SolverConfig amgf;
      amgf.name = "AMGF";
      amgf.use_amgf = true;
      amgf.use_schwarz = false;
      configs.push_back(amgf);
   }

   if (run_schwarz_amgf)
   {
      SolverConfig schwarz_amgf;
      schwarz_amgf.name = "Schwarz-AMGF";
      schwarz_amgf.use_amgf = true;
      schwarz_amgf.use_schwarz = true;
      schwarz_amgf.schwarz_weight = schwarz_weight;
      schwarz_amgf.schwarz_unweighted = schwarz_unweighted;
      schwarz_amgf.schwarz_uniform_weight = schwarz_uniform_weight;
      configs.push_back(schwarz_amgf);
   }

   // Run benchmarks for each interpolation step
   vector<SolverResults> results;

   for (int step = 0; step < num_interp_steps; step++)
   {
      // Compute interpolation parameter
      real_t alpha = (num_interp_steps > 1) ?
                     (real_t)step / (num_interp_steps - 1) : 0.0;

      if (myid == 0)
      {
         cout << "\n=== Interpolation step " << step << "/" << num_interp_steps
              << " (alpha=" << alpha << ") ===" << endl;
      }

      // Create interpolated D
      HypreParMatrix* D_interp = InterpolateD(D, alpha);

      // Compute J^T D_interp J
      HypreParMatrix* JtDJ_interp = ComputeJtDJ(J, D_interp);

      // Compute modified system: A_modified = Areduced + (JtDJ_interp - JtDJ_orig)
      // and PTAP_modified = PTAP_orig + P^T * (JtDJ_interp - JtDJ_orig) * P
      HypreParMatrix* A = nullptr;
      HypreParMatrix* PTAP = nullptr;
      HypreParMatrix* JtDJ_diff = nullptr;
      HypreParMatrix* PtJtDJ_diff_P = nullptr;

      if (alpha == 0.0)
      {
         // For alpha=0, just use originals directly (no modification)
         A = Areduced;
         PTAP = PTAP_orig;
      }
      else
      {
         // JtDJ_diff = JtDJ_interp - JtDJ_orig
         JtDJ_diff = Add(1.0, *JtDJ_interp, -1.0, *JtDJ_orig);
         // A = Areduced + JtDJ_diff
         A = Add(1.0, *Areduced, 1.0, *JtDJ_diff);
         // PTAP = PTAP_orig + P^T * JtDJ_diff * P
         PtJtDJ_diff_P = RAP(JtDJ_diff, P);
         PTAP = Add(1.0, *PTAP_orig, 1.0, *PtJtDJ_diff_P);
      }

      // Create RHS vector (all ones for benchmark)
      Vector b(A->Height());
      b = 1.0;

      // Create initial guess
      Vector x0(A->Height());
      x0 = 0.0;

      if (myid == 0)
      {
         cout << "Running solver benchmarks..." << endl;
      }

      // Run each solver configuration
      for (const auto& config : configs)
      {
         if (myid == 0)
         {
            cout << "\n--- Running " << config.name << " ---" << endl;
         }

         SolverResults result = BenchmarkSolver(A, b, x0, config, J, D_interp, P, PTAP);
         result.interp_step = step;
         result.alpha = alpha;
         results.push_back(result);

         if (myid == 0)
         {
            cout << "Setup time:       " << scientific << setprecision(4)
                 << result.setup_time << " s" << endl;
            cout << "Solve time:       " << scientific << setprecision(4)
                 << result.solve_time << " s" << endl;
            cout << "Iterations:       " << result.num_iterations << endl;
            cout << "Final residual:   " << scientific << setprecision(4)
                 << result.final_residual << endl;
            cout << "Converged:        " << (result.converged ? "Yes" : "No") << endl;
         }
      }

      // Cleanup for this step
      delete D_interp;
      delete JtDJ_interp;
      if (JtDJ_diff) delete JtDJ_diff;
      if (PtJtDJ_diff_P) delete PtJtDJ_diff_P;
      if (alpha != 0.0 && A != nullptr) delete A;
      if (alpha != 0.0 && PTAP != nullptr) delete PTAP;
   }

   // Print CSV results
   PrintResultsCSV(results, iteration, myid);

   // Write to file if requested
   if (strlen(output_file) > 0 && myid == 0)
   {
      ofstream out(output_file, ios::app);
      if (!out.is_open())
      {
         // Create new file with header
         out.open(output_file);
         out << "Iteration,InterpStep,Alpha,Solver,SetupTime_s,SolveTime_s,Iterations,FinalResidual,Converged" << endl;
      }

      for (const auto& r : results)
      {
         out << iteration << ","
             << r.interp_step << ","
             << scientific << setprecision(6) << r.alpha << ","
             << r.solver_name << ","
             << scientific << setprecision(6) << r.setup_time << ","
             << scientific << setprecision(6) << r.solve_time << ","
             << r.num_iterations << ","
             << scientific << setprecision(6) << r.final_residual << ","
             << (r.converged ? "True" : "False") << endl;
      }
      out.close();

      cout << "Results appended to: " << output_file << endl;
   }

   // Cleanup
   delete J;
   delete D;
   delete P;
   delete JtDJ_orig;
   delete PTAP_orig;
   delete Areduced;

   return 0;
}
