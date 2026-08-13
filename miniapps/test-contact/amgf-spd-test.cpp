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

//                  AMGF Preconditioner SPD Test
//
// Description:
// This miniapp tests whether the AMGF preconditioner (with optional Schwarz
// smoothing) is symmetric positive definite (SPD) by loading a matrix from
// disk and repeatedly checking:
//   1. Symmetry: (M u, v) = (u, M v) for random vectors u and v
//   2. Positive-definiteness: (u, M u) > 0 for random vector u
//
// The test exposes AMGF configuration options including reversed order and
// Schwarz preconditioner settings through command-line arguments.
//
// Compile with: make amgf-spd-test
//
// Sample runs:
// mpirun -np 4 ./amgf-spd-test -mats-dir mats -iteration 0
// mpirun -np 4 ./amgf-spd-test -mats-dir mats -iteration 0 -num-tests 100
// mpirun -np 4 ./amgf-spd-test -mats-dir mats -iteration 0 -reversed-order
// mpirun -np 4 ./amgf-spd-test -mats-dir mats -iteration 0 -use-schwarz

#include "mfem.hpp"
#include "solver_utils.hpp"
#include "../common/schwarz_solver.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <random>

using namespace std;
using namespace mfem;

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
   mat->Read(comm, filename);

   if (myid == 0)
   {
      cout << "Matrix loaded: " << mat->M() << " x " << mat->N() << endl;
   }

   return mat;
}

// Build Schwarz subdomains from J matrix mapped to contact subspace via P
void BuildSchwarzSubdomainsFromJ(HypreParMatrix* J, HypreSchwarz* schwarz,
                                  HypreParMatrix* PTAP,
                                  HypreParMatrix* P,
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

   // Access PTAP diagonal for graph queries
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

         if (schwarz_min_diag_value > 0.0 && i < D_diag.Size())
         {
            if (D_diag(i) < schwarz_min_diag_value)
            {
               continue;
            }
         }

         std::set<HYPRE_BigInt> subdomain_set;

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
   }

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

   if (myid == 0)
   {
      cout << "\nSchwarz subdomain statistics:" << endl;
      cout << "  Total number of subdomains: " << total_subdomains << endl;
   }

   if (!schwarz->schwarz_solver)
   {
      HYPRE_SchwarzCreate(&schwarz->schwarz_solver);
      HYPRE_SchwarzSetVariant(schwarz->schwarz_solver, HypreSchwarz::RequiredVariant);
   }

   HYPRE_Int use_nonsymm = 0;
   schwarz->SetCustomSubdomains(subdomains, *PTAP, schwarz_relax_weight, use_nonsymm, schwarz_unweighted, schwarz_uniform_weight);
}

// Test if preconditioner is SPD by checking symmetry and positive-definiteness
void TestPreconditionerSPD(Solver* prec, int n, int num_tests, MPI_Comm comm, int seed)
{
   int myid;
   MPI_Comm_rank(comm, &myid);

   if (myid == 0)
   {
      cout << "\n=== Running SPD Tests ===" << endl;
      cout << "Number of tests: " << num_tests << endl;
      cout << "Vector size: " << n << endl;
   }

   std::mt19937 rng(seed + myid);
   std::normal_distribution<real_t> dist(0.0, 1.0);

   int symmetry_failures = 0;
   int positive_def_failures = 0;
   real_t max_symmetry_error = 0.0;
   real_t min_eigenvalue_estimate = std::numeric_limits<real_t>::max();

   for (int test = 0; test < num_tests; test++)
   {
      // Generate random vectors u and v
      Vector u(n), v(n);
      for (int i = 0; i < n; i++)
      {
         u(i) = dist(rng);
         v(i) = dist(rng);
      }

      // Compute M*u and M*v
      Vector Mu(n), Mv(n);
      prec->Mult(u, Mu);
      prec->Mult(v, Mv);

      // Test symmetry: (M u, v) = (u, M v)
      real_t Mu_dot_v = Mu * v;
      real_t u_dot_Mv = u * Mv;

      real_t local_dots[2] = {Mu_dot_v, u_dot_Mv};
      real_t global_dots[2];
      MPI_Allreduce(local_dots, global_dots, 2, HYPRE_MPI_REAL, MPI_SUM, comm);

      real_t sym_error = std::abs(global_dots[0] - global_dots[1]) /
                         (std::abs(global_dots[0]) + std::abs(global_dots[1]) + 1e-16);

      max_symmetry_error = std::max(max_symmetry_error, sym_error);

      if (sym_error > 1e-10)
      {
         symmetry_failures++;
      }

      // Test positive-definiteness: (u, M u) > 0
      real_t u_dot_Mu = u * Mu;
      real_t global_u_dot_Mu;
      MPI_Allreduce(&u_dot_Mu, &global_u_dot_Mu, 1, HYPRE_MPI_REAL, MPI_SUM, comm);

      min_eigenvalue_estimate = std::min(min_eigenvalue_estimate, global_u_dot_Mu);

      if (global_u_dot_Mu <= 0.0)
      {
         positive_def_failures++;
      }
   }

   // Gather results
   int global_symmetry_failures, global_pd_failures;
   MPI_Reduce(&symmetry_failures, &global_symmetry_failures, 1, MPI_INT, MPI_SUM, 0, comm);
   MPI_Reduce(&positive_def_failures, &global_pd_failures, 1, MPI_INT, MPI_SUM, 0, comm);

   real_t global_max_sym_error, global_min_eigenvalue;
   MPI_Reduce(&max_symmetry_error, &global_max_sym_error, 1, HYPRE_MPI_REAL, MPI_MAX, 0, comm);
   MPI_Reduce(&min_eigenvalue_estimate, &global_min_eigenvalue, 1, HYPRE_MPI_REAL, MPI_MIN, 0, comm);

   if (myid == 0)
   {
      cout << "\n=== Test Results ===" << endl;
      cout << "Symmetry test:" << endl;
      cout << "  Failures: " << global_symmetry_failures << " / " << num_tests << endl;
      cout << "  Max relative error: " << scientific << setprecision(6)
           << global_max_sym_error << endl;
      cout << "  Status: " << (global_symmetry_failures == 0 ? "PASS" : "FAIL") << endl;

      cout << "\nPositive-definiteness test:" << endl;
      cout << "  Failures: " << global_pd_failures << " / " << num_tests << endl;
      cout << "  Min (u, M u): " << scientific << setprecision(6)
           << global_min_eigenvalue << endl;
      cout << "  Status: " << (global_pd_failures == 0 ? "PASS" : "FAIL") << endl;

      cout << "\nOverall SPD status: "
           << ((global_symmetry_failures == 0 && global_pd_failures == 0) ? "PASS" : "FAIL")
           << endl;
   }
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
   int num_tests = 20;
   int seed = 12345;
   bool use_schwarz = false;
   bool reversed_order = false;
   real_t schwarz_weight = 1.0;
   bool schwarz_unweighted = false;
   real_t schwarz_uniform_weight = 0.1;

   OptionsParser args(argc, argv);
   args.AddOption(&mats_dir, "-mats-dir", "--matrices-directory",
                  "Directory containing matrix files.");
   args.AddOption(&iteration, "-iteration", "--iteration",
                  "Iteration number to load.");
   args.AddOption(&num_tests, "-num-tests", "--num-tests",
                  "Number of random vector tests to perform.");
   args.AddOption(&seed, "-seed", "--random-seed",
                  "Random seed for reproducibility.");
   args.AddOption(&use_schwarz, "-use-schwarz", "--use-schwarz",
                  "-no-schwarz", "--no-schwarz",
                  "Use Schwarz preconditioner in AMGF subspace solver.");
   args.AddOption(&reversed_order, "-reversed-order", "--reversed-order",
                  "-no-reversed-order", "--no-reversed-order",
                  "Use reversed order in AMGF (AMG after filtered subspace solve).");
   args.AddOption(&schwarz_weight, "-schwarz-weight", "--schwarz-weight",
                  "Schwarz relaxation weight.");
   args.AddOption(&schwarz_unweighted, "-schwarz-unweighted", "--schwarz-unweighted",
                  "-no-schwarz-unweighted", "--no-schwarz-unweighted",
                  "Disable per-DOF scaling in Schwarz preconditioner.");
   args.AddOption(&schwarz_uniform_weight, "-schwarz-uniform-weight",
                  "--schwarz-uniform-weight",
                  "Set uniform weight for Schwarz (default 0.1, -1 to disable).");

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
   HypreParMatrix* PTAP = LoadHypreMatrix(ss_ptap.str(), MPI_COMM_WORLD);
   HypreParMatrix* Areduced = LoadHypreMatrix(ss_areduced.str(), MPI_COMM_WORLD);

   if (myid == 0)
   {
      cout << "\nMatrix dimensions:" << endl;
      cout << "  J:        " << J->M() << " x " << J->N() << endl;
      cout << "  D:        " << D->M() << " x " << D->N() << endl;
      cout << "  P:        " << P->M() << " x " << P->N() << endl;
      cout << "  PTAP:     " << PTAP->M() << " x " << PTAP->N() << endl;
      cout << "  Areduced: " << Areduced->M() << " x " << Areduced->N() << endl;
   }

   // Build AMGF preconditioner
   if (myid == 0)
   {
      cout << "\n=== Building AMGF Preconditioner ===" << endl;
      cout << "Configuration:" << endl;
      cout << "  Reversed order: " << (reversed_order ? "Yes" : "No") << endl;
      cout << "  Use Schwarz: " << (use_schwarz ? "Yes" : "No") << endl;
      if (use_schwarz)
      {
         cout << "  Schwarz weight: " << schwarz_weight << endl;
         cout << "  Schwarz unweighted: " << (schwarz_unweighted ? "Yes" : "No") << endl;
         cout << "  Schwarz uniform weight: " << schwarz_uniform_weight << endl;
      }
   }

   double setup_start = MPI_Wtime();

   AMGFSolver* amgf = new AMGFSolver();

   // Configure AMG component
   const int relax_type = 8;
   const int coarsen_type = 10;
   amgf->GetAMG().SetSystemsOptions(3);
   amgf->GetAMG().SetPrintLevel(0);
   amgf->GetAMG().SetRelaxType(relax_type);
   amgf->GetAMG().SetCoarsening(coarsen_type);

   amgf->SetPrintLevel(0);
   amgf->SetReversedOrder(reversed_order);

   // Configure subspace solver
   Solver* subspace_solver = nullptr;
   HypreSchwarz* schwarz_ptr = nullptr;

   if (use_schwarz)
   {
      schwarz_ptr = new HypreSchwarz();
      schwarz_ptr->SetPrintLevel(0);

      HYPRE_SchwarzCreate(&(schwarz_ptr->schwarz_solver));
      HYPRE_SchwarzSetVariant(schwarz_ptr->schwarz_solver, HypreSchwarz::RequiredVariant);

      BuildSchwarzSubdomainsFromJ(J, schwarz_ptr, PTAP, P,
                                  false, 0.0,
                                  schwarz_weight, schwarz_unweighted,
                                  schwarz_uniform_weight, D);

      subspace_solver = schwarz_ptr;
   }
   else
   {
      auto* direct_solver = new ParallelDirectSolver(Areduced->GetComm(), "auto");
      direct_solver->SetPrintLevel(0);
      subspace_solver = direct_solver;
   }

   amgf->SetFilteredSubspaceSolver(*subspace_solver);
   amgf->SetFilteredSubspaceTransferOperator(*P);
   amgf->SetOperator(*Areduced);

   double setup_end = MPI_Wtime();

   if (myid == 0)
   {
      cout << "Setup time: " << scientific << setprecision(4)
           << (setup_end - setup_start) << " s" << endl;
   }

   // Run SPD tests
   TestPreconditionerSPD(amgf, Areduced->Height(), num_tests, MPI_COMM_WORLD, seed);

   // Cleanup
   delete amgf;
   if (subspace_solver) { delete subspace_solver; }
   delete J;
   delete D;
   delete P;
   delete PTAP;
   delete Areduced;

   return 0;
}
