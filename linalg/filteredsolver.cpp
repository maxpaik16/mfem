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

#include "filteredsolver.hpp"
#include "sparsemat.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <random>
#include <iomanip>
#include <fstream>
#ifdef MFEM_USE_PETSC
#include "petsc.hpp"
#endif

namespace mfem
{

namespace
{

double WallTime()
{
   using clock = std::chrono::steady_clock;
   return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

#ifdef MFEM_USE_MPI
bool GetOperatorComm(const Operator *op, MPI_Comm &comm)
{
   if (const auto *Ah = dynamic_cast<const HypreParMatrix*>(op))
   {
      comm = Ah->GetComm();
      return true;
   }
#ifdef MFEM_USE_PETSC
   if (const auto *Ap = dynamic_cast<const PetscParMatrix*>(op))
   {
      comm = Ap->GetComm();
      return true;
   }
#endif
   return false;
}
#endif

} // namespace

std::unique_ptr<const Operator> FilteredSolver::GetPtAP(const Operator *Aop,
                                                        const Operator *Pop) const
{
#ifdef MFEM_USE_MPI
   const HypreParMatrix * Ah = dynamic_cast<const HypreParMatrix*>(Aop);
   const HypreParMatrix * Ph = dynamic_cast<const HypreParMatrix*>(Pop);
   if (Ah && Ph) { return std::unique_ptr<const Operator>(RAP(Ah, Ph)); }
#endif
#ifdef MFEM_USE_PETSC
   PetscParMatrix* Ap = const_cast<PetscParMatrix*>(
                           dynamic_cast<const PetscParMatrix*>(Aop));
   PetscParMatrix* Pp = const_cast<PetscParMatrix*>(
                           dynamic_cast<const PetscParMatrix*>(Pop));
   if (Ap && Pp) { return std::unique_ptr<const Operator>(RAP(Ap, Pp)); }
#endif
   const SparseMatrix * Asp = dynamic_cast<const SparseMatrix*>(Aop);
   const SparseMatrix * Psp = dynamic_cast<const SparseMatrix*>(Pop);
   if (Asp && Psp)   { return std::unique_ptr<const Operator>(RAP(*Asp, *Psp)); }

   return std::unique_ptr<const Operator>(new RAPOperator(*Pop, *Aop, *Pop));
}

void FilteredSolver::InitVectors() const
{
   MFEM_VERIFY(A, "Operator not set");
   MFEM_VERIFY(P, "Transfer operator not set");
   MFEM_VERIFY(B, "Solver is not set.");
   MFEM_VERIFY(S, "Filtered space solver is not set.");

   z.SetSize(height);
   z.UseDevice(true);
   r.SetSize(height);
   r.UseDevice(true);
   xf.SetSize(P->Width());
   xf.UseDevice(true);
   rf.SetSize(P->Width());
   rf.UseDevice(true);
}

void FilteredSolver::MakeSolver() const
{
   if (solver_set) { return; }

   InitVectors();

   const double setup_start = WallTime();

   // Original space solver
   const double base_start = WallTime();
   B->SetOperator(*A);
   setup_base_solver_time = WallTime() - base_start;

   // Filtered space operator
   const double ptap_start = WallTime();
   PtAP = GetPtAP(A, P);
   setup_ptap_time = WallTime() - ptap_start;

   // Filtered space solver
   const double subspace_start = WallTime();
   S->SetOperator(*PtAP);
   setup_subspace_solver_time = WallTime() - subspace_start;

   setup_total_time = WallTime() - setup_start;
   have_timing_data = true;

   solver_set = true;
}

void FilteredSolver::SetOperator(const Operator &op)
{
   FlushTimingData();
   ResetTimingData();
   A = &op;
   height = op.Height();
   width = op.Width();
   solver_set = false;
}

void FilteredSolver::SetSolver(Solver &B_)
{
   FlushTimingData();
   ResetTimingData();
   B = &B_;
   solver_set = false;
}

void FilteredSolver::SetFilteredSubspaceTransferOperator(const Operator &P_)
{
   FlushTimingData();
   ResetTimingData();
   P = &P_;
   solver_set = false;
}

void FilteredSolver::SetFilteredSubspaceSolver(Solver &S_)
{
   FlushTimingData();
   ResetTimingData();
   S = &S_;
   solver_set = false;
}

void FilteredSolver::Mult(const Vector &b, Vector &x) const
{
   MFEM_VERIFY(b.Size() == x.Size(), "Inconsistent b and x size");
   MakeSolver();

   const double total_start = WallTime();

   x = 0.0;
   r = b;
   double amg_time = 0.0;
   double subspace_time = 0.0;

   if (reversed_order)
   {
      // Reversed order: subspace -> AMG -> subspace

      // rf = Pᵀ r
      const double subspace_start_1 = WallTime();
      P->MultTranspose(b, rf);

      // xf = S rf
      S->Mult(rf, xf);

      // z = P xf
      P->Mult(xf, z);
      subspace_time += WallTime() - subspace_start_1;

      // x = x + z
      x+=z;

      // r = b - A x = r - A z
      A->AddMult(z, r, -1.0);

      // z = B r
      const double amg_start = WallTime();
      B->Mult(r, z);
      amg_time += WallTime() - amg_start;
      // x = x + z
      x+=z;

      // r = b - A x = r - A z
      A->AddMult(z, r, -1.0);

      // rf = Pᵀ r
      const double subspace_start_2 = WallTime();
      P->MultTranspose(r, rf);

      // xf = S rf
      S->Mult(rf, xf);

      // z = P xf
      P->Mult(xf, z);
      subspace_time += WallTime() - subspace_start_2;

      // x = x + z
      x+=z;
   }
   else
   {
      // Standard order: AMG -> subspace -> AMG

      // z = B x
      const double amg_start_1 = WallTime();
      B->Mult(b, z);
      amg_time += WallTime() - amg_start_1;
      // x = x + z
      x+=z;

      // r = b - A x = r - A z
      A->AddMult(z, r, -1.0);

      // rf = Pᵀ r
      const double subspace_start = WallTime();
      P->MultTranspose(r, rf);

      // xf = S rf
      S->Mult(rf, xf);

      // z = P xf
      P->Mult(xf, z);
      subspace_time += WallTime() - subspace_start;

      // x = x + z
      x+=z;

      // r = b - A x = r - A z
      A->AddMult(z, r, -1.0);

      // z = B r
      const double amg_start_2 = WallTime();
      B->Mult(r, z);
      amg_time += WallTime() - amg_start_2;
      x+=z;
   }

   const double total_time = WallTime() - total_start;
   const double other_time = std::max(0.0, total_time - amg_time - subspace_time);

   cycle_total_time += total_time;
   cycle_amg_time += amg_time;
   cycle_subspace_time += subspace_time;
   cycle_other_time += other_time;
   cycle_num_mult_calls++;
   have_timing_data = true;
}

void FilteredSolver::SelectFilteredSubspace(const FilteredSolver::SubspaceSelectionMethod m_)
{
   /*auto Ah = dynamic_cast<const HypreParMatrix*>(A);
   MFEM_VERIFY(Ah, "AMGFSolver::SelectFilteredSubspace: HypreParMatrix expected.");

   HypreParVector l1_row_norms(*Ah);

   SparseMatrix merged_A;
   A.MergeDiagAndOffd(merged_A);

   HYPRE_BigInt row_start = Ah.GetRowStarts()[0];
   HYPRE_BigInt row_end = Ah.GetRowStarts()[1];
   int local_num_rows = Ah.GetNumRows();
   HYPRE_BigInt global_num_cols = Ah.N();

   // Process each row
   for (int local_row = 0; local_row < local_num_rows; ++local_row)
   {
      const real_t *vals = merged_A.GetRowEntries(local_row);
      int row_size = merged_A.RowSize(local_row);
      double norm = 0.0;
      for (int i = 0; i < row_size; ++i)
      {
         norm += fabs(vals[i]);
      }
      l1_row_norms(local_row) = norm; 
   }

   Synch

   switch (m_)
   {
      case FilteredSolver::SubspaceSelectionMethod::GRADIENT:
         break;
      case FilteredSolver::SubspaceSelectionMethod::KNEE:
         break;
      case FilteredSolver::SubspaceSelectionMethod::GMM:
         break;
      case FilteredSolver::SubspaceSelectionMethod::COST:
         break;
   }

   P = ???;
   solver_set = false;*/
}

void FilteredSolver::ResetTimingData() const
{
   setup_total_time = 0.0;
   setup_base_solver_time = 0.0;
   setup_ptap_time = 0.0;
   setup_subspace_solver_time = 0.0;
   cycle_total_time = 0.0;
   cycle_amg_time = 0.0;
   cycle_subspace_time = 0.0;
   cycle_other_time = 0.0;
   cycle_num_mult_calls = 0;
   have_timing_data = false;
}

void FilteredSolver::FlushTimingData() const
{
   if (print_level <= 0 || !have_timing_data) { return; }

   int myid = 0;

   std::array<double, 4> setup_local = {
      setup_total_time,
      setup_base_solver_time,
      setup_ptap_time,
      setup_subspace_solver_time
   };
   std::array<double, 4> solve_local = {
      cycle_total_time,
      cycle_amg_time,
      cycle_subspace_time,
      cycle_other_time
   };
   int reported_num_mult_calls = cycle_num_mult_calls;

#ifdef MFEM_USE_MPI
   MPI_Comm comm = MPI_COMM_NULL;
   if (GetOperatorComm(A, comm) || GetOperatorComm(P, comm))
   {
      MPI_Comm_rank(comm, &myid);
   }
#else
   MFEM_CONTRACT_VAR(A);
   MFEM_CONTRACT_VAR(P);
#endif

   if (myid == 0)
   {
      const double setup_subspace_total = setup_local[2] + setup_local[3];
      mfem::out << "AMGF filtered setup time [s]: total=" << setup_local[0]
                << ", AMG=" << setup_local[1]
                << ", subspace=" << setup_subspace_total
                << " (PtAP=" << setup_local[2]
                << ", solver=" << setup_local[3] << ")" << std::endl;

      if (reported_num_mult_calls > 0)
      {
         const double solve_total =
            solve_local[1] + solve_local[2] + solve_local[3];
         const double inv_total = (solve_total > 0.0) ? 1.0 / solve_total : 0.0;
         mfem::out << "AMGF solve time split [s]: total=" << solve_local[0]
                   << ", AMG=" << solve_local[1]
                   << " (" << 100.0 * solve_local[1] * inv_total << "%)"
                   << ", subspace=" << solve_local[2]
                   << " (" << 100.0 * solve_local[2] * inv_total << "%)"
                   << ", other=" << solve_local[3]
                   << " (" << 100.0 * solve_local[3] * inv_total << "%)"
                   << ", applies=" << reported_num_mult_calls << std::endl;
      }
   }
}

FilteredSolver::~FilteredSolver()
{ }

#ifdef MFEM_USE_MPI

void AMGFSolver::SetOperator(const Operator &A_)
{
   auto Ah = dynamic_cast<const HypreParMatrix*>(&A_);
   MFEM_VERIFY(Ah, "AMGFSolver::SetOperator: HypreParMatrix expected.");
   FilteredSolver::SetOperator(*Ah);
}

void AMGFSolver::SetFilteredSubspaceTransferOperator(const HypreParMatrix &Pop)
{
   FilteredSolver::SetFilteredSubspaceTransferOperator(Pop);
}

bool AMGFSolver::TestSPD(int num_tests, int seed, bool verbose) const
{
   MFEM_VERIFY(A != nullptr, "AMGFSolver::TestSPD: Operator must be set before testing.");
   MFEM_VERIFY(solver_set, "AMGFSolver::TestSPD: Solver must be set up before testing.");

   const auto *Ah = dynamic_cast<const HypreParMatrix*>(A);
   MFEM_VERIFY(Ah, "AMGFSolver::TestSPD: HypreParMatrix expected.");

   MPI_Comm comm = Ah->GetComm();
   int myid, num_procs;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   const int n = height;

   if (verbose && myid == 0)
   {
      mfem::out << "\n=== AMGFSolver SPD Test ===" << std::endl;
      mfem::out << "Number of tests per category: " << num_tests << std::endl;
      mfem::out << "Local vector size: " << n << std::endl;
      if (P)
      {
         mfem::out << "Testing both full space and complement space vectors" << std::endl;
      }
      else
      {
         mfem::out << "Testing full space vectors only (P not set)" << std::endl;
      }
   }

   std::mt19937 rng(seed + myid);
   std::normal_distribution<real_t> dist(0.0, 1.0);

   int symmetry_failures = 0;
   int positive_def_failures = 0;
   real_t max_symmetry_error = 0.0;
   real_t min_eigenvalue_estimate = std::numeric_limits<real_t>::max();

   const auto *Ph = dynamic_cast<const HypreParMatrix*>(P);
   const int num_test_types = (Ph != nullptr) ? 2 : 1;

   for (int test_type = 0; test_type < num_test_types; test_type++)
   {
      for (int test = 0; test < num_tests; test++)
      {
         Vector u(n), v(n);
         for (int i = 0; i < n; i++)
         {
            u(i) = dist(rng);
            v(i) = dist(rng);
         }

         // For test_type == 1, project out the contact subspace component
         // to get vectors in the complement space: u_comp = u - P(P^T u)
         if (test_type == 1)
         {
            const int n_subspace = Ph->Width();
            Vector Ptu(n_subspace), PPtu(n);

            Ph->MultTranspose(u, Ptu);
            Ph->Mult(Ptu, PPtu);
            u.Add(-1.0, PPtu);

            Vector Ptv(n_subspace), PPtv(n);
            Ph->MultTranspose(v, Ptv);
            Ph->Mult(Ptv, PPtv);
            v.Add(-1.0, PPtv);
         }

         Vector Mu(n), Mv(n);
         Mult(u, Mu);
         Mult(v, Mv);

         // Test symmetry: (M u, v) = (u, M v)
         real_t Mu_dot_v = InnerProduct(Mu, v);
         real_t u_dot_Mv = InnerProduct(u, Mv);

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
         real_t u_dot_Mu = InnerProduct(u, Mu);
         real_t global_u_dot_Mu;
         MPI_Allreduce(&u_dot_Mu, &global_u_dot_Mu, 1, HYPRE_MPI_REAL, MPI_SUM, comm);

         min_eigenvalue_estimate = std::min(min_eigenvalue_estimate, global_u_dot_Mu);

         if (global_u_dot_Mu <= 0.0)
         {
            positive_def_failures++;
         }
      }
   }

   const int total_tests = num_tests * num_test_types;

   int global_symmetry_failures, global_pd_failures;
   MPI_Reduce(&symmetry_failures, &global_symmetry_failures, 1, MPI_INT, MPI_SUM, 0, comm);
   MPI_Reduce(&positive_def_failures, &global_pd_failures, 1, MPI_INT, MPI_SUM, 0, comm);

   real_t global_max_sym_error, global_min_eigenvalue;
   MPI_Reduce(&max_symmetry_error, &global_max_sym_error, 1, HYPRE_MPI_REAL, MPI_MAX, 0, comm);
   MPI_Reduce(&min_eigenvalue_estimate, &global_min_eigenvalue, 1, HYPRE_MPI_REAL, MPI_MIN, 0, comm);

   bool is_spd = (global_symmetry_failures == 0 && global_pd_failures == 0);

   if (verbose && myid == 0)
   {
      mfem::out << "\n=== Test Results ===" << std::endl;
      mfem::out << "Total tests: " << total_tests << std::endl;
      if (Ph)
      {
         mfem::out << "  - Full space vectors: " << num_tests << std::endl;
         mfem::out << "  - Complement space vectors (orthogonal to contact): " << num_tests << std::endl;
      }
      mfem::out << "\nSymmetry test:" << std::endl;
      mfem::out << "  Failures: " << global_symmetry_failures << " / " << total_tests << std::endl;
      mfem::out << "  Max relative error: " << std::scientific << std::setprecision(6)
                << global_max_sym_error << std::endl;
      mfem::out << "  Status: " << (global_symmetry_failures == 0 ? "PASS" : "FAIL") << std::endl;

      mfem::out << "\nPositive-definiteness test:" << std::endl;
      mfem::out << "  Failures: " << global_pd_failures << " / " << total_tests << std::endl;
      mfem::out << "  Min (u, M u): " << std::scientific << std::setprecision(6)
                << global_min_eigenvalue << std::endl;
      mfem::out << "  Status: " << (global_pd_failures == 0 ? "PASS" : "FAIL") << std::endl;

      mfem::out << "\nOverall SPD status: " << (is_spd ? "PASS" : "FAIL") << std::endl;
   }

   return is_spd;
}

void AMGFSolver::VerifyOperatorFormulations(int num_tests, int seed, bool verbose) const
{
   MFEM_VERIFY(A != nullptr, "AMGFSolver::VerifyOperatorFormulations: Operator must be set.");
   MFEM_VERIFY(solver_set, "AMGFSolver::VerifyOperatorFormulations: Solver must be set up.");
   MFEM_VERIFY(P != nullptr, "AMGFSolver::VerifyOperatorFormulations: P must be set.");
   MFEM_VERIFY(S != nullptr, "AMGFSolver::VerifyOperatorFormulations: S must be set.");
   MFEM_VERIFY(amg != nullptr, "AMGFSolver::VerifyOperatorFormulations: AMG must be set.");

   const auto *Ah = dynamic_cast<const HypreParMatrix*>(A);
   MFEM_VERIFY(Ah, "AMGFSolver::VerifyOperatorFormulations: HypreParMatrix expected.");

   MPI_Comm comm = Ah->GetComm();
   int myid, num_procs;
   MPI_Comm_rank(comm, &myid);
   MPI_Comm_size(comm, &num_procs);

   const int n = height;

   if (verbose && myid == 0)
   {
      mfem::out << "\n=== Operator Formulation Verification ===" << std::endl;
      mfem::out << "Number of tests: " << num_tests << std::endl;
      mfem::out << "Vector size: " << n << std::endl;
   }

   // Helper lambda: Apply subspace operator M = P S P^T
   auto ApplySubspace = [&](const Vector &in, Vector &out) -> void
   {
      Vector rf_local(P->Width()), xf_local(P->Width());
      P->MultTranspose(in, rf_local);
      S->Mult(rf_local, xf_local);
      P->Mult(xf_local, out);
   };

   // Helper lambda: Apply original code operator C_code
   auto ApplyCodeOperator = [&](const Vector &b, Vector &x) -> void
   {
      x = 0.0;
      Vector r(b), z(n);

      // Step 1: Subspace
      ApplySubspace(r, z);
      x += z;
      A->AddMult(z, r, -1.0);

      // Step 2: AMG
      amg->Mult(r, z);
      x += z;
      A->AddMult(z, r, -1.0);

      // Step 3: Subspace
      ApplySubspace(r, z);
      x += z;
   };

   // Helper lambda: Apply symmetric sandwich operator C_sandwich = M + (I-MA)B(I-AM)
   auto ApplySandwichOperator = [&](const Vector &b, Vector &out) -> void
   {
      Vector u(n), r1(n), v(n), Av(n), MAv(n);

      // u = M b
      ApplySubspace(b, u);

      // r1 = (I - A M)b
      r1 = b;
      A->AddMult(u, r1, -1.0);

      // v = B r1
      amg->Mult(r1, v);

      // MAv = M(A v)
      A->Mult(v, Av);
      ApplySubspace(Av, MAv);

      // out = u + v - MAv
      out = u;
      out += v;
      out -= MAv;
   };

   // Helper lambda: Apply extra term H_extra = M - MAM
   auto ApplyExtraTerm = [&](const Vector &b, Vector &out) -> void
   {
      Vector Mb(n), AMb(n), MAMb(n);

      ApplySubspace(b, Mb);
      A->Mult(Mb, AMb);
      ApplySubspace(AMb, MAMb);

      out = Mb;
      out -= MAMb;
   };

   // Helper lambda: Apply decomposed operator C_decomposed = C_sandwich + H_extra
   auto ApplyDecomposedOperator = [&](const Vector &b, Vector &out) -> void
   {
      Vector sandwich_out(n), extra_out(n);
      ApplySandwichOperator(b, sandwich_out);
      ApplyExtraTerm(b, extra_out);
      out = sandwich_out;
      out += extra_out;
   };

   // Test vectors
   std::mt19937 rng(seed + myid);
   std::normal_distribution<real_t> dist(0.0, 1.0);

   // Statistics
   real_t max_code_vs_decomp_error = 0.0;
   real_t max_subspace_sym_error = 0.0;
   real_t max_amg_sym_error = 0.0;
   real_t max_sandwich_sym_error = 0.0;
   real_t max_extra_sym_error = 0.0;
   real_t max_code_sym_error = 0.0;
   int subspace_pos_failures = 0;
   int amg_pos_failures = 0;
   int sandwich_pos_failures = 0;
   int extra_pos_failures = 0;
   int code_pos_failures = 0;
   real_t max_mam_ratio = 0.0;

   for (int test = 0; test < num_tests; test++)
   {
      // Generate random vectors
      Vector b(n), u(n), v(n);
      for (int i = 0; i < n; i++)
      {
         b(i) = dist(rng);
         u(i) = dist(rng);
         v(i) = dist(rng);
      }

      // Test 1: Verify C_code ≈ C_decomposed
      Vector code_out(n), decomp_out(n);
      ApplyCodeOperator(b, code_out);
      ApplyDecomposedOperator(b, decomp_out);

      Vector diff(code_out);
      diff -= decomp_out;
      real_t local_diff_norm = diff.Norml2();
      real_t local_code_norm = code_out.Norml2();

      real_t global_diff_norm, global_code_norm;
      MPI_Allreduce(&local_diff_norm, &global_diff_norm, 1, HYPRE_MPI_REAL, MPI_SUM, comm);
      MPI_Allreduce(&local_code_norm, &global_code_norm, 1, HYPRE_MPI_REAL, MPI_SUM, comm);

      real_t rel_error = global_diff_norm / std::max(global_code_norm, 1e-16);
      max_code_vs_decomp_error = std::max(max_code_vs_decomp_error, rel_error);

      // Test 2: Symmetry tests
      Vector Mu(n), Mv(n), Bu(n), Bv(n);
      Vector sandwich_u(n), sandwich_v(n);
      Vector extra_u(n), extra_v(n);

      ApplySubspace(u, Mu);
      ApplySubspace(v, Mv);
      amg->Mult(u, Bu);
      amg->Mult(v, Bv);
      ApplySandwichOperator(u, sandwich_u);
      ApplySandwichOperator(v, sandwich_v);
      ApplyExtraTerm(u, extra_u);
      ApplyExtraTerm(v, extra_v);
      ApplyCodeOperator(u, code_out);
      Vector code_v(n);
      ApplyCodeOperator(v, code_v);

      // Compute inner products
      real_t local_dots[10];
      local_dots[0] = InnerProduct(Mu, v);
      local_dots[1] = InnerProduct(u, Mv);
      local_dots[2] = InnerProduct(Bu, v);
      local_dots[3] = InnerProduct(u, Bv);
      local_dots[4] = InnerProduct(sandwich_u, v);
      local_dots[5] = InnerProduct(u, sandwich_v);
      local_dots[6] = InnerProduct(extra_u, v);
      local_dots[7] = InnerProduct(u, extra_v);
      local_dots[8] = InnerProduct(code_out, v);
      local_dots[9] = InnerProduct(u, code_v);

      real_t global_dots[10];
      MPI_Allreduce(local_dots, global_dots, 10, HYPRE_MPI_REAL, MPI_SUM, comm);

      auto compute_sym_error = [](real_t dot1, real_t dot2) -> real_t
      {
         return std::abs(dot1 - dot2) / std::max(std::abs(dot1) + std::abs(dot2), 1e-16);
      };

      max_subspace_sym_error = std::max(max_subspace_sym_error, compute_sym_error(global_dots[0], global_dots[1]));
      max_amg_sym_error = std::max(max_amg_sym_error, compute_sym_error(global_dots[2], global_dots[3]));
      max_sandwich_sym_error = std::max(max_sandwich_sym_error, compute_sym_error(global_dots[4], global_dots[5]));
      max_extra_sym_error = std::max(max_extra_sym_error, compute_sym_error(global_dots[6], global_dots[7]));
      max_code_sym_error = std::max(max_code_sym_error, compute_sym_error(global_dots[8], global_dots[9]));

      // Test 3: Positivity tests
      real_t local_pos[5];
      local_pos[0] = InnerProduct(u, Mu);
      local_pos[1] = InnerProduct(u, Bu);
      local_pos[2] = InnerProduct(u, sandwich_u);
      local_pos[3] = InnerProduct(u, extra_u);
      local_pos[4] = InnerProduct(u, code_out);

      real_t global_pos[5];
      MPI_Allreduce(local_pos, global_pos, 5, HYPRE_MPI_REAL, MPI_SUM, comm);

      if (global_pos[0] <= 0) subspace_pos_failures++;
      if (global_pos[1] <= 0) amg_pos_failures++;
      if (global_pos[2] <= 0) sandwich_pos_failures++;
      if (global_pos[3] <= 0) extra_pos_failures++;
      if (global_pos[4] <= 0) code_pos_failures++;

      // Test 4: Measure ||M - MAM|| / ||M||
      Vector Mb(n), AMb(n), MAMb(n);
      ApplySubspace(b, Mb);
      A->Mult(Mb, AMb);
      ApplySubspace(AMb, MAMb);

      Vector M_minus_MAM_b(Mb);
      M_minus_MAM_b -= MAMb;

      real_t local_mam_norm = M_minus_MAM_b.Norml2();
      real_t local_mb_norm = Mb.Norml2();

      real_t global_mam_norm, global_mb_norm;
      MPI_Allreduce(&local_mam_norm, &global_mam_norm, 1, HYPRE_MPI_REAL, MPI_SUM, comm);
      MPI_Allreduce(&local_mb_norm, &global_mb_norm, 1, HYPRE_MPI_REAL, MPI_SUM, comm);

      real_t mam_ratio = global_mam_norm / std::max(global_mb_norm, 1e-16);
      max_mam_ratio = std::max(max_mam_ratio, mam_ratio);
   }

   // Gather results on rank 0
   real_t global_max_code_vs_decomp, global_max_mam_ratio;
   real_t global_max_sym_errors[5];
   int global_pos_failures[5];

   real_t local_max_sym[5] = {max_subspace_sym_error, max_amg_sym_error, max_sandwich_sym_error,
                               max_extra_sym_error, max_code_sym_error};
   int local_pos_fail[5] = {subspace_pos_failures, amg_pos_failures, sandwich_pos_failures,
                            extra_pos_failures, code_pos_failures};

   MPI_Reduce(&max_code_vs_decomp_error, &global_max_code_vs_decomp, 1, HYPRE_MPI_REAL, MPI_MAX, 0, comm);
   MPI_Reduce(&max_mam_ratio, &global_max_mam_ratio, 1, HYPRE_MPI_REAL, MPI_MAX, 0, comm);
   MPI_Reduce(local_max_sym, global_max_sym_errors, 5, HYPRE_MPI_REAL, MPI_MAX, 0, comm);
   MPI_Reduce(local_pos_fail, global_pos_failures, 5, MPI_INT, MPI_SUM, 0, comm);

   if (verbose && myid == 0)
   {
      mfem::out << "\n=== Equivalence Test ===" << std::endl;
      mfem::out << "Max ||C_code - C_decomposed|| / ||C_code||: "
                << std::scientific << std::setprecision(6) << global_max_code_vs_decomp << std::endl;
      mfem::out << "Status: " << (global_max_code_vs_decomp < 1e-8 ? "PASS" : "FAIL")
                << " (expected near machine precision)" << std::endl;

      mfem::out << "\n=== Symmetry Tests ===" << std::endl;
      mfem::out << "Operator           | Max |(u,Tv) - (Tu,v)| / (|(u,Tv)| + |(Tu,v)|) | Status" << std::endl;
      mfem::out << "-------------------+-----------------------------------------------+--------" << std::endl;

      auto print_sym = [&](const char* name, real_t err, int idx)
      {
         mfem::out << std::left << std::setw(19) << name
                   << "| " << std::scientific << std::setprecision(6) << std::setw(46) << err
                   << "| " << (err < 1e-10 ? "PASS" : "FAIL") << std::endl;
      };

      print_sym("M (subspace)", global_max_sym_errors[0], 0);
      print_sym("B (AMG)", global_max_sym_errors[1], 1);
      print_sym("C_sandwich", global_max_sym_errors[2], 2);
      print_sym("H_extra", global_max_sym_errors[3], 3);
      print_sym("C_code", global_max_sym_errors[4], 4);

      mfem::out << "\n=== Positivity Tests ===" << std::endl;
      mfem::out << "Operator           | Failures / " << num_tests << " | Expected Result" << std::endl;
      mfem::out << "-------------------+------------+----------------------------------" << std::endl;

      auto print_pos = [&](const char* name, int failures, const char* expected)
      {
         mfem::out << std::left << std::setw(19) << name
                   << "| " << std::setw(11) << failures
                   << "| " << expected << std::endl;
      };

      print_pos("M (subspace)", global_pos_failures[0], "Should be 0 (PSD)");
      print_pos("B (AMG)", global_pos_failures[1], "Should be 0 (SPD)");
      print_pos("C_sandwich", global_pos_failures[2], "Should be 0 if A SPD, M PSD, B SPD");
      print_pos("H_extra", global_pos_failures[3], "Can be positive or negative");
      print_pos("C_code", global_pos_failures[4], "Can be negative (inexact subspace)");

      mfem::out << "\n=== Exact Subspace Identity Test ===" << std::endl;
      mfem::out << "Max ||(M - MAM)b|| / ||Mb||: "
                << std::scientific << std::setprecision(6) << global_max_mam_ratio << std::endl;
      mfem::out << "This measures failure of MAM = M (exact subspace identity)." << std::endl;
      mfem::out << "Value near zero indicates exact Galerkin inverse; " << std::endl;
      mfem::out << "larger values indicate inexact subspace solve." << std::endl;
   }
}

#endif

} // namespace mfem
