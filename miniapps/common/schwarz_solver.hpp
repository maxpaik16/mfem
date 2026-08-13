// Copyright (c) 2010-2024, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_SCHWARZ_SOLVER_HPP
#define MFEM_SCHWARZ_SOLVER_HPP

#include "mfem.hpp"
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <unordered_map>

using namespace mfem;

// HYPRE and LAPACK declarations
extern "C"
{
#include "HYPRE.h"
#include "_hypre_parcsr_ls.h"
#include "_hypre_utilities.h"

void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info);
void dpotrs_(char *uplo, int *n, int *nrhs, double *a, int *lda,
             double *b, int *ldb, int *info);
void dgetrs_(const char *trans, int *n, int *nrhs, double *a, int *lda,
             int *ipiv, double *b, int *ldb, int *info);
void dsyev_(const char *jobz, const char *uplo, const int *n, double *a,
            const int *lda, double *w, double *work, const int *lwork, int *info);
void dgeev_(const char *jobvl, const char *jobvr, const int *n, double *a,
            const int *lda, double *wr, double *wi, double *vl, const int *ldvl,
            double *vr, const int *ldvr, double *work, const int *lwork, int *info);

typedef struct
{
   HYPRE_Int      variant;
   HYPRE_Int      domain_type;
   HYPRE_Int      overlap;
   HYPRE_Int      num_functions;
   HYPRE_Int      use_nonsymm;
   HYPRE_Real   relax_weight;

   hypre_CSRMatrix *domain_structure;
   hypre_CSRMatrix *A_boundary;
   hypre_ParVector *Vtemp;
   HYPRE_Real  *scale;
   HYPRE_Int     *dof_func;
   HYPRE_Int     *pivots;

} hypre_SchwarzData;
}

// Dummy setup function for HYPRE Schwarz solver
static HYPRE_Int DummyParSolverFcn(HYPRE_Solver solver,
                                   HYPRE_ParCSRMatrix A,
                                   HYPRE_ParVector b,
                                   HYPRE_ParVector x)
{
   (void)solver;
   (void)A;
   (void)b;
   (void)x;
   return 0;
}

/// Schwarz domain decomposition solver wrapping HYPRE's Schwarz implementation
class HypreSchwarz : public HypreSolver
{
public:
   static constexpr HYPRE_Int RequiredVariant = 2;
   HYPRE_Solver schwarz_solver;
   int print_level = 0;
   MPI_Comm custom_comm = MPI_COMM_NULL;
   HYPRE_BigInt custom_local_row_start = 0;
   HYPRE_Int custom_local_num_rows = 0;
   bool custom_use_nonsymm = false;
   bool use_custom_solve = false;
   std::vector<std::vector<HYPRE_BigInt>> custom_subdomains;
   std::vector<HYPRE_Int> custom_dense_offsets;
   std::vector<HYPRE_Int> custom_pivot_offsets;
   std::vector<HYPRE_Real> custom_dense_data;
   std::vector<HYPRE_Int> custom_pivots;
   std::unordered_map<HYPRE_BigInt, HYPRE_Real> custom_scale;
   std::vector<int> rhs_request_send_counts;
   std::vector<int> rhs_request_send_displs;
   std::vector<int> rhs_request_recv_counts;
   std::vector<int> rhs_request_recv_displs;
   std::vector<HYPRE_BigInt> rhs_request_send_ids;
   std::vector<HYPRE_BigInt> rhs_request_recv_ids;
   std::unordered_map<HYPRE_BigInt, HYPRE_Int> rhs_value_index;
   std::vector<int> correction_send_counts;
   std::vector<int> correction_send_displs;
   std::vector<int> correction_recv_counts;
   std::vector<int> correction_recv_displs;
   std::vector<HYPRE_BigInt> correction_send_ids;
   std::vector<HYPRE_BigInt> correction_recv_ids;

   static std::unordered_map<HYPRE_Solver, int> & PrintLevels()
   {
      static std::unordered_map<HYPRE_Solver, int> levels;
      return levels;
   }

   static std::unordered_map<HYPRE_Solver, HypreSchwarz *> & Instances()
   {
      static std::unordered_map<HYPRE_Solver, HypreSchwarz *> instances;
      return instances;
   }

   void SetPrintLevel(int level)
   {
      print_level = level;
      if (schwarz_solver)
      {
         PrintLevels()[schwarz_solver] = level;
         Instances()[schwarz_solver] = this;
      }
   }

   void SetOperator(const Operator &op) override
   {
      const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
      MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

      //HYPRE_SchwarzDestroy(schwarz_solver); TODO: fix leak :)
      //HYPRE_SchwarzCreate(&schwarz_solver);

      // update base classes: Operator, Solver, HypreSolver
      height = new_A->Height();
      width  = new_A->Width();
      A = const_cast<HypreParMatrix *>(new_A);
      setup_called = 0;
      delete X;
      delete B;
      B = X = NULL;
      auxB.Delete(); auxB.Reset();
      auxX.Delete(); auxX.Reset();

      if (schwarz_solver)
      {
         PrintLevels()[schwarz_solver] = print_level;
         Instances()[schwarz_solver] = this;
      }
   }

   HYPRE_CSRMatrix BuildDomainStructure(
      const std::vector<std::vector<HYPRE_Int>>& subdomains)
   {
      const HYPRE_Int num_domains = static_cast<HYPRE_Int>(subdomains.size());

      HYPRE_Int max_domain_size = 0;
      HYPRE_Int total_memberships = 0;
      HYPRE_Int total_dense_data = 0;

      for (const auto& domain : subdomains)
      {
         const HYPRE_Int sz = static_cast<HYPRE_Int>(domain.size());
         max_domain_size = std::max(max_domain_size, sz);
         total_memberships += sz;
         total_dense_data += sz * sz;
      }

      hypre_CSRMatrix* csr =
         hypre_CSRMatrixCreate(num_domains, max_domain_size, total_memberships);

      hypre_CSRMatrixInitialize(csr);

      HYPRE_Int* I = hypre_CSRMatrixI(csr);
      HYPRE_Int* J = hypre_CSRMatrixJ(csr);

      HYPRE_Int nnz_counter = 0;
      I[0] = 0;

      for (HYPRE_Int i = 0; i < num_domains; i++)
      {
         const auto& domain = subdomains[i];
         for (HYPRE_Int k = 0; k < static_cast<HYPRE_Int>(domain.size()); k++)
         {
            J[nnz_counter++] = domain[k];
         }
         I[i + 1] = nnz_counter;
      }

      HYPRE_Real* data =
         hypre_CTAlloc(HYPRE_Real, total_dense_data, HYPRE_MEMORY_HOST);

      for (HYPRE_Int i = 0; i < total_dense_data; i++)
      {
         data[i] = 0.0;
      }

      hypre_CSRMatrixData(csr) = data;

      return (HYPRE_CSRMatrix) csr;
   }

   HYPRE_CSRMatrix domain_structure = nullptr;
   HYPRE_Int *pivots = nullptr;
   HYPRE_Real *scale = nullptr;

   HYPRE_Real LookupLocalEntry(HYPRE_BigInt gi, HYPRE_BigInt gj,
                               HYPRE_BigInt local_row_start,
                               HYPRE_BigInt local_row_end,
                               HYPRE_Int *Ai, HYPRE_Int *Aj, HYPRE_Real *Aa,
                               HYPRE_Int *Ai_offd, HYPRE_Int *Aj_offd,
                               HYPRE_Real *Aa_offd,
                               HYPRE_BigInt *col_map_offd) const
   {
      if (gi < local_row_start || gi >= local_row_end)
      {
         return 0.0;
      }

      const HYPRE_Int local_i = static_cast<HYPRE_Int>(gi - local_row_start);
      if (gj >= local_row_start && gj < local_row_end)
      {
         const HYPRE_Int local_j = static_cast<HYPRE_Int>(gj - local_row_start);
         for (HYPRE_Int jj = Ai[local_i]; jj < Ai[local_i + 1]; ++jj)
         {
            if (Aj[jj] == local_j)
            {
               return Aa[jj];
            }
         }
         return 0.0;
      }

      for (HYPRE_Int jj = Ai_offd[local_i]; jj < Ai_offd[local_i + 1]; ++jj)
      {
         if (col_map_offd[Aj_offd[jj]] == gj)
         {
            return Aa_offd[jj];
         }
      }
      return 0.0;
   }

   void BuildCustomScale(const std::vector<std::vector<HYPRE_BigInt>> &subdomains,
                         MPI_Comm comm,
                         HYPRE_Real relax_weight,
                         bool unweighted,
                         HYPRE_Real uniform_weight)
   {
      custom_scale.clear();

      if (uniform_weight >= 0.0)
      {
         for (const auto &subdomain : subdomains)
         {
            for (HYPRE_BigInt gdof : subdomain)
            {
               custom_scale[gdof] = uniform_weight;
            }
         }
         return;
      }

      if (unweighted)
      {
         for (const auto &subdomain : subdomains)
         {
            for (HYPRE_BigInt gdof : subdomain)
            {
               custom_scale[gdof] = 1.0;
            }
         }
         return;
      }

      int nranks = 1;
      MPI_Comm_size(comm, &nranks);
      int local_memberships = 0;
      for (const auto &subdomain : subdomains)
      {
         local_memberships += static_cast<int>(subdomain.size());
      }

      std::vector<int> membership_counts(nranks, 0);
      MPI_Allgather(&local_memberships, 1, MPI_INT,
                    membership_counts.data(), 1, MPI_INT, comm);

      std::vector<int> membership_displs(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         membership_displs[r + 1] = membership_displs[r] + membership_counts[r];
      }

      std::vector<HYPRE_BigInt> local_flat_subdomains;
      local_flat_subdomains.reserve(local_memberships);
      for (const auto &subdomain : subdomains)
      {
         local_flat_subdomains.insert(local_flat_subdomains.end(),
                                      subdomain.begin(), subdomain.end());
      }

      std::vector<HYPRE_BigInt> all_flat_subdomains(membership_displs[nranks]);
      MPI_Allgatherv(local_flat_subdomains.empty() ? nullptr : local_flat_subdomains.data(),
                     local_memberships, HYPRE_MPI_BIG_INT,
                     all_flat_subdomains.empty() ? nullptr : all_flat_subdomains.data(),
                     membership_counts.data(), membership_displs.data(),
                     HYPRE_MPI_BIG_INT, comm);

      std::unordered_map<HYPRE_BigInt, int> multiplicity;
      multiplicity.reserve(all_flat_subdomains.size());
      for (HYPRE_BigInt gdof : all_flat_subdomains)
      {
         multiplicity[gdof]++;
      }

      custom_scale.reserve(multiplicity.size());
      for (const auto &entry : multiplicity)
      {
         custom_scale[entry.first] = relax_weight / entry.second;
      }
   }

   HYPRE_Int ApplyCustomSolve(HYPRE_ParCSRMatrix A,
                              HYPRE_ParVector b,
                              HYPRE_ParVector x) const
   {
      (void)A;

      hypre_Vector *b_local = hypre_ParVectorLocalVector((hypre_ParVector *)b);
      hypre_Vector *x_local = hypre_ParVectorLocalVector((hypre_ParVector *)x);
      HYPRE_Real *b_data = hypre_VectorData(b_local);
      HYPRE_Real *x_data = hypre_VectorData(x_local);

      std::vector<HYPRE_Real> rhs_response_values(rhs_request_recv_ids.size(), 0.0);
      for (size_t i = 0; i < rhs_request_recv_ids.size(); ++i)
      {
         const HYPRE_BigInt gdof = rhs_request_recv_ids[i];
         MFEM_VERIFY(gdof >= custom_local_row_start &&
                     gdof < custom_local_row_start + custom_local_num_rows,
                     "Requested RHS DOF is not locally owned.");
         rhs_response_values[i] = b_data[gdof - custom_local_row_start];
      }

      std::vector<HYPRE_Real> remote_rhs_values(rhs_request_send_ids.size(), 0.0);
      MPI_Alltoallv(rhs_response_values.empty() ? nullptr : rhs_response_values.data(),
                    rhs_request_recv_counts.data(), rhs_request_recv_displs.data(),
                    HYPRE_MPI_REAL,
                    remote_rhs_values.empty() ? nullptr : remote_rhs_values.data(),
                    rhs_request_send_counts.data(), rhs_request_send_displs.data(),
                    HYPRE_MPI_REAL, custom_comm);

      std::vector<HYPRE_Real> local_correction(custom_local_num_rows, 0.0);
      std::unordered_map<HYPRE_BigInt, HYPRE_Real> remote_correction;
      remote_correction.reserve(correction_send_ids.size());
      char uplo = 'L';
      const char trans = 'N';
      int nrhs = 1;

      for (size_t d = 0; d < custom_subdomains.size(); ++d)
      {
         const int n = static_cast<int>(custom_subdomains[d].size());
         if (n == 0) { continue; }

         std::vector<HYPRE_Real> rhs(n, 0.0);
         for (int i = 0; i < n; ++i)
         {
            const HYPRE_BigInt gdof = custom_subdomains[d][i];
            if (gdof >= custom_local_row_start &&
                gdof < custom_local_row_start + custom_local_num_rows)
            {
               rhs[i] = b_data[gdof - custom_local_row_start];
            }
            else
            {
               auto it = rhs_value_index.find(gdof);
               MFEM_VERIFY(it != rhs_value_index.end(),
                           "Missing remote RHS value for Schwarz subdomain DOF.");
               rhs[i] = remote_rhs_values[it->second];
            }
         }

         int info = 0;
         int nn = n;
         int ldb = n;
         HYPRE_Real *factor = const_cast<HYPRE_Real *>(&custom_dense_data[custom_dense_offsets[d]]);
         if (custom_use_nonsymm)
         {
            HYPRE_Int *pivot = const_cast<HYPRE_Int *>(&custom_pivots[custom_pivot_offsets[d]]);
            dgetrs_(&trans, &nn, &nrhs, factor, &nn,
                    reinterpret_cast<int *>(pivot), rhs.data(), &ldb, &info);
            MFEM_VERIFY(info == 0, "LU backsolve failed for Schwarz subdomain.");
         }
         else
         {
            dpotrs_(&uplo, &nn, &nrhs, factor, &nn,
                    rhs.data(), &ldb, &info);
            MFEM_VERIFY(info == 0, "Cholesky backsolve failed for Schwarz subdomain.");
         }

         for (int i = 0; i < n; ++i)
         {
            const HYPRE_BigInt gdof = custom_subdomains[d][i];
            const auto it = custom_scale.find(gdof);
            const HYPRE_Real dof_scale = (it != custom_scale.end()) ? it->second : 1.0;
            const HYPRE_Real contribution = dof_scale * rhs[i];
            if (gdof >= custom_local_row_start &&
                gdof < custom_local_row_start + custom_local_num_rows)
            {
               local_correction[gdof - custom_local_row_start] += contribution;
            }
            else
            {
               remote_correction[gdof] += contribution;
            }
         }
      }

      std::vector<HYPRE_Real> correction_send_values(correction_send_ids.size(), 0.0);
      for (size_t i = 0; i < correction_send_ids.size(); ++i)
      {
         auto it = remote_correction.find(correction_send_ids[i]);
         if (it != remote_correction.end())
         {
            correction_send_values[i] = it->second;
         }
      }

      std::vector<HYPRE_Real> correction_recv_values(correction_recv_ids.size(), 0.0);
      MPI_Alltoallv(correction_send_values.empty() ? nullptr : correction_send_values.data(),
                    correction_send_counts.data(), correction_send_displs.data(),
                    HYPRE_MPI_REAL,
                    correction_recv_values.empty() ? nullptr : correction_recv_values.data(),
                    correction_recv_counts.data(), correction_recv_displs.data(),
                    HYPRE_MPI_REAL, custom_comm);

      for (HYPRE_Int i = 0; i < custom_local_num_rows; ++i)
      {
         x_data[i] = local_correction[i];
      }

      for (size_t i = 0; i < correction_recv_ids.size(); ++i)
      {
         const HYPRE_BigInt gdof = correction_recv_ids[i];
         MFEM_VERIFY(gdof >= custom_local_row_start &&
                     gdof < custom_local_row_start + custom_local_num_rows,
                     "Received Schwarz correction for non-local DOF.");
         x_data[gdof - custom_local_row_start] += correction_recv_values[i];
      }

      return 0;
   }

   void SetCustomSubdomains(const std::vector<std::vector<HYPRE_BigInt>> &subdomains,
                            HypreParMatrix &A,
                            HYPRE_Real relax_weight = 1.0,
                            HYPRE_Int use_nonsymm = 0,
                            bool unweighted = false,
                            HYPRE_Real uniform_weight = -1.0)
   {
      hypre_SchwarzData *sd = (hypre_SchwarzData *)schwarz_solver;
      MFEM_VERIFY(sd != nullptr, "Invalid Schwarz solver.");
      if (sd->variant != RequiredVariant)
      {
         int myrank = 0;
         MPI_Comm_rank(hypre_ParCSRMatrixComm((hypre_ParCSRMatrix *)A), &myrank);
         if (myrank == 0)
         {
            mfem::out << "HypreSchwarz: overriding Schwarz variant " << sd->variant
                      << " with required variant " << RequiredVariant << std::endl;
         }
         HYPRE_SchwarzSetVariant(schwarz_solver, RequiredVariant);
      }
      sd->relax_weight = relax_weight;
      sd->use_nonsymm = use_nonsymm;

      hypre_ParCSRMatrix *parA = (hypre_ParCSRMatrix *)A;
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(parA);
      MFEM_VERIFY(A_diag != nullptr, "A_diag is null.");

      HYPRE_Int num_dofs = hypre_CSRMatrixNumRows(A_diag);

      // STEP 4a: Add MPI context and gather row partition
      MPI_Comm comm = hypre_ParCSRMatrixComm(parA);
      int myrank, nranks;
      MPI_Comm_rank(comm, &myrank);
      MPI_Comm_size(comm, &nranks);

      // Get row partition for all ranks
      // With assumed partition: row_starts is size 2 [local_start, local_end]
      // Without: row_starts is size nranks+1 with all boundaries
      const HYPRE_BigInt *row_starts = hypre_ParCSRMatrixRowStarts(parA);
      const HYPRE_BigInt local_row_start = row_starts[0];
      const HYPRE_BigInt local_row_end = row_starts[1];
      const HYPRE_Int local_num_rows = local_row_end - local_row_start;

      MFEM_VERIFY(num_dofs == local_num_rows,
                  "A_diag size does not match local row count");

      // Gather all row partitions
      std::vector<HYPRE_BigInt> all_row_starts(nranks);
      std::vector<HYPRE_BigInt> all_row_ends(nranks);
      MPI_Allgather(&local_row_start, 1, HYPRE_MPI_BIG_INT,
                    all_row_starts.data(), 1, HYPRE_MPI_BIG_INT, comm);
      MPI_Allgather(&local_row_end, 1, HYPRE_MPI_BIG_INT,
                    all_row_ends.data(), 1, HYPRE_MPI_BIG_INT, comm);

      // Access diagonal and off-diagonal blocks
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(parA);
      HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(parA);
      HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

      HYPRE_Int *Ai_offd = hypre_CSRMatrixI(A_offd);
      HYPRE_Int *Aj_offd = hypre_CSRMatrixJ(A_offd);
      HYPRE_Real *Aa_offd = hypre_CSRMatrixData(A_offd);

      // Build mapping from global PTAP index to extended local index
      // Extended indexing: [0, num_dofs) = local, [num_dofs, num_dofs+num_cols_offd) = ghost
      std::map<HYPRE_BigInt, HYPRE_Int> global_to_extended_local;

      // Map local DOFs
      for (HYPRE_Int i = 0; i < num_dofs; ++i) {
         global_to_extended_local[local_row_start + i] = i;
      }

      // Map ghost DOFs (from A_offd's column map)
      for (HYPRE_Int j = 0; j < num_cols_offd; ++j) {
         global_to_extended_local[col_map_offd[j]] = num_dofs + j;
      }

      // STEP 4a VERIFICATION
      if (myrank == 0 && print_level > 1)
      {
         mfem::out << "SetCustomSubdomains: Processing " << subdomains.size()
                   << " local subdomains, " << num_dofs << " local DOFs, "
                   << num_cols_offd << " ghost DOFs" << std::endl;
      }

      if (domain_structure)
      {
         hypre_CSRMatrixDestroy((hypre_CSRMatrix *)domain_structure);
         domain_structure = nullptr;
      }
      if (pivots)
      {
         hypre_TFree(pivots, HYPRE_MEMORY_HOST);
         pivots = nullptr;
      }
      if (scale)
      {
         hypre_TFree(scale, HYPRE_MEMORY_HOST);
         scale = nullptr;
      }

      if (sd->Vtemp)
      {
         hypre_ParVectorDestroy(sd->Vtemp);
         sd->Vtemp = nullptr;
      }

      if (sd->A_boundary)
      {
         hypre_CSRMatrixDestroy(sd->A_boundary);
         sd->A_boundary = nullptr;
      }

      {
         hypre_ParVector *Vtemp =
            hypre_ParVectorCreate(hypre_ParCSRMatrixComm(parA),
                                  hypre_ParCSRMatrixGlobalNumRows(parA),
                                  hypre_ParCSRMatrixRowStarts(parA));
         hypre_ParVectorInitialize(Vtemp);
         sd->Vtemp = Vtemp;
      }

      const HYPRE_Int num_domains = (HYPRE_Int)subdomains.size();
      HYPRE_Int total_pivots = 0;
      size_t total_dense_data_sz = 0;
      for (const auto &dom : subdomains)
      {
         const HYPRE_Int sz = (HYPRE_Int)dom.size();
         MFEM_VERIFY(sz >= 0, "Invalid subdomain size.");
         total_pivots += sz;
         total_dense_data_sz += (size_t)sz * (size_t)sz;
      }

      MFEM_VERIFY(total_dense_data_sz <= (size_t)std::numeric_limits<HYPRE_Int>::max(),
                  "Dense subdomain storage exceeds HYPRE_Int capacity.");

      custom_comm = comm;
      custom_local_row_start = local_row_start;
      custom_local_num_rows = local_num_rows;
      custom_use_nonsymm = (use_nonsymm != 0);
      use_custom_solve = true;
      custom_subdomains = subdomains;
      custom_dense_offsets.assign(num_domains + 1, 0);
      custom_pivot_offsets.assign(num_domains + 1, 0);
      for (HYPRE_Int d = 0; d < num_domains; ++d)
      {
         const HYPRE_Int sz = (HYPRE_Int)custom_subdomains[d].size();
         custom_dense_offsets[d + 1] = custom_dense_offsets[d] + sz * sz;
         custom_pivot_offsets[d + 1] = custom_pivot_offsets[d] + sz;
      }
      custom_dense_data.assign(custom_dense_offsets[num_domains], 0.0);
      custom_pivots.assign(custom_use_nonsymm ? custom_pivot_offsets[num_domains] : 0, 0);

      auto owner_rank = [&](HYPRE_BigInt gdof) -> int
      {
         for (int r = 0; r < nranks; ++r)
         {
            if (gdof >= all_row_starts[r] && gdof < all_row_ends[r])
            {
               return r;
            }
         }
         return -1;
      };

      std::vector<std::set<HYPRE_BigInt>> rhs_request_sets(nranks);
      std::vector<std::set<HYPRE_BigInt>> correction_send_sets(nranks);
      for (const auto &domain : custom_subdomains)
      {
         for (HYPRE_BigInt gdof : domain)
         {
            const int owner = owner_rank(gdof);
            MFEM_VERIFY(owner >= 0, "Failed to determine owner of Schwarz DOF.");
            if (owner != myrank)
            {
               rhs_request_sets[owner].insert(gdof);
               correction_send_sets[owner].insert(gdof);
            }
         }
      }

      rhs_request_send_counts.assign(nranks, 0);
      rhs_request_send_displs.assign(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         rhs_request_send_counts[r] = static_cast<int>(rhs_request_sets[r].size());
         rhs_request_send_displs[r + 1] = rhs_request_send_displs[r] + rhs_request_send_counts[r];
      }
      rhs_request_send_ids.resize(rhs_request_send_displs[nranks]);
      int rhs_idx = 0;
      for (int r = 0; r < nranks; ++r)
      {
         for (HYPRE_BigInt gdof : rhs_request_sets[r])
         {
            rhs_request_send_ids[rhs_idx++] = gdof;
         }
      }
      rhs_request_recv_counts.assign(nranks, 0);
      rhs_request_recv_displs.assign(nranks + 1, 0);
      MPI_Alltoall(rhs_request_send_counts.data(), 1, MPI_INT,
                   rhs_request_recv_counts.data(), 1, MPI_INT, comm);
      for (int r = 0; r < nranks; ++r)
      {
         rhs_request_recv_displs[r + 1] = rhs_request_recv_displs[r] + rhs_request_recv_counts[r];
      }
      rhs_request_recv_ids.resize(rhs_request_recv_displs[nranks]);
      MPI_Alltoallv(rhs_request_send_ids.empty() ? nullptr : rhs_request_send_ids.data(),
                    rhs_request_send_counts.data(), rhs_request_send_displs.data(),
                    HYPRE_MPI_BIG_INT,
                    rhs_request_recv_ids.empty() ? nullptr : rhs_request_recv_ids.data(),
                    rhs_request_recv_counts.data(), rhs_request_recv_displs.data(),
                    HYPRE_MPI_BIG_INT, comm);
      rhs_value_index.clear();
      rhs_value_index.reserve(rhs_request_send_ids.size());
      for (size_t i = 0; i < rhs_request_send_ids.size(); ++i)
      {
         rhs_value_index[rhs_request_send_ids[i]] = static_cast<HYPRE_Int>(i);
      }

      correction_send_counts.assign(nranks, 0);
      correction_send_displs.assign(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         correction_send_counts[r] = static_cast<int>(correction_send_sets[r].size());
         correction_send_displs[r + 1] =
            correction_send_displs[r] + correction_send_counts[r];
      }
      correction_send_ids.resize(correction_send_displs[nranks]);
      int corr_idx = 0;
      for (int r = 0; r < nranks; ++r)
      {
         for (HYPRE_BigInt gdof : correction_send_sets[r])
         {
            correction_send_ids[corr_idx++] = gdof;
         }
      }
      correction_recv_counts.assign(nranks, 0);
      correction_recv_displs.assign(nranks + 1, 0);
      MPI_Alltoall(correction_send_counts.data(), 1, MPI_INT,
                   correction_recv_counts.data(), 1, MPI_INT, comm);
      for (int r = 0; r < nranks; ++r)
      {
         correction_recv_displs[r + 1] =
            correction_recv_displs[r] + correction_recv_counts[r];
      }
      correction_recv_ids.resize(correction_recv_displs[nranks]);
      MPI_Alltoallv(correction_send_ids.empty() ? nullptr : correction_send_ids.data(),
                    correction_send_counts.data(), correction_send_displs.data(),
                    HYPRE_MPI_BIG_INT,
                    correction_recv_ids.empty() ? nullptr : correction_recv_ids.data(),
                    correction_recv_counts.data(), correction_recv_displs.data(),
                    HYPRE_MPI_BIG_INT, comm);

      BuildCustomScale(subdomains, comm, relax_weight, unweighted, uniform_weight);
      Instances()[schwarz_solver] = this;

      HYPRE_Int *Ai = hypre_CSRMatrixI(A_diag);
      HYPRE_Int *Aj = hypre_CSRMatrixJ(A_diag);
      HYPRE_Real *Aa = hypre_CSRMatrixData(A_diag);

      struct MatrixEntryRequest
      {
         HYPRE_BigInt row;
         HYPRE_BigInt col;
      };

      std::vector<std::vector<MatrixEntryRequest>> requests_to_send(nranks);
      for (const auto &domain : custom_subdomains)
      {
         for (HYPRE_BigInt gi : domain)
         {
            if (gi >= local_row_start && gi < local_row_end)
            {
               continue;
            }

            int owner_rank = -1;
            for (int r = 0; r < nranks; ++r)
            {
               if (gi >= all_row_starts[r] && gi < all_row_ends[r])
               {
                  owner_rank = r;
                  break;
               }
            }

            MFEM_VERIFY(owner_rank >= 0, "Failed to determine owner of remote subdomain row.");
            for (HYPRE_BigInt gj : domain)
            {
               requests_to_send[owner_rank].push_back({gi, gj});
            }
         }
      }

      std::vector<int> send_counts(nranks, 0), recv_counts(nranks, 0);
      for (int r = 0; r < nranks; ++r)
      {
         send_counts[r] = 2 * static_cast<int>(requests_to_send[r].size());
      }
      MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

      std::vector<int> send_displs(nranks + 1, 0), recv_displs(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         send_displs[r + 1] = send_displs[r] + send_counts[r];
         recv_displs[r + 1] = recv_displs[r] + recv_counts[r];
      }

      std::vector<HYPRE_BigInt> send_buffer(send_displs[nranks]);
      int idx = 0;
      for (int r = 0; r < nranks; ++r)
      {
         for (const auto &req : requests_to_send[r])
         {
            send_buffer[idx++] = req.row;
            send_buffer[idx++] = req.col;
         }
      }

      std::vector<HYPRE_BigInt> recv_buffer(recv_displs[nranks]);
      MPI_Alltoallv(send_buffer.data(), send_counts.data(), send_displs.data(),
                    HYPRE_MPI_BIG_INT, recv_buffer.data(), recv_counts.data(),
                    recv_displs.data(), HYPRE_MPI_BIG_INT, comm);

      std::vector<HYPRE_Real> response_values(recv_buffer.size() / 2, 0.0);
      for (size_t i = 0; i < response_values.size(); ++i)
      {
         response_values[i] = LookupLocalEntry(recv_buffer[2 * i], recv_buffer[2 * i + 1],
                                               local_row_start, local_row_end,
                                               Ai, Aj, Aa,
                                               Ai_offd, Aj_offd, Aa_offd,
                                               col_map_offd);
      }

      std::vector<int> send_value_counts(nranks, 0), recv_value_counts(nranks, 0);
      std::vector<int> send_value_displs(nranks + 1, 0), recv_value_displs(nranks + 1, 0);
      for (int r = 0; r < nranks; ++r)
      {
         send_value_counts[r] = recv_counts[r] / 2;
         recv_value_counts[r] = send_counts[r] / 2;
         send_value_displs[r + 1] = send_value_displs[r] + send_value_counts[r];
         recv_value_displs[r + 1] = recv_value_displs[r] + recv_value_counts[r];
      }

      std::vector<HYPRE_Real> recv_values(send_buffer.size() / 2, 0.0);
      MPI_Alltoallv(response_values.data(), send_value_counts.data(),
                    send_value_displs.data(), HYPRE_MPI_REAL,
                    recv_values.data(), recv_value_counts.data(),
                    recv_value_displs.data(), HYPRE_MPI_REAL, comm);

      std::map<std::pair<HYPRE_BigInt, HYPRE_BigInt>, HYPRE_Real> remote_entries;
      idx = 0;
      for (int r = 0; r < nranks; ++r)
      {
         for (const auto &req : requests_to_send[r])
         {
            remote_entries[{req.row, req.col}] = recv_values[idx++];
         }
      }

      char uplo = 'L';
      for (HYPRE_Int d = 0; d < num_domains; ++d)
      {
         const HYPRE_Int n = (HYPRE_Int)custom_subdomains[d].size();
         HYPRE_Real *AE = &custom_dense_data[custom_dense_offsets[d]];
         std::fill(AE, AE + (size_t)n * (size_t)n, 0.0);

         for (HYPRE_Int i = 0; i < n; ++i)
         {
            const HYPRE_BigInt gi = custom_subdomains[d][i];
            for (HYPRE_Int j = 0; j < n; ++j)
            {
               const HYPRE_BigInt gj = custom_subdomains[d][j];
               if (gi >= local_row_start && gi < local_row_end)
               {
                  AE[i + j * n] = LookupLocalEntry(gi, gj, local_row_start, local_row_end,
                                                   Ai, Aj, Aa, Ai_offd, Aj_offd,
                                                   Aa_offd, col_map_offd);
               }
               else
               {
                  auto it = remote_entries.find({gi, gj});
                  if (it != remote_entries.end())
                  {
                     AE[i + j * n] = it->second;
                  }
               }
            }
         }

         if (n > 0)
         {
            // Eigendecomposition diagnostics (before factorization)
            if (print_level > 0)
            {
               std::vector<HYPRE_Real> AE_copy(n * n);
               std::copy(AE, AE + n * n, AE_copy.begin());

               int nn = (int)n;
               int info = 0;
               bool has_negative = false;
               HYPRE_Real min_abs_eval = 1e100, max_abs_eval = -1e100;

               if (custom_use_nonsymm)
               {
                  // General eigenvalue problem
                  std::vector<HYPRE_Real> wr(n), wi(n);
                  int lwork = 4 * nn;
                  std::vector<HYPRE_Real> work(lwork);
                  char jobvl = 'N', jobvr = 'N';

                  dgeev_(&jobvl, &jobvr, &nn, AE_copy.data(), &nn,
                         wr.data(), wi.data(), nullptr, &nn, nullptr, &nn,
                         work.data(), &lwork, &info);

                  if (info == 0)
                  {
                     for (int i = 0; i < n; ++i)
                     {
                        HYPRE_Real eval_mag = std::sqrt(wr[i]*wr[i] + wi[i]*wi[i]);
                        if (eval_mag < min_abs_eval) min_abs_eval = eval_mag;
                        if (eval_mag > max_abs_eval) max_abs_eval = eval_mag;
                        if (wr[i] < 0.0 && std::abs(wi[i]) < 1e-14) has_negative = true;
                     }

                     if (myrank == 0)
                     {
                        mfem::out << "Subdomain " << d << " (size=" << n << "): ";
                        if (has_negative) mfem::out << "HAS NEGATIVE EIGENVALUES, ";
                        mfem::out << "min|eval|=" << min_abs_eval
                                  << ", max|eval|=" << max_abs_eval << std::endl;
                     }
                  }
               }
               else
               {
                  // Symmetric eigenvalue problem
                  std::vector<HYPRE_Real> w(n);
                  int lwork = 3 * nn;
                  std::vector<HYPRE_Real> work(lwork);
                  char jobz = 'N';

                  dsyev_(&jobz, &uplo, &nn, AE_copy.data(), &nn, w.data(),
                         work.data(), &lwork, &info);

                  if (info == 0)
                  {
                     for (int i = 0; i < n; ++i)
                     {
                        HYPRE_Real eval_abs = std::abs(w[i]);
                        if (eval_abs < min_abs_eval) min_abs_eval = eval_abs;
                        if (eval_abs > max_abs_eval) max_abs_eval = eval_abs;
                        if (w[i] < 0.0) has_negative = true;
                     }

                     if (myrank == 0)
                     {
                        mfem::out << "Subdomain " << d << " (size=" << n << "): ";
                        if (has_negative) mfem::out << "HAS NEGATIVE EIGENVALUES, ";
                        mfem::out << "min|eval|=" << min_abs_eval
                                  << ", max|eval|=" << max_abs_eval << std::endl;
                     }
                  }
               }
            }

            int info = 0;
            int nn = (int)n;
            if (custom_use_nonsymm)
            {
               dgetrf_(&nn, &nn, AE, &nn,
                       reinterpret_cast<int *>(&custom_pivots[custom_pivot_offsets[d]]), &info);
               MFEM_VERIFY(info == 0, "LU factorization failed for Schwarz subdomain.");
            }
            else
            {
               dpotrf_(&uplo, &nn, AE, &nn, &info);
               MFEM_VERIFY(info == 0, "Cholesky factorization failed for Schwarz subdomain.");
            }
         }
      }

      sd->domain_structure = nullptr;
      sd->pivots = nullptr;
      sd->scale = nullptr;
      sd->A_boundary = nullptr;
   }

   operator HYPRE_Solver() const override { return schwarz_solver; }
   HYPRE_PtrToParSolverFcn SetupFcn() const override
   { return (HYPRE_PtrToParSolverFcn) DummyParSolverFcn; }
   HYPRE_PtrToParSolverFcn SolveFcn() const override
   {
      static auto wrapped_solve = [](HYPRE_Solver solver, HYPRE_ParCSRMatrix A,
                                      HYPRE_ParVector b, HYPRE_ParVector x) -> HYPRE_Int
      {
         HypreSchwarz *instance =
            Instances().count(solver) ? Instances()[solver] : nullptr;
         const int print_level =
            PrintLevels().count(solver) ? PrintLevels()[solver] : 0;
         const double solve_start = MPI_Wtime();
         HYPRE_Int result =
            (instance && instance->use_custom_solve) ?
            instance->ApplyCustomSolve(A, b, x) :
            HYPRE_SchwarzSolve(solver, A, b, x);
         const double solve_elapsed_local = MPI_Wtime() - solve_start;
         double solve_elapsed = 0.0;
         MPI_Comm comm = hypre_ParCSRMatrixComm((hypre_ParCSRMatrix*)A);
         MPI_Allreduce(&solve_elapsed_local, &solve_elapsed, 1, MPI_DOUBLE, MPI_MAX, comm);
         int myrank = 0;
         MPI_Comm_rank(comm, &myrank);
         if (myrank == 0 && print_level > 0)
         {
            mfem::out << "HypreSchwarz solve time [s]: " << solve_elapsed << std::endl;
         }

         // Compute residual: r = b - A*x
         hypre_ParVector *residual = hypre_ParVectorCreate(
            hypre_ParCSRMatrixComm((hypre_ParCSRMatrix*)A),
            hypre_ParCSRMatrixGlobalNumRows((hypre_ParCSRMatrix*)A),
            hypre_ParCSRMatrixRowStarts((hypre_ParCSRMatrix*)A));
         hypre_ParVectorInitialize(residual);
         hypre_ParVectorCopy((hypre_ParVector*)b, residual);
         hypre_ParCSRMatrixMatvec(-1.0, (hypre_ParCSRMatrix*)A,
                                  (hypre_ParVector*)x, 1.0, residual);

         HYPRE_Real residual_norm = hypre_ParVectorInnerProd(residual, residual);
         residual_norm = std::sqrt(residual_norm);

         HYPRE_Real b_norm = hypre_ParVectorInnerProd((hypre_ParVector*)b, (hypre_ParVector*)b);
         b_norm = std::sqrt(b_norm);

         if (myrank == 0 && print_level > 1)
         {
            mfem::out << "HypreSchwarz residual norm: " << residual_norm
                      << ", RHS norm: " << b_norm << std::endl;
         }

         hypre_ParVectorDestroy(residual);
         return result;
      };
      return (HYPRE_PtrToParSolverFcn) +wrapped_solve;
   }
   using HypreSolver::Mult;
};

/// Wrapper around HypreBoomerAMG with custom smoother combining Schwarz and Gauss-Seidel
class HypreBoomerAMGWithSchwarzSmoother : public HypreSolver
{
private:
   HypreBoomerAMG *amg;
   HypreSchwarz *schwarz_smoother;
   HypreSmoother *gs_smoother;
   Solver *contact_solver;  // Either Schwarz or direct solver
   bool owns_amg;
   bool use_direct_contact_solver;

   // Track which DoFs are touched by Schwarz
   std::set<HYPRE_BigInt> schwarz_dofs;

   // Temporary vectors for smoothing
   mutable HypreParVector *temp_residual;
   mutable HypreParVector *temp_correction;

   // Modified matrix with contact DoF rows/columns zeroed
   HypreParMatrix *masked_matrix;

   // Option to skip zeroing contact DoFs in the masked matrix
   bool zero_contact_dofs;

   // Transfer operator for direct contact solver (not owned)
   HypreParMatrix *transfer_operator;  // Transfer operator P mapping contact subspace to full space

public:
   /// Constructor taking ownership of an existing BoomerAMG instance
   HypreBoomerAMGWithSchwarzSmoother(HypreBoomerAMG *boomeramg, bool take_ownership = true,
                                      bool zero_contact_dofs_in_gs = true)
      : amg(boomeramg), schwarz_smoother(nullptr), gs_smoother(nullptr),
        contact_solver(nullptr), owns_amg(take_ownership), use_direct_contact_solver(false),
        temp_residual(nullptr), temp_correction(nullptr),
        masked_matrix(nullptr), zero_contact_dofs(zero_contact_dofs_in_gs),
        transfer_operator(nullptr)
   {
      MFEM_VERIFY(amg, "BoomerAMG pointer cannot be null");

      // Note: We disable finest level sweeps in SetOperator (after amg->SetOperator is called)
      // Cannot call HYPRE_BoomerAMGSetNumGridSweeps here in constructor before operator is set
   }

   /// Set the Schwarz smoother and the DoFs it will operate on
   /// Pass schwarz=nullptr when using direct solver (DoFs are still needed for masking)
   /// NOTE: This only stores the smoother and DoFs. The operator will be set
   /// when SetOperator is called on the wrapper.
   void SetSchwarzSmoother(HypreSchwarz *schwarz,
                           const std::vector<std::vector<HYPRE_BigInt>> &subdomains)
   {
      schwarz_smoother = schwarz;
      if (schwarz)
      {
         contact_solver = schwarz;
         use_direct_contact_solver = false;
      }

      // Build set of all DoFs touched by contact solver (Schwarz or direct)
      schwarz_dofs.clear();
      for (const auto &subdomain : subdomains)
      {
         for (HYPRE_BigInt dof : subdomain)
         {
            schwarz_dofs.insert(dof);
         }
      }

      // Note: We don't call schwarz_smoother->SetOperator here because A might be stale.
      // Instead, SetOperator on the wrapper will configure schwarz_smoother after updating A.
   }

   /// Set a direct solver for the contact subspace
   /// @param direct_solver The solver for the contact subspace (not owned)
   /// @param P_transfer Transfer operator P mapping contact subspace to full space (not owned)
   /// @param PTAP_submatrix The projected operator P^T A P (not owned)
   /// The direct solver's operator must be set externally to PTAP_submatrix
   void SetDirectContactSolver(Solver *direct_solver,
                               const HypreParMatrix *P_transfer,
                               const HypreParMatrix *PTAP_submatrix)
   {
      MFEM_VERIFY(direct_solver, "Direct solver cannot be null");
      MFEM_VERIFY(P_transfer, "Transfer operator cannot be null");

      contact_solver = direct_solver;
      use_direct_contact_solver = true;
      schwarz_smoother = nullptr;
      transfer_operator = const_cast<HypreParMatrix*>(P_transfer);

      // Set the PTAP operator on the direct solver
      // PTAP changes every iteration, so we must call SetOperator each time
      if (PTAP_submatrix)
      {
         contact_solver->SetOperator(*PTAP_submatrix);
      }

      // Extract contact DoFs from transfer operator structure
      ExtractContactDoFsFromTransfer(P_transfer);

      // If the full-space operator has already been set, configure masked matrix
      if (A)
      {
         SetupMaskedMatrix();
      }
   }


   void SetOperator(const Operator &op) override
   {
      const HypreParMatrix *new_A = dynamic_cast<const HypreParMatrix *>(&op);
      MFEM_VERIFY(new_A, "new Operator must be a HypreParMatrix!");

      // Update base class members
      height = new_A->Height();
      width  = new_A->Width();
      A = const_cast<HypreParMatrix *>(new_A);
      setup_called = 0;
      delete X;
      delete B;
      B = X = NULL;
      auxB.Delete(); auxB.Reset();
      auxX.Delete(); auxX.Reset();

      // Update AMG operator
      amg->SetOperator(op);

      // Disable finest level sweeps - the hybrid smoother handles the finest level
      //HYPRE_BoomerAMGSetNumSweeps(*amg, 0);
      //HYPRE_BoomerAMGSetCycleNumSweeps(*amg, 1, 1);
      //HYPRE_BoomerAMGSetCycleNumSweeps(*amg, 1, 2);
      //HYPRE_BoomerAMGSetCycleNumSweeps(*amg, 1, 3);

      //HYPRE_Int *sweeps = (HYPRE_Int*) malloc(4 * sizeof(*sweeps));
      //sweeps[0] = 0;
      //sweeps[1] = 1;
      //sweeps[2] = 1;
      //sweeps[3] = 1;
      //HYPRE_BoomerAMGSetNumGridSweeps(*amg, sweeps);
      HYPRE_BoomerAMGSetLevelRelaxWt(*amg, 0.0, 0);

      // Update contact solver operator if it has been configured
      if (schwarz_smoother)
      {
         // Set operator on the Schwarz smoother
         schwarz_smoother->SetOperator(*new_A);
         SetupMaskedMatrix();
      }
      else if (use_direct_contact_solver)
      {
         // For direct solver, just set up masked matrix and GS smoother
         // The direct solver operator (P^T A P) must be set externally
         SetupMaskedMatrix();
      }
   }

   void Mult(const HypreParVector &b, HypreParVector &x) const override
   {
      MFEM_VERIFY(A, "Operator must be set before calling Mult");
      MFEM_VERIFY(contact_solver, "Contact solver must be set (either Schwarz or direct)");
      MFEM_VERIFY(gs_smoother, "GS smoother must be initialized");
      if (use_direct_contact_solver)
      {
         MFEM_VERIFY(transfer_operator, "Transfer operator must be set for direct contact solver");
      }

      // Initialize x to zero if not in iterative mode
      if (!iterative_mode)
      {
         x.HypreWrite();
         hypre_ParVectorSetConstantValues(x, 0.0);
      }

      // Apply hybrid contact/GS pre-smoothing
      ApplyHybridSmoother(b, x);

      // Clear HYPRE's global error flag in case ParallelDirectSolver corrupted it
      hypre_error_flag = 0;

      // Apply BoomerAMG (with disabled finest level sweeps)
      // AMG's Mult will call Setup automatically if needed
      amg->Mult(b, x);

      // Apply hybrid contact/GS post-smoothing
      ApplyHybridSmoother(b, x);
   }

private:
   /// Extract contact DoFs from the transfer operator P
   /// P is a matrix of size (num_displacement_dofs × num_contact_dofs)
   /// where each column corresponds to a contact DoF in the full space
   void ExtractContactDoFsFromTransfer(const HypreParMatrix *P_transfer)
   {
      schwarz_dofs.clear();

      if (!P_transfer) return;

      hypre_ParCSRMatrix *parP = (hypre_ParCSRMatrix *)(*P_transfer);
      hypre_CSRMatrix *P_diag = hypre_ParCSRMatrixDiag(parP);
      hypre_CSRMatrix *P_offd = hypre_ParCSRMatrixOffd(parP);

      HYPRE_BigInt local_row_start = hypre_ParCSRMatrixFirstRowIndex(parP);
      HYPRE_Int local_num_rows = hypre_CSRMatrixNumRows(P_diag);
      HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(parP);

      HYPRE_Int *P_diag_i = hypre_CSRMatrixI(P_diag);
      HYPRE_Int *P_diag_j = hypre_CSRMatrixJ(P_diag);
      HYPRE_Int *P_offd_i = hypre_CSRMatrixI(P_offd);
      HYPRE_Int *P_offd_j = hypre_CSRMatrixJ(P_offd);

      // Collect all DoFs that have nonzero entries in P (these are contact DoFs)
      for (HYPRE_Int i = 0; i < local_num_rows; ++i)
      {
         HYPRE_BigInt global_row = local_row_start + i;
         bool has_contact_entry = false;

         // Check diagonal block
         for (HYPRE_Int jj = P_diag_i[i]; jj < P_diag_i[i + 1]; ++jj)
         {
            if (std::abs(hypre_CSRMatrixData(P_diag)[jj]) > 1e-14)
            {
               has_contact_entry = true;
               break;
            }
         }

         // Check off-diagonal block if needed
         if (!has_contact_entry)
         {
            for (HYPRE_Int jj = P_offd_i[i]; jj < P_offd_i[i + 1]; ++jj)
            {
               if (std::abs(hypre_CSRMatrixData(P_offd)[jj]) > 1e-14)
               {
                  has_contact_entry = true;
                  break;
               }
            }
         }

         if (has_contact_entry)
         {
            schwarz_dofs.insert(global_row);
         }
      }
   }

   /// Setup masked matrix and temporary vectors
   void SetupMaskedMatrix()
   {
      MFEM_VERIFY(A, "Operator must be set before setting up masked matrix");

      if (masked_matrix)
      {
         delete masked_matrix;
      }
      masked_matrix = new HypreParMatrix(*A);
      if (zero_contact_dofs)
      {
         ZeroContactDoFsInMatrix(*masked_matrix);
      }

      if (gs_smoother)
      {
         delete gs_smoother;
      }
      gs_smoother = new HypreSmoother(*masked_matrix, HypreSmoother::l1GS, 1);

      if (temp_residual)
      {
         delete temp_residual;
         delete temp_correction;
      }
      temp_residual = new HypreParVector(*A, 0);
      temp_correction = new HypreParVector(*A, 0);
   }

   /// Zero out all matrix entries in rows or columns corresponding to Schwarz DoFs
   /// Set diagonal to 1.0 for contact DoF rows
   void ZeroContactDoFsInMatrix(HypreParMatrix &mat)
   {
      hypre_ParCSRMatrix *parA = (hypre_ParCSRMatrix *)mat;
      hypre_CSRMatrix *A_diag = hypre_ParCSRMatrixDiag(parA);
      hypre_CSRMatrix *A_offd = hypre_ParCSRMatrixOffd(parA);

      HYPRE_BigInt local_row_start = hypre_ParCSRMatrixFirstRowIndex(parA);
      HYPRE_Int local_num_rows = hypre_CSRMatrixNumRows(A_diag);

      HYPRE_Int *diag_i = hypre_CSRMatrixI(A_diag);
      HYPRE_Int *diag_j = hypre_CSRMatrixJ(A_diag);
      HYPRE_Real *diag_data = hypre_CSRMatrixData(A_diag);

      HYPRE_Int *offd_i = hypre_CSRMatrixI(A_offd);
      HYPRE_Int *offd_j = hypre_CSRMatrixJ(A_offd);
      HYPRE_Real *offd_data = hypre_CSRMatrixData(A_offd);
      HYPRE_BigInt *col_map_offd = hypre_ParCSRMatrixColMapOffd(parA);
      HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

      // Build set of local Schwarz column indices
      std::set<HYPRE_Int> local_schwarz_cols;
      for (HYPRE_BigInt gdof : schwarz_dofs)
      {
         if (gdof >= local_row_start && gdof < local_row_start + local_num_rows)
         {
            local_schwarz_cols.insert(static_cast<HYPRE_Int>(gdof - local_row_start));
         }
      }

      // Build set of offd Schwarz column indices
      std::set<HYPRE_Int> offd_schwarz_cols;
      for (HYPRE_Int j = 0; j < num_cols_offd; ++j)
      {
         if (schwarz_dofs.count(col_map_offd[j]) > 0)
         {
            offd_schwarz_cols.insert(j);
         }
      }

      // Zero out rows and columns in diagonal block, set diagonal to 1.0 for contact DoFs
      for (HYPRE_Int i = 0; i < local_num_rows; ++i)
      {
         HYPRE_BigInt global_row = local_row_start + i;
         bool is_schwarz_row = schwarz_dofs.count(global_row) > 0;

         for (HYPRE_Int jj = diag_i[i]; jj < diag_i[i + 1]; ++jj)
         {
            HYPRE_Int j = diag_j[jj];
            bool is_schwarz_col = local_schwarz_cols.count(j) > 0;

            if (is_schwarz_row)
            {
               // Contact row: zero all entries except diagonal which is set to 1.0
               if (i == j)
               {
                  diag_data[jj] = 1.0;
               }
               else
               {
                  diag_data[jj] = 0.0;
               }
            }
            else if (is_schwarz_col)
            {
               // Non-contact row but contact column: zero the entry
               diag_data[jj] = 0.0;
            }
         }
      }

      // Zero out rows and columns in off-diagonal block
      for (HYPRE_Int i = 0; i < local_num_rows; ++i)
      {
         HYPRE_BigInt global_row = local_row_start + i;
         bool is_schwarz_row = schwarz_dofs.count(global_row) > 0;

         for (HYPRE_Int jj = offd_i[i]; jj < offd_i[i + 1]; ++jj)
         {
            HYPRE_Int j = offd_j[jj];
            bool is_schwarz_col = offd_schwarz_cols.count(j) > 0;

            if (is_schwarz_row)
            {
               // Contact row: zero all off-diagonal block entries
               offd_data[jj] = 0.0;
            }
            else if (is_schwarz_col)
            {
               // Non-contact row but contact column: zero the entry
               offd_data[jj] = 0.0;
            }
         }
      }
   }

   /// Apply hybrid smoother: solve A*e = r for residual r = b - A*x, then update x += e
   void ApplyHybridSmoother(const HypreParVector &b, HypreParVector &x) const
   {
      MFEM_VERIFY(A, "Operator must be set");
      MFEM_VERIFY(gs_smoother, "GS smoother must be initialized");

      hypre_ParCSRMatrix *parA = (hypre_ParCSRMatrix *)*A;
      HYPRE_BigInt local_row_start = hypre_ParCSRMatrixFirstRowIndex(parA);
      hypre_ParVector *par_x = (hypre_ParVector *)x;
      hypre_Vector *local_x = hypre_ParVectorLocalVector(par_x);
      HYPRE_Real *x_data = hypre_VectorData(local_x);
      HYPRE_Int local_size = hypre_VectorSize(local_x);

      // Compute residual: r = b - A*x
      HypreParVector residual(b);
      A->Mult(-1.0, x, 1.0, residual);  // residual = b - A*x

      // Get local residual data
      hypre_ParVector *par_r = (hypre_ParVector *)residual;
      hypre_Vector *local_r = hypre_ParVectorLocalVector(par_r);
      HYPRE_Real *r_data = hypre_VectorData(local_r);

      // Prepare masked residual for GS smoother if zero_contact_dofs is true
      HypreParVector gs_residual(residual);
      HypreParVector gs_error(x);
      gs_error = 0.0;

      if (zero_contact_dofs)
      {
         // Zero out contact DoFs in residual for GS
         hypre_ParVector *par_gs_r = (hypre_ParVector *)gs_residual;
         hypre_Vector *local_gs_r = hypre_ParVectorLocalVector(par_gs_r);
         HYPRE_Real *gs_r_data = hypre_VectorData(local_gs_r);

         for (HYPRE_Int i = 0; i < local_size; ++i)
         {
            HYPRE_BigInt global_dof = local_row_start + i;
            if (schwarz_dofs.count(global_dof) > 0)
            {
               gs_r_data[i] = 0.0;
            }
         }
      }

      // Step 1: Apply GS smoother to solve for error in non-contact DoFs
      gs_smoother->Mult(gs_residual, gs_error);

      // Update x with GS error
      hypre_ParVector *par_gs_error = (hypre_ParVector *)gs_error;
      hypre_Vector *local_gs_error = hypre_ParVectorLocalVector(par_gs_error);
      HYPRE_Real *gs_error_data = hypre_VectorData(local_gs_error);

      for (HYPRE_Int i = 0; i < local_size; ++i)
      {
         x_data[i] += gs_error_data[i];
      }

      // Step 2: Apply contact solver to compute and add contact correction
      if (contact_solver)
      {
         if (use_direct_contact_solver)
         {
            MFEM_VERIFY(transfer_operator, "Transfer operator must be set for direct contact solver");

            // For direct solver: extract subspace residual, solve, scatter correction back
            HypreParVector residual_contact(*transfer_operator, 0);  // Domain of P = contact space
            HypreParVector error_contact(*transfer_operator, 0);     // Domain of P = contact space
            HypreParVector error_full(*transfer_operator, 1);        // Range of P = full space
            residual_contact = 0.0;
            error_contact = 0.0;
            error_full = 0.0;

            // Project residual to contact subspace: residual_contact = P^T * r
            transfer_operator->MultTranspose(residual, residual_contact);

            // Solve in contact subspace: error_contact = (P^T A P)^{-1} * residual_contact
            contact_solver->Mult(residual_contact, error_contact);

            // Scatter error back to full space: error_full = P * error_contact
            transfer_operator->Mult(error_contact, error_full);

            // Add contact correction to x (only affects contact DoFs)
            hypre_ParVector *par_err = (hypre_ParVector *)error_full;
            hypre_Vector *local_err = hypre_ParVectorLocalVector(par_err);
            HYPRE_Real *err_data = hypre_VectorData(local_err);

            for (HYPRE_Int i = 0; i < local_size; ++i)
            {
               HYPRE_BigInt global_dof = local_row_start + i;
               if (schwarz_dofs.count(global_dof) > 0)
               {
                  x_data[i] += err_data[i];
               }
            }
         }
         else
         {
            // For Schwarz: operates on full space directly
            HypreParVector contact_error(x);
            contact_error = 0.0;
            contact_solver->Mult(residual, contact_error);

            // Add contact correction to x (only affects contact DoFs)
            hypre_ParVector *par_err = (hypre_ParVector *)contact_error;
            hypre_Vector *local_err = hypre_ParVectorLocalVector(par_err);
            HYPRE_Real *err_data = hypre_VectorData(local_err);

            for (HYPRE_Int i = 0; i < local_size; ++i)
            {
               HYPRE_BigInt global_dof = local_row_start + i;
               if (schwarz_dofs.count(global_dof) > 0)
               {
                  x_data[i] += err_data[i];
               }
            }
         }
      }
   }

public:

   using HypreSolver::Mult;

   operator HYPRE_Solver() const override
   {
      return *amg;
   }

   HYPRE_PtrToParSolverFcn SetupFcn() const override
   {
      return amg->SetupFcn();
   }

   HYPRE_PtrToParSolverFcn SolveFcn() const override
   {
      return amg->SolveFcn();
   }

   ~HypreBoomerAMGWithSchwarzSmoother()
   {
      if (owns_amg && amg)
      {
         delete amg;
      }
      if (gs_smoother)
      {
         delete gs_smoother;
      }
      if (temp_residual)
      {
         delete temp_residual;
         delete temp_correction;
      }
      if (masked_matrix)
      {
         delete masked_matrix;
      }
      // Don't delete transfer_operator or contact_solver - not owned by us
   }
};

#endif // MFEM_SCHWARZ_SOLVER_HPP
