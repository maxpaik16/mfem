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

#ifndef MFEM_FILTEREDSOLVER
#define MFEM_FILTEREDSOLVER

#include "../config/config.hpp"
#include "operator.hpp"
#include "handle.hpp"
#include <memory>

namespace mfem
{
/**
* @class FilteredSolver
* @brief Base class for solvers with filtering (subspace correction).
*
* FilteredSolver is designed to augment an existing solver with an additional
* filtering step that targets small subspaces where the solver is less
* effective. The filtered subspace is defined by a transfer operator @p P,
* which maps the subspace into the full space.
*
* ### Typical usage
* 1. Call SetOperator() to define the operator @p A that acts on the full space.
* 2. Call SetSolver() to provide the underlying solver @p B for the full-space operator.
* 3. Call SetFilteredSubspaceTransferOperator() to set transfer operator @p P.
* 4. Call SetFilteredSubspaceSolver() to set the subspace solver @p S.
* 5. Use Mult() to apply the solver.
*
* ---
*
* The preconditioner applied by Mult() is
*  $$
* M = B + P S P^T (I - A B) + B (I - A P S P^T) (I- A B),
*  $$
* and the corresponding iteration matrix is
*  $$
* I - M A = (I - B A) (I - P S P^T A) (I - B A).
*  $$
*/
class FilteredSolver : public Solver
{
public:

   enum class SubspaceSelectionMethod
   {
      NONE,
      GRADIENT,
      KNEE,
      GMM,
      COST
   };

   /// Construct an empty filtered solver. Must set operator and solver before use.
   FilteredSolver() : Solver() { }

   /// Set the system operator @a A.
   virtual void SetOperator(const Operator &A) override;

   /// Set the solver @a B that operates on the full space.
   virtual void SetSolver(Solver &B);

   /// Set the transfer operator @a P from filtered subspace to the full space.
   void SetFilteredSubspaceTransferOperator(const Operator &P);

   /// Set a solver @a S that operates on the filtered subspace operator $ P^T A P $.
   void SetFilteredSubspaceSolver(Solver &S);

   /// Set print level for timing diagnostics.
   virtual void SetPrintLevel(int print_lvl) { print_level = print_lvl; }

   /// Enable or disable reversed order (subspace → AMG → subspace instead of AMG → subspace → AMG).
   void SetReversedOrder(bool reversed) { reversed_order = reversed; }

   /// Apply the filtered solver
   void Mult(const Vector &x, Vector &y) const override;

   void SelectFilteredSubspace(const SubspaceSelectionMethod);

   virtual ~FilteredSolver() override;

   FilteredSolver(const FilteredSolver&) = delete;
   FilteredSolver& operator=(const FilteredSolver&) = delete;
   FilteredSolver(FilteredSolver&&) = default;
   FilteredSolver& operator=(FilteredSolver&&) = default;

protected:
   /// System operator (not owned).
   const Operator * A = nullptr;
   /// Transfer operator (not owned).
   const Operator * P = nullptr;
   /// Base solver (not owned).
   Solver * B = nullptr;
   /// Subspace solver (not owned).
   Solver * S = nullptr;
   /// Projected operator.
   mutable std::unique_ptr<const Operator> PtAP = nullptr;
   /// Initialize work vectors.
   void InitVectors() const;
   bool mutable solver_set = false;
   SubspaceSelectionMethod selectionMethod = SubspaceSelectionMethod::NONE;
private:

   /// Build and/or return cached projected operator $ P^T A P $.
   std::unique_ptr<const Operator> GetPtAP(const Operator *Aop,
                                           const Operator *Pop) const;
   /// Finalize solver
   void MakeSolver() const;
   /// Reset timing state for the current operator/setup cycle.
   void ResetTimingData() const;
   /// Print timing data accumulated for the current operator/setup cycle.
   void FlushTimingData() const;

   // Work vectors used in Mult.
   mutable Vector z;
   mutable Vector rf;
   mutable Vector xf;
   mutable Vector r;

   /// Print level for timing diagnostics.
   int print_level = 0;
   /// If true, apply subspace → AMG → subspace instead of AMG → subspace → AMG.
   bool reversed_order = false;
   /// Setup time split [s] for the current operator/setup cycle.
   mutable double setup_total_time = 0.0;
   mutable double setup_base_solver_time = 0.0;
   mutable double setup_ptap_time = 0.0;
   mutable double setup_subspace_solver_time = 0.0;
   /// Aggregate apply time split [s] for the current operator/setup cycle.
   mutable double cycle_total_time = 0.0;
   mutable double cycle_amg_time = 0.0;
   mutable double cycle_subspace_time = 0.0;
   mutable double cycle_other_time = 0.0;
   /// Number of preconditioner applications in the current cycle.
   mutable int cycle_num_mult_calls = 0;
   /// True once setup or apply timings have been accumulated for the current cycle.
   mutable bool have_timing_data = false;

}; // mfem::FilteredSolver class


#ifdef MFEM_USE_MPI
/**
* @class AMGFSolver
* @brief AMG with Filtering: specialization of FilteredSolver.
*
* AMGFSolver is a convenience wrapper that fixes the base solver @a B of a
* FilteredSolver to HypreBoomerAMG.
* AMGF is particularly effective for constrained optimization
* problems such as frictionless contact. For more details, see:
* [AMG with Filtering: An Efficient Preconditioner for Interior Point Methods in Large-Scale Contact Mechanics Optimization](https://arxiv.org/abs/2505.18576)
*
* The internal HypreBoomerAMG instance can be accessed and configured via AMG().
*/
class AMGFSolver : public FilteredSolver
{
private:
   /// Owned HypreBoomerAMG instance.
   std::unique_ptr<HypreBoomerAMG> amg;
public:
   /// Construct AMGF solver with default HypreBoomerAMG.
   AMGFSolver() : FilteredSolver()
   {
      amg = std::make_unique<HypreBoomerAMG>();
      FilteredSolver::SetSolver(*amg);
   }

   /// Set the system operator @a A.
   virtual void SetOperator(const Operator &A) override;

   /// Access to the internal HypreBoomerAMG instance.
   HypreBoomerAMG& GetAMG() { return *amg; }

   /// Const access to the internal HypreBoomerAMG instance.
   const HypreBoomerAMG& GetAMG() const { return *amg; }

   void SetSolver(Solver &B) override
   {
      MFEM_ABORT("SetSolver is not supported in AMGFSolver. It is set to AMG by default");
   }

   /// Set the parallel transfer operator @a P for the filtered subspace.
   void SetFilteredSubspaceTransferOperator(const HypreParMatrix &Pop);

   /// Test if the preconditioner is SPD by checking symmetry and positive-definiteness
   /// with random vectors. Performs num_tests iterations, checking:
   ///   1. Symmetry: (M u, v) = (u, M v) for random vectors u and v
   ///   2. Positive-definiteness: (u, M u) > 0 for random vector u
   /// Returns true if all tests pass, false otherwise.
   /// If verbose is true, prints detailed statistics to rank 0.
   bool TestSPD(int num_tests = 20, int seed = 12345, bool verbose = true) const;

   /// Verify operator formulations and SPD properties.
   /// Tests the equivalence of different operator representations and their symmetry/positivity.
   /// Should only be called after SetOperator and solver setup.
   void VerifyOperatorFormulations(int num_tests = 10, int seed = 12345, bool verbose = true) const;

   /// Destructor.
   ~AMGFSolver() override = default;

};
#endif


} // namespace mfem

#endif
