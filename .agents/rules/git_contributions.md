# Continuous Incremental Git Workflow & Contribution Invariants

This workspace operates under a strict incremental-push protocol designed for maximum GitHub contribution granularity, real-time activity tracking, and clean version control.

## 1. Continuous Incremental Pushes (No Mega-Batches)
- **Immediate Push Policy:** Every individual task, feature implementation, refactoring, benchmark script, asset generation, or documentation update must be staged, committed, and pushed **immediately**.
- **No Delayed Batching:** Never accumulate multi-commit backlogs (e.g., 5–25 commits) before pushing. Pushing immediately upon each atomic milestone triggers real-time GitHub webhook contribution events on the profile heatmap.

## 2. Maximum Contribution Structure & Atomic Granularity
- Deconstruct complex implementations into modular, single-responsibility commits (e.g., mathematical derivations, discrete solvers, adjoint kernels, unit tests, experiment runs, visualization updates).
- Maintain conventional commit formatting (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `exp:`, `refactor:`).
- Author identity must strictly adhere to:
  - **Name:** `Aradhya Shinde`
  - **Email:** `aradhyashinde2330@gmail.com`

## 3. Direct to `origin/main`
- All commits are pushed directly to the default `main` branch to guarantee immediate synchronization with the remote repository and immediate contribution graph registration.
