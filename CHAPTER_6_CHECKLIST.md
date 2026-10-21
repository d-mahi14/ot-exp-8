# Chapter 6 Syllabus Coverage & Traceability Checklist
**Course**: Optimization Techniques — Assignment 8  
**Scope**: Every algorithm, technique, and experiment explicitly mandated in Section 5 and Section 6.

---

## 1. Foundations of Optimization in ML
- [ ] **ML as Optimization**: Parameters, Loss Functions, Regularization (`track_A_continuous/src/mlp_backprop.py`)
- [ ] **Continuous vs. Discrete Landscapes**: Formulation and properties comparison (`track_A_continuous/experiments/landscapes.py`)
- [ ] **Convex vs. Non-Convex Landscapes**: Trajectory analysis (`track_A_continuous/experiments/landscapes.py`)
- [ ] **Local Minima, Global Minima & Saddle Points**: Small verifiable test cases (`track_A_continuous/experiments/saddle_points.py`)
- [ ] **Role of Optimization in QML**: Mathematical formulation (`track_I_quantum_algorithms/src/qml_benchmark.py`)

---

## 2. Continuous Optimization (Track A)
- [x] **Gradient Descent (GD)** (Vanilla batch gradient descent) (`track_A_continuous/src/first_order_optimizers.py`)
- [x] **Stochastic Gradient Descent (SGD)** & **Mini-batch SGD** (`track_A_continuous/src/first_order_optimizers.py`)
- [x] **Momentum Optimizer** (`track_A_continuous/src/first_order_optimizers.py`)
- [x] **Nesterov Accelerated Gradient (NAG)** (`track_A_continuous/src/first_order_optimizers.py`)
- [x] **AdaGrad** (`track_A_continuous/src/first_order_optimizers.py`)
- [x] **RMSProp** (`track_A_continuous/src/first_order_optimizers.py`)
- [x] **Adam** (`track_A_continuous/src/first_order_optimizers.py`)
- [x] **Newton's Method** (Full Hessian inversion with Levenberg-Marquardt damping) (`track_A_continuous/src/second_order_methods.py`)
- [x] **Gauss-Newton Method** (Non-linear least squares $J^T J$) (`track_A_continuous/src/second_order_methods.py`)
- [x] **Curvature Approximations**:
  - [x] Empirical Fisher Information Matrix (`track_A_continuous/src/second_order_methods.py`)
  - [x] Diagonal Hessian approximation (`track_A_continuous/src/second_order_methods.py`)
  - [x] Low-rank Hessian spectral approximation (`track_A_continuous/src/second_order_methods.py`)
  - [x] Hessian-Vector Products (Pearlmutter finite-difference trick) (`track_A_continuous/src/second_order_methods.py`)
- [x] **Backpropagation**: Separate gradient computation engine on an MLP (`track_A_continuous/src/mlp_backpropagation.py`)
- [x] **Convergence Analysis**: Convex, strongly convex, non-convex, and saddle-point benchmarks (`track_A_continuous/results/first_order_metrics.md`, `second_order_metrics.md`)
- [x] **Step-size & Hyperparameter Tuning**: Sensitivity analysis across learning rates and damping (`track_A_continuous/results/`)
- [x] **Noisy Objectives**: Robustness evaluation on noisy linear regression (`track_A_continuous/src/benchmark_functions.py`)

---

## 3. Discrete Optimization (Track B)
- [x] **Greedy Heuristics**: 0-1 Knapsack value-to-weight ratio heuristic (`track_B_discrete/src/knapsack_solvers.py`)
- [x] **Integer Linear Programming (ILP)**: Exact modeling via HiGHS branch-and-cut solver (`track_B_discrete/src/knapsack_solvers.py`)
- [x] **Branch-and-Bound**: Exact tree search with fractional LP relaxation upper bounds (`track_B_discrete/src/knapsack_solvers.py`)
- [x] **Dynamic Programming**: Exact 0-1 Knapsack $O(NW)$ table with backtracking (`track_B_discrete/src/knapsack_solvers.py`)
- [x] **QUBO Formulation**: Binary-to-spin variable transformation ($s_i = 1 - 2x_i$) (`track_B_discrete/src/qubo_ising.py`)
- [x] **Ising Model Penalty Formulation**: Constraint enforcement ($P \cdot (\sum x_i - K)^2$) (`track_B_discrete/src/qubo_ising.py`)
- [x] **Classical Baseline**: Simulated Annealing solver (serves as baseline for Track C) (`track_B_discrete/src/qubo_ising.py`)

---

## 4. Quantum Optimization (Track C)
- [x] **QAOA**: Quantum Approximate Optimization Algorithm ($p=1, 2, 3$) on Track B QUBO / Graph Max-Cut (`track_C_quantum/src/qaoa_maxcut.py`)
- [x] **VQE**: Variational Quantum Eigensolver on Ising Hamiltonian (`track_C_quantum/src/vqe_ising.py`)
- [x] **PQCs & QNNs**: Parameterized Quantum Circuits and Quantum Classifiers (`track_C_quantum/src/pqc_qnn.py`)
- [x] **Quantum Gradients**:
  - [x] Parameter-shift rule (`track_C_quantum/src/quantum_gradients.py`)
  - [x] Simultaneous Perturbation Stochastic Approximation (SPSA) (`track_C_quantum/src/quantum_gradients.py`)
  - [x] Quantum Natural Gradient (QNG with Fubini-Study metric) (`track_C_quantum/src/quantum_gradients.py`)
- [x] **Empirical Quantum Experiments**:
  - [x] Classical vs. Quantum baseline comparison (QNN vs Classical MLP, QAOA/VQE vs BF/SA) (`track_C_quantum/results/`)
  - [x] Finite-shot measurement variance ($100, 1000, 10000$ shots) (`track_C_quantum/src/quantum_noise_experiments.py`)
  - [x] Quantum hardware noise model (depolarizing/readout noise) (`track_C_quantum/src/quantum_noise_experiments.py`)
  - [x] Barren plateaus empirical study ($N \in [2..10]$ qubits) (`track_C_quantum/src/quantum_noise_experiments.py`)

---

## 5. Submodular, Parallel & Distributed Optimization (Track D)
- [x] **Submodular Optimization**:
  - [x] Diminishing returns property verification ($f(A \cup \{e\}) - f(A) \ge f(B \cup \{e\}) - f(B)$) (`track_D_submodular_parallel/src/submodular_maximization.py`)
  - [x] Greedy selection implementation (Standard Greedy & Accelerated Lazy Greedy / Minoux) (`track_D_submodular_parallel/src/submodular_maximization.py`)
  - [x] Empirical check of $(1 - 1/e)$ approximation bound vs exact Brute Force (`track_D_submodular_parallel/results/submodular_metrics.md`)
- [x] **Parallel Program Optimization**:
  - [x] Multi-core load balancing, makespan, and work stealing simulation (`track_D_submodular_parallel/src/parallel_optimization.py`)
  - [x] Locality, synchronization, and communication overhead analysis (`track_D_submodular_parallel/results/parallel_scaling_metrics.md`)
  - [x] Data, model, and pipeline parallelism comparison (Ring All-Reduce DDP & 1F1B bubble models) (`track_D_submodular_parallel/src/parallel_optimization.py`)
  - [x] Speedup $S(p)$ and parallel efficiency $E(p)$ measurement (1, 2, 4, 8 workers) (`track_D_submodular_parallel/results/parallel_scaling_metrics.csv`)
  - [x] Amdahl's Law and Gustafson's Law theoretical fits (`track_D_submodular_parallel/results/parallel_scaling_metrics.md`, `track_d_benchmarks.png`)
- [x] **Distributed & Multi-Task Optimization**:
  - [x] Federated Learning (**FedAvg**) vs. Centralized training across local epochs $E$ and rounds $R$ (`track_D_submodular_parallel/src/federated_multitask.py`)
  - [x] Multiple Gradient Descent Algorithm (**MGDA**) for Pareto-stationarity (`track_D_submodular_parallel/src/federated_multitask.py`)
  - [x] **GradNorm** dynamic gradient balancing vs. Uniform task weighting (`track_D_submodular_parallel/src/federated_multitask.py`, `track_D_submodular_parallel/results/federated_multitask_metrics.md`)

---

## 6. Main Real-Process Case Study (Track E)
- [x] **Problem Domain**: EV Fleet Smart Charging & Route Scheduling (Independent of F–J) (`track_E_case_study/src/ev_charging_problem.py`)
- [x] **Section 4 Workflow Steps 1–7 executed**: Process identification, baseline ASAP simulation, mathematical formulation, solver implementations, empirical benchmarks, and sensitivity analysis (`track_E_case_study/results/ev_fleet_metrics.md`)
- [x] **Strategy Selection Table completed**: Continuous vs Linearized vs Greedy comparison across variables, constraints, size, hardware, and parallelism (`track_E_case_study/results/ev_fleet_metrics.md`)
- [x] **Experimental comparison of at least two Chapter 6 methods against baseline**: Uncontrolled ASAP vs Continuous Convex (SLSQP) vs Linear Programming (HiGHS) vs Priority Greedy (`track_E_case_study/results/ev_fleet_metrics.csv`, `track_E_case_study/results/ev_fleet_charging_profiles.png`)

---

## 7. Systems & Application Tracks (Tracks F, G, H, I, J)
- [x] **Track F (Compiler)**: Mini-compiler with 4 passes (Const Fold, DCE, CSE, LICM) + ILP Register Allocation + Output Equivalence (`track_F_compiler/src/ir.py`, `optimizations.py`, `track_F_compiler/results/compiler_metrics.md`)
- [ ] **Track G (xv6 OS)**: Priority/Stride Scheduler + Stats Syscall + 3 Workloads in QEMU + `usertests` pass + Patch
- [ ] **Track H (DBMS)**: Knapsack/ILP Index Selection + Join-Order DP + TPC-H SF 0.1 benchmark + EXPLAIN comparison
- [ ] **Track I (Quantum Algorithms)**: DJ, BV, Grover, QFT, QPE, Shor (15), Annealing + 2 handcrafted circuits + VQC vs Classical SVM + Grover scaling analysis
- [ ] **Track J (Software Engineering)**: Test Suite Minimization on 1,000+ LOC codebase via 0-1 ILP and Greedy vs Naive
