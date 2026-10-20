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
- [ ] **QAOA**: Quantum Approximate Optimization Algorithm ($p=1, 2$) on Track B QUBO
- [ ] **VQE**: Variational Quantum Eigensolver on Ising Hamiltonian
- [ ] **PQCs & QNNs**: Parameterized Quantum Circuits
- [ ] **Quantum Gradients**:
  - [ ] Parameter-shift rule
  - [ ] Simultaneous Perturbation Stochastic Approximation (SPSA)
  - [ ] Quantum Natural Gradient (QNG)
- [ ] **Empirical Quantum Experiments**:
  - [ ] Classical vs. Quantum baseline comparison
  - [ ] Finite-shot measurement variance ($100, 1000, 10000$ shots)
  - [ ] Quantum hardware noise model (depolarizing/readout noise)
  - [ ] Barren plateaus empirical study ($N \in [2..10]$ qubits)

---

## 5. Submodular, Parallel & Distributed Optimization (Track D)
- [ ] **Submodular Optimization**:
  - [ ] Diminishing returns property verification ($f(A \cup \{e\}) - f(A) \ge f(B \cup \{e\}) - f(B)$)
  - [ ] Greedy selection implementation
  - [ ] Empirical check of $(1 - 1/e)$ approximation bound
- [ ] **Parallel Program Optimization**:
  - [ ] Multi-core load balancing, makespan, and work stealing
  - [ ] Locality, synchronization, and communication overhead
  - [ ] Data, model, and pipeline parallelism comparison
  - [ ] Speedup $S(p)$ and parallel efficiency $E(p)$ measurement (1, 2, 4, 8 workers)
  - [ ] Amdahl's Law and Gustafson's Law theoretical fits
- [ ] **Distributed & Multi-Task Optimization**:
  - [ ] Federated Learning (**FedAvg**) vs. Centralized training across local epochs $E$ and rounds $R$
  - [ ] Multiple Gradient Descent Algorithm (**MGDA**) for Pareto-stationarity
  - [ ] **GradNorm** dynamic gradient balancing vs. Uniform task weighting

---

## 6. Main Real-Process Case Study (Track E)
- [ ] Problem Domain: EV Fleet Smart Charging & Route Scheduling (Independent of F–J)
- [ ] Section 4 Workflow Steps 1–7 executed
- [ ] Strategy Selection Table completed (variables, constraints, size, hardware, parallelism)
- [ ] Experimental comparison of at least two Chapter 6 methods against baseline

---

## 7. Systems & Application Tracks (Tracks F, G, H, I, J)
- [ ] **Track F (Compiler)**: Mini-compiler with 4 passes (Const Fold, DCE, CSE, LICM) + ILP Register Allocation + Output Equivalence
- [ ] **Track G (xv6 OS)**: Priority/Stride Scheduler + Stats Syscall + 3 Workloads in QEMU + `usertests` pass + Patch
- [ ] **Track H (DBMS)**: Knapsack/ILP Index Selection + Join-Order DP + TPC-H SF 0.1 benchmark + EXPLAIN comparison
- [ ] **Track I (Quantum Algorithms)**: DJ, BV, Grover, QFT, QPE, Shor (15), Annealing + 2 handcrafted circuits + VQC vs Classical SVM + Grover scaling analysis
- [ ] **Track J (Software Engineering)**: Test Suite Minimization on 1,000+ LOC codebase via 0-1 ILP and Greedy vs Naive
