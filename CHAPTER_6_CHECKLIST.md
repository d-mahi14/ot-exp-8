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
- [ ] **Gradient Descent (GD)** (Vanilla batch gradient descent)
- [ ] **Stochastic Gradient Descent (SGD)** & **Mini-batch SGD**
- [ ] **Momentum Optimizer**
- [ ] **Nesterov Accelerated Gradient (NAG)**
- [ ] **AdaGrad**
- [ ] **RMSProp**
- [ ] **Adam**
- [ ] **Newton's Method** (Full Hessian inversion)
- [ ] **Gauss-Newton Method** (Non-linear least squares)
- [ ] **Curvature Approximations**:
  - [ ] Fisher Information Matrix approximation
  - [ ] Diagonal / Low-rank Hessian approximation
  - [ ] Hessian-Vector Products (Pearlmutter trick)
- [ ] **Backpropagation**: Separate gradient computation engine on an MLP
- [ ] **Convergence Analysis**: Convex, strongly convex, and non-convex benchmark comparison
- [ ] **Step-size & Hyperparameter Tuning**: Sensitivity grid sweeps
- [ ] **Noisy Objectives**: Optimizer robustness under noisy gradient estimations

---

## 3. Discrete Optimization (Track B)
- [ ] **Greedy Heuristics**: Knapsack / Max-Cut greedy approximation
- [ ] **Integer Linear Programming (ILP)**: Exact modeling via PuLP / HiGHS
- [ ] **Branch-and-Bound**: Exact tree search with upper/lower bound pruning
- [ ] **Dynamic Programming**: Optimal substructure and memoization
- [ ] **QUBO Formulation**: Binary-to-spin variable transformation ($s_i = 1 - 2x_i$)
- [ ] **Ising Model Penalty Formulation**: Constraint enforcement ($P \cdot (\sum x_i - K)^2$)
- [ ] **Classical Baseline**: Simulated Annealing solver (serves as baseline for Track C)

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
