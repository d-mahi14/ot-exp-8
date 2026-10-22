# Optimization Techniques (Lab Assignment 8): Master Technical Report
## Optimize Anything — The 14-Day Vibe-Coding Challenge
**Course**: Optimization Techniques  
**Repository**: `c:\dev\ot-8`  
**Submission Artifact**: Comprehensive Technical Report & Portfolio Audit  

---

## Executive Summary

This technical report documents the complete, empirical 10-track optimization portfolio for Optimization Techniques Lab Assignment 8. Across 14 days of AI-assisted (vibe) coding with strict verification protocols, this project investigated optimization opportunities spanning machine learning, combinatorial algorithms, quantum computing, distributed systems, compilers, operating systems, database management systems, quantum machine learning, and software engineering.

Every track establishes an unoptimized baseline, formulates the optimization problem mathematically (decision variables, objective functions, constraints, and assumptions), implements at least one named Chapter 6 optimization algorithm, validates correctness via independent automated unit tests (151/151 passing), and benchmarks performance using raw CSV metrics, detailed Markdown summaries, and high-resolution visual plots.

---

## 1. Strategy Selection Table (Section 4 Workflow)

The table below summarizes the problem characteristics, decision variables, constraints, and optimization strategies deployed across all 10 compulsory tracks:

| Track | Problem Domain | Decision Variables | Constraints | Chapter 6 Optimization Method(s) | Baseline | Measured Gain / Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Track A** | Continuous & Curvature ML Optimization | Weight vectors $\theta \in \mathbb{R}^d$ | Unconstrained / Smooth | GD, SGD, Momentum, NAG, AdaGrad, RMSProp, Adam, Newton, Gauss-Newton, Fisher/Low-rank/HVP | Vanilla Batch GD | Momentum achieved $500\times$ lower loss than GD on Rosenbrock; Newton converged in 1 step on quadratics |
| **Track B** | Discrete & QUBO Combinatorial Optimization | Binary indicator vector $x \in \{0, 1\}^n$, Spin $s_i \in \{-1, +1\}$ | Capacity $\sum w_i x_i \le W$, Cardinality $\sum x_i = K$ | Greedy ratio heuristic, 0-1 Knapsack Dynamic Programming, Branch-and-Bound, 0-1 ILP (HiGHS), Simulated Annealing | Greedy heuristic / Random assignment | ILP, DP, and B&B achieved identical global optimum ($V=280$); SA achieved 100% ground state accuracy on 6-node Max-Cut |
| **Track C** | Variational Quantum Optimization | Variational angles $\gamma, \beta \in [0, 2\pi)^p$, ansatz weights $\theta$ | Unitary gate constraints, parameter bounds | QAOA ($p=1, 2, 3$), VQE with RealAmplitudes, Parameter-shift rule, SPSA, Quantum Natural Gradient (QNG) | Classical Brute-Force & Simulated Annealing | QAOA approximation ratio improved from $0.78$ ($p=1$) to $0.93$ ($p=3$); shot-noise variance decayed as $1/\sqrt{M}$ |
| **Track D** | Submodular, Parallel & Distributed Optimization | Subset $S \subseteq V$, task assignments, client federated model weights | Budget $|S| \le k$, worker synchronization, communication epochs | Greedy Set Cover with Minoux Lazy evaluation, Ring All-Reduce DDP, 1F1B Pipeline, FedAvg, MGDA, GradNorm | Exhaustive brute force, Single-core sequential, Centralized training, Uniform multi-task weighting | Lazy Greedy reduced function evaluations by $73.3\%$; ThreadPool BLAS parallel speedup reached $2.8\times$ on 4 cores |
| **Track E** | Main Case Study: EV Fleet Smart Charging Scheduling | Continuous charging powers $P_{i, t} \in [0, P_{\max}]$, binary activation $u_{i, t}$ | Vehicle energy demands $E_i$, station transformer capacity $P_{\text{grid}}$, departure deadlines | Continuous Convex (SLSQP), Linear Programming (HiGHS), Priority Valley-Filling Greedy | Uncontrolled ASAP Charging (Immediate plug-in) | LP/SLSQP reduced electricity costs by **67.0%** ($1,048.50 \to 346.00$), eliminating peak transformer overload entirely |
| **Track F** | Mini-Compiler Code Optimization Passes & Register Allocation | IR instructions, physical register mapping $r_v \in \{R_1, \dots, R_k\}$ | Interference graph edges $(u, v) \notin \text{same register}$, live range conflicts | 4 IR passes (ConstFold, DCE, CSE, LICM) + 0-1 ILP Register Allocation | Unoptimized Three-Address Code (TAC) IR | Execution instructions reduced by **45.0%** ($20 \to 11$), loop invariants hoisted, 0 spills with $K=3$ registers |
| **Track G** | xv6 Teaching OS Kernel Optimization | Process tickets, stride values, buffer cache replacement pointers | Stride proportionality $\text{stride}_i \propto 1/\text{tickets}_i$, fixed buffer cache memory size | Stride Scheduling with pass compensation + MRU/LRU adaptive buffer cache hit profiling | Standard xv6 Round-Robin Scheduler & Unprofiled Bcache | CPU allocation perfectly proportional ($2.01:1.00$ for $20:10$ tickets); bcache hit rate improved to **93.5%** on hot sequential reads |
| **Track H** | DBMS Workload & Physical Design Optimization | Binary index indicators $x_i \in \{0, 1\}$, join sequence permutations $\pi$ | Storage budget $\sum s_i x_i \le B_{\max}$, valid relational algebra join trees | 0-1 Knapsack/ILP Index Selection + System-R Join-Order Dynamic Programming | Unindexed sequential scans & Arbitrary left-deep join order | Workload execution latency accelerated by **15.9x** ($2,560\text{ ms} \to 161\text{ ms}$); Join DP cost dropped from $1.4\times 10^7$ to $8.2\times 10^4$ |
| **Track I** | Canonical Quantum Algorithms & QML vs Classical Baselines | Quantum state evolution $|\psi\rangle = U |0\rangle$, Kernel matrix $K(x_i, x_j)$ | Unitary constraints, NISQ depolarizing & readout noise budgets | Deutsch-Jozsa, Bernstein-Vazirani, QFT, QPE, Shor-15, Grover Search, Fidelity Quantum Kernel QSVM | Classical linear search, Classical RBF SVM, Classical Linear SVM | Grover achieved 100% target detection on 2–6 qubits ($O(\sqrt{N})$ iterations); QSVM achieved **85.7%** accuracy, matching classical RBF |
| **Track J** | Software-Engineering Optimization: Test Suite Minimization | Binary test retention indicators $x_j \in \{0, 1\}$ | Full statement line coverage $\sum A_{ij} x_j \ge 1$, Mutation kill retention $\sum M_{kj} x_j \ge 1$ | 0-1 ILP Minimum Weight Set Cover (HiGHS) + Chvátal Greedy Set Cover Heuristic | Exhaustive Full Suite Execution (80 tests) | Test suite reduced by **33.8%** ($80 \to 53$ tests), delivering **1.56x speedup** while retaining **100% statement coverage** and **100% fault detection** |

---

## 2. Track-by-Track Technical Audit & Evidence

### Track A: Continuous Optimization
- **Problem**: Optimization of first-order and second-order methods over convex (Sphere, Strongly Convex Quadratic), non-convex (Rosenbrock "banana valley"), and saddle-point landscapes.
- **Formulation**: Minimizing objective $f(\theta)$, where gradient $\nabla f(\theta)$ and Hessian $\nabla^2 f(\theta)$ govern parameter updates.
- **Key Findings**:
  - On the ill-conditioned Rosenbrock valley, Momentum ($\gamma=0.9$) achieved a final loss of $0.00558$ and NAG achieved $0.00750$, beating vanilla GD ($2.726$) by nearly $500\times$. Adaptive optimizers (Adam, AdaGrad) required smaller steps along the canyon floor ($1.803$ and $1.478$).
  - Second-order Newton with Levenberg-Marquardt damping converged to machine precision ($10^{-16}$) on quadratic bowls in 1 iteration, while Gauss-Newton solved non-linear least squares without computing full second derivatives.
  - Backpropagation was verified on a custom MLP with finite-difference gradient checking, demonstrating an analytical gradient error $< 8.94 \times 10^{-11}$.
- **Artifacts**: [`first_order_metrics.csv`](file:///c:/dev/ot-8/track_A_continuous/results/first_order_metrics.csv), [`second_order_metrics.csv`](file:///c:/dev/ot-8/track_A_continuous/results/second_order_metrics.csv), [`first_order_convergence.png`](file:///c:/dev/ot-8/track_A_continuous/results/first_order_convergence.png), [`saddle_point_escape.png`](file:///c:/dev/ot-8/track_A_continuous/results/saddle_point_escape.png).
- **Tests**: 22 passing tests in [`test_continuous.py`](file:///c:/dev/ot-8/track_A_continuous/tests/test_continuous.py).

### Track B: Discrete Optimization
- **Problem**: 0-1 Knapsack problem ($N=10$ items, $W=50$ kg) and Quadratic Unconstrained Binary Optimization (QUBO) for Graph Max-Cut ($N=6$ nodes, 9 edges).
- **Formulation**:
  - 0-1 Knapsack: $\max \sum v_i x_i \text{ s.t. } \sum w_i x_i \le W, x_i \in \{0, 1\}$.
  - Max-Cut QUBO: $H = -\sum_{(i,j) \in E} w_{ij} x_i (1 - x_j) = x^T Q x$, transformed to Ising spin Hamiltonian via $s_i = 1 - 2x_i$.
- **Key Findings**:
  - Dynamic Programming ($O(NW)$), Branch-and-Bound with fractional LP relaxation, and 0-1 ILP (HiGHS branch-and-cut) all found the exact identical optimal value ($V = 280$) and identical item selection subset.
  - The Greedy value-to-weight ratio heuristic achieved $V = 275$ ($98.2\%$ optimality) in $0.05$ ms, demonstrating the theoretical greedy approximation behavior.
  - Simulated Annealing solved the Max-Cut QUBO with 100% ground-state hit rate across 10 repeated runs, establishing the classical baseline for Track C QAOA.
- **Artifacts**: [`knapsack_metrics.csv`](file:///c:/dev/ot-8/track_B_discrete/results/knapsack_metrics.csv), [`qubo_benchmark_metrics.csv`](file:///c:/dev/ot-8/track_B_discrete/results/qubo_benchmark_metrics.csv), [`knapsack_comparison.png`](file:///c:/dev/ot-8/track_B_discrete/results/knapsack_comparison.png), [`qubo_penalty_and_annealing.png`](file:///c:/dev/ot-8/track_B_discrete/results/qubo_penalty_and_annealing.png).
- **Tests**: 25 passing tests in [`test_knapsack.py`](file:///c:/dev/ot-8/track_B_discrete/tests/test_knapsack.py) and [`test_qubo_ising.py`](file:///c:/dev/ot-8/track_B_discrete/tests/test_qubo_ising.py).

### Track C: Quantum Optimization
- **Problem**: Variational quantum algorithms (QAOA, VQE), Quantum Neural Networks (PQC vs classical MLP), analytical quantum gradients (Parameter-Shift, SPSA, QNG), and NISQ noise / barren plateau scaling.
- **Formulation**: Minimizing expectation value $\langle \psi(\theta) | H | \psi(\theta) \rangle$ over parameterized quantum circuits.
- **Key Findings**:
  - QAOA for Max-Cut demonstrated monotonic approximation ratio scaling as circuit depth increased: $r(p=1) = 0.784$, $r(p=2) = 0.865$, $r(p=3) = 0.932$.
  - VQE ground-state energy on a 2-qubit Ising Hamiltonian reached $-2.23606$ (0.0000% error relative to exact diagonalization).
  - Finite-shot measurement variance on non-eigenstate superpositions verified the Central Limit Theorem: $\sigma \propto 1/\sqrt{M}$ ($0.0903 \to 0.0253 \to 0.0097$ for $M \in \{100, 1000, 10000\}$ shots).
  - Barren plateau scaling from $N=2$ to $N=10$ qubits exhibited exponential gradient variance decay: $\text{Var}[\partial_\theta \langle H \rangle] \propto 2^{-cN}$.
- **Artifacts**: [`qaoa_benchmark_metrics.csv`](file:///c:/dev/ot-8/track_C_quantum/results/qaoa_benchmark_metrics.csv), [`vqe_benchmark_metrics.csv`](file:///c:/dev/ot-8/track_C_quantum/results/vqe_benchmark_metrics.csv), [`quantum_advanced_metrics.csv`](file:///c:/dev/ot-8/track_C_quantum/results/quantum_advanced_metrics.csv), [`qaoa_benchmarks.png`](file:///c:/dev/ot-8/track_C_quantum/results/qaoa_benchmarks.png), [`quantum_advanced_benchmarks.png`](file:///c:/dev/ot-8/track_C_quantum/results/quantum_advanced_benchmarks.png).
- **Tests**: 40 passing tests across [`test_qaoa_maxcut.py`](file:///c:/dev/ot-8/track_C_quantum/tests/test_qaoa_maxcut.py), [`test_vqe_ising.py`](file:///c:/dev/ot-8/track_C_quantum/tests/test_vqe_ising.py), [`test_quantum_gradients.py`](file:///c:/dev/ot-8/track_C_quantum/tests/test_quantum_gradients.py), and [`test_quantum_noise.py`](file:///c:/dev/ot-8/track_C_quantum/tests/test_quantum_noise.py).

### Track D: Submodular, Parallel & Distributed Optimization
- **Problem**: Submodular sensor placement (diminishing returns), multi-core parallel scaling (Amdahl & Gustafson laws), Federated Learning (FedAvg), and Multi-Task Optimization (MGDA and GradNorm).
- **Formulation**: Submodular set function $f(A \cup \{e\}) - f(A) \ge f(B \cup \{e\}) - f(B)$ for $A \subseteq B$.
- **Key Findings**:
  - Minoux Lazy Greedy matched standard Greedy value exactly while saving **73.3%** of objective evaluations.
  - Multi-threaded BLAS parallel execution achieved $2.17\times$ speedup on 2 cores and $2.84\times$ on 4 cores. Amdahl's law fit demonstrated a parallel fraction $s = 0.88$.
  - FedAvg with 5 local epochs converged within $1.2\%$ of centralized training loss while reducing network communication rounds by $5\times$.
  - Multi-Task Optimization: MGDA identified Pareto-stationary descent directions, while GradNorm dynamically balanced task gradient magnitudes, preventing the larger task from dominating.
- **Artifacts**: [`submodular_metrics.csv`](file:///c:/dev/ot-8/track_D_submodular_parallel/results/submodular_metrics.csv), [`parallel_scaling_metrics.csv`](file:///c:/dev/ot-8/track_D_submodular_parallel/results/parallel_scaling_metrics.csv), [`federated_multitask_metrics.csv`](file:///c:/dev/ot-8/track_D_submodular_parallel/results/federated_multitask_metrics.csv), [`track_d_benchmarks.png`](file:///c:/dev/ot-8/track_D_submodular_parallel/results/track_d_benchmarks.png).
- **Tests**: 18 passing tests in [`test_submodular.py`](file:///c:/dev/ot-8/track_D_submodular_parallel/tests/test_submodular.py), [`test_parallel.py`](file:///c:/dev/ot-8/track_D_submodular_parallel/tests/test_parallel.py), and [`test_federated_multitask.py`](file:///c:/dev/ot-8/track_D_submodular_parallel/tests/test_federated_multitask.py).

### Track E: Main Case Study — EV Fleet Smart Charging Scheduling
- **Problem**: Scheduling overnight charging for an enterprise fleet of 10 commercial Electric Vehicles (5 delivery vans, 3 service trucks, 2 transit shuttles) over a 24-hour horizon (96 time intervals of 15 minutes) under dynamic Time-of-Use (TOU) tariffs, vehicle arrival/departure availability windows, and a strict 120 kW grid transformer peak capacity.
- **Formulation**:
  $$\min_{P_{i, t}} \sum_{t=1}^{96} c_t \left( \sum_{i=1}^{10} P_{i, t} \right) \Delta t + \lambda_{\text{peak}} \max_t \left( \sum_{i=1}^{10} P_{i, t} \right)$$
  subject to:
  $$\sum_{t=1}^{96} P_{i, t} \eta \Delta t = E_i^{\text{req}} \quad \forall i, \quad \sum_{i=1}^{10} P_{i, t} \le P_{\text{grid}}^{\max} \quad \forall t, \quad P_{i, t} \le P_i^{\max} \cdot A_{i, t}$$
- **Key Findings**:
  - **Baseline (Uncontrolled ASAP)**: Every vehicle charges at maximum power immediately upon arrival. This caused an extreme coincident demand spike of **170.0 kW** (exceeding the 120 kW transformer limit by 41.7%) and incurred an electricity cost of **$1,048.50**.
  - **Linear Programming (HiGHS)** & **Continuous Convex (SLSQP)**: Solved in 24 ms and 185 ms, delivering an identical minimum cost of **$346.00** (**67.0% cost reduction**), while keeping peak grid demand at **65.0 kW** (well below the 120 kW limit).
  - **Priority Greedy (Valley-Filling)**: Achieved a cost of **$382.40** (**63.5% cost reduction**) in 2.1 ms, demonstrating near-optimal performance suitable for embedded charging station microcontrollers.
- **Artifacts**: [`ev_fleet_metrics.csv`](file:///c:/dev/ot-8/track_E_case_study/results/ev_fleet_metrics.csv), [`ev_fleet_metrics.md`](file:///c:/dev/ot-8/track_E_case_study/results/ev_fleet_metrics.md), [`ev_fleet_charging_profiles.png`](file:///c:/dev/ot-8/track_E_case_study/results/ev_fleet_charging_profiles.png).
- **Tests**: 7 passing tests in [`test_ev_charging.py`](file:///c:/dev/ot-8/track_E_case_study/tests/test_ev_charging.py).

### Track F: Mini-Compiler Optimization
- **Problem**: Optimization of a Three-Address Code (TAC) intermediate representation across 4 optimization passes, and formulating Register Allocation as a 0-1 Integer Linear Program (ILP).
- **Formulation**: Register interference graph vertex coloring mapped to 0-1 ILP:
  $$\min \sum_{v} \text{SpillCost}_v \cdot s_v \quad \text{s.t.} \quad x_{u, r} + x_{v, r} \le 1 \quad \forall (u, v) \in E, \quad \sum_{r=1}^K x_{v, r} + s_v = 1$$
- **Key Findings**:
  - The 4-pass optimization pipeline (Constant Folding/Propagation &rarr; Common Subexpression Elimination &rarr; Loop Invariant Code Motion &rarr; Dead Code Elimination) reduced executed instructions by **45.0%** ($20 \to 11$ instructions).
  - The 0-1 ILP register allocator placed all variables into $K=3$ physical registers with 0 spills.
  - Output equivalence test verified identical numerical output between unoptimized and optimized programs ($100.0\%$).
- **Artifacts**: [`compiler_metrics.csv`](file:///c:/dev/ot-8/track_F_compiler/results/compiler_metrics.csv), [`compiler_metrics.md`](file:///c:/dev/ot-8/track_F_compiler/results/compiler_metrics.md), [`compiler_benchmarks.png`](file:///c:/dev/ot-8/track_F_compiler/results/compiler_benchmarks.png).
- **Tests**: 8 passing tests in [`test_compiler.py`](file:///c:/dev/ot-8/track_F_compiler/tests/test_compiler.py).

### Track G: xv6 Operating-System Optimization
- **Problem**: Modifying the MIT xv6 RISC-V teaching operating system to implement a deterministic Stride Process Scheduler and kernel-level Buffer Cache (bcache) hit/miss profiling syscalls.
- **Formulation**: Each process is assigned tickets $T_i \ge 1$. $\text{Stride}_i = \text{LargeConstant} / T_i$. At every timer interrupt, the scheduler selects process $p^* = \arg\min_p \text{pass}_p$ and increments $\text{pass}_{p^*} += \text{stride}_{p^*}$. Dynamic wakeup pass compensation prevents newly awakened processes from starving the CPU.
- **Key Findings**:
  - CPU runtime allocation was benchmarked in QEMU across 3 concurrent workloads (`forktest`, `schedtest`, `bcachetest`).
  - Two processes with tickets 20 and 10 achieved an empirical tick ratio of $66.8\% : 33.2\%$ ($2.01 : 1.00$), matching theoretical proportionality.
  - Kernel buffer cache hit rate reached **93.5%** on sequential I/O workloads, validated via new system call `getbcachestats()`.
  - System passed full kernel build, boot sequence, and `usertests`.
- **Artifacts**: [`xv6_stride_and_bcache.patch`](file:///c:/dev/ot-8/track_G_xv6/patch/xv6_stride_and_bcache.patch), [`xv6_metrics.csv`](file:///c:/dev/ot-8/track_G_xv6/results/xv6_metrics.csv), [`xv6_metrics.md`](file:///c:/dev/ot-8/track_G_xv6/results/xv6_metrics.md), [`xv6_benchmarks.png`](file:///c:/dev/ot-8/track_G_xv6/results/xv6_benchmarks.png).
- **Tests**: 5 passing tests in [`test_xv6.py`](file:///c:/dev/ot-8/track_G_xv6/tests/test_xv6.py).

### Track H: DBMS Workload Optimization
- **Problem**: Automated Physical Design (0-1 Knapsack / ILP index selection) and System-R Join-Order Dynamic Programming on TPC-H (Scale Factor 0.1, 100,000+ rows).
- **Formulation**:
  - Index Selection: $\max \sum_{i} \Delta \text{Benefit}_i \cdot x_i \text{ s.t. } \sum s_i x_i \le B_{\text{storage}}, x_i \in \{0, 1\}$.
  - Join Order DP: $C(S) = \min_{R \subset S} [C(R) + C(S \setminus R) + \text{JoinCost}(R, S \setminus R)]$.
- **Key Findings**:
  - Unindexed baseline workload latency: **2,560 ms** (sequential table scans on `lineitem` and `orders`).
  - 0-1 ILP Index Selection chose the optimal 3-index configuration under a 15 MB storage budget, reducing workload latency to **161 ms** (**15.9x speedup**).
  - System-R Join-Order DP evaluated all $2^k - 1$ sub-plans for TPC-H Q5 (6-table join), selecting an optimal join tree with estimated cost $8.2 \times 10^4$ compared to $1.4 \times 10^7$ for an arbitrary left-deep plan. Query output hash was bit-for-bit identical before and after.
- **Artifacts**: [`dbms_metrics.csv`](file:///c:/dev/ot-8/track_H_dbms/results/dbms_metrics.csv), [`dbms_metrics.md`](file:///c:/dev/ot-8/track_H_dbms/results/dbms_metrics.md), [`dbms_benchmarks.png`](file:///c:/dev/ot-8/track_H_dbms/results/dbms_benchmarks.png).
- **Tests**: 6 passing tests in [`test_dbms.py`](file:///c:/dev/ot-8/track_H_dbms/tests/test_dbms.py).

### Track I: Quantum Algorithms & QML Advantage
- **Problem**: Implementation and verification of canonical quantum algorithms (Deutsch-Jozsa, Bernstein-Vazirani, QFT, QPE, Shor's period-finding for $N=15=3\times 5$, Grover's search with $O(\sqrt{N})$ scaling), NISQ hardware noise modeling, and Fidelity Quantum Kernel Estimation (QSVM) vs Classical SVM.
- **Formulation**: Circuit statevector transformations and quantum feature mapping $\Phi(x) = U_\Phi(x)|0\rangle$, $K(x_i, x_j) = |\langle \Phi(x_i) | \Phi(x_j) \rangle|^2$.
- **Key Findings**:
  - Grover search achieved 100% target state detection on ideal simulators across 2 to 6 qubits using exact trigonometric rotations $R = \text{round}(\frac{\pi}{4\theta} - 0.5)$.
  - Under a realistic NISQ noise model (depolarizing error $p_1=0.1\%, p_2=1.2\%$ and readout error $2.0\%$), success probability degraded from $100\%$ to $74.2\%$ on 2 qubits and $39.5\%$ on 6 qubits, proving that two-qubit CX gate accumulation limits NISQ scaling.
  - QSVM achieved **85.7%** test accuracy on non-linear structured data, matching classical RBF SVM (85.7%) and outperforming classical linear SVM (81.0%).
- **Artifacts**: [`track_i_metrics.csv`](file:///c:/dev/ot-8/track_I_quantum_algorithms/results/track_i_metrics.csv), [`track_i_metrics.md`](file:///c:/dev/ot-8/track_I_quantum_algorithms/results/track_i_metrics.md), [`track_i_benchmarks.png`](file:///c:/dev/ot-8/track_I_quantum_algorithms/results/track_i_benchmarks.png).
- **Tests**: 11 passing tests in [`test_quantum_algorithms.py`](file:///c:/dev/ot-8/track_I_quantum_algorithms/tests/test_quantum_algorithms.py).

### Track J: Software-Engineering Optimization
- **Problem**: Test Suite Minimization (TSM) on a real 1,055 LOC enterprise financial and inventory order management system.
- **Formulation**: Minimum Weight Set Cover under dual constraints: 100% statement coverage AND 100% mutation fault retention.
- **Key Findings**:
  - Full suite (80 tests) took 107.5 ms.
  - 0-1 ILP (HiGHS) selected 53 tests (**33.8% reduction**), achieving **1.56x speedup** (68.9 ms) with **100% statement coverage** and **100% mutation kill score**.
  - Chvátal Greedy selected 57 tests (**28.7% reduction**, 1.52x speedup) in only **4.55 ms** solver overhead.
  - Random Subsampling ($k=53$) dropped coverage to 82.9% and missed 26.7% of domain faults.
- **Artifacts**: [`track_j_metrics.csv`](file:///c:/dev/ot-8/track_J_software_engineering/results/track_j_metrics.csv), [`track_j_metrics.md`](file:///c:/dev/ot-8/track_J_software_engineering/results/track_j_metrics.md), [`track_j_benchmarks.png`](file:///c:/dev/ot-8/track_J_software_engineering/results/track_j_benchmarks.png).
- **Tests**: 9 passing tests in [`test_track_j.py`](file:///c:/dev/ot-8/track_J_software_engineering/tests/test_track_j.py).

---

## 3. Answers to Section 11 Final Questions

### Question 1: What process did you choose, and what evidence showed that it needed optimization?
**Answer**:
For our main case study (Track E), we chose **Commercial Electric Vehicle (EV) Fleet Charging Scheduling**.
- *Evidence of Inefficiency*: In the baseline uncontrolled charging scenario (ASAP charging), commercial fleet vehicles were plugged in immediately upon completing their daytime delivery routes (17:00–18:00). Because driver arrival times were clustered, this created a coincident peak demand spike of **170.0 kW**, severely exceeding the local distribution transformer's thermal safety rating of **120.0 kW** (a 41.7% overload that would trip utility circuit breakers). Furthermore, charging coincided with peak Time-of-Use (TOU) utility rate periods ($0.28/kWh), causing an exorbitant daily electricity cost of **$1,048.50**.

### Question 2: How did you formulate the objective function and constraints?
**Answer**:
We formulated the problem over a 24-hour horizon discretized into 96 intervals of $\Delta t = 15$ minutes ($t \in \{1, \dots, 96\}$).
- **Decision Variables**: Continuous charging power $P_{i, t} \in [0, P_i^{\max}]$ allocated to vehicle $i$ during interval $t$.
- **Objective Function**: Minimize composite total cost:
  $$\min \sum_{t=1}^{96} c_t \left( \sum_{i=1}^{10} P_{i, t} \right) \Delta t + \lambda_{\text{peak}} \max_t \left( \sum_{i=1}^{10} P_{i, t} \right)$$
  where $c_t$ is the TOU electricity price vector ($0.08/kWh off-peak, $0.14/kWh mid-peak, $0.28/kWh on-peak), and $\lambda_{\text{peak}}$ penalizes peak station power.
- **Constraints**:
  1. *Energy Fulfillment*: $\sum_{t=1}^{96} P_{i, t} \eta \Delta t = E_i^{\text{req}}$ for all vehicles $i \in \{1, \dots, 10\}$ (100% required state of charge delivered before departure).
  2. *Grid Transformer Envelope*: $\sum_{i=1}^{10} P_{i, t} \le P_{\text{grid}}^{\max} = 120.0\text{ kW}$ for all intervals $t$.
  3. *Vehicle Availability Windows*: $P_{i, t} \le P_i^{\max} \cdot A_{i, t}$, where $A_{i, t} \in \{0, 1\}$ is 1 only when vehicle $i$ is parked at the depot between arrival and departure.

### Question 3: Which Chapter 6 methods did you test, and why were they appropriate for their respective experiments?
**Answer**:
We tested an extensive portfolio across Chapter 6:
1. **Continuous First & Second-Order Methods (SLSQP, Newton, Adam, NAG)**: Tested on Track E and Track A. Appropriate because EV power variables and neural network weights are continuous and differentiable.
2. **0-1 Integer Linear Programming (HiGHS Branch-and-Cut)**: Tested on Tracks B, E, F, H, and J. Appropriate because knapsack item selection, register allocation interference, DBMS index selection, and test suite minimization are discrete NP-hard combinatorial problems with linear constraints.
3. **Dynamic Programming**: Tested on 0-1 Knapsack (Track B) and System-R Join Ordering (Track H). Appropriate due to optimal substructure and overlapping subproblems ($O(2^k)$ vs $O(k!)$ join trees).
4. **Submodular Greedy with Lazy Evaluation**: Tested on Sensor Placement (Track D) and Test Suite Minimization (Track J). Appropriate because coverage exhibits diminishing marginal returns, guaranteeing $(1 - 1/e)$ approximation.
5. **Quantum Optimization (QAOA, VQE, QSVM)**: Tested on Tracks C and I. Appropriate for evaluating quantum approximate combinatorial search and fidelity quantum kernel feature spaces on NISQ hardware models.

### Question 4: How did the optimized solution compare with the baseline?
**Answer**:
Across all domains, optimized solutions demonstrated dramatic, measurable improvements over unoptimized baselines:
- **Track E (EV Fleet)**: Reduced daily cost by **67.0%** ($1,048.50 \to $346.00) and eliminated transformer overload ($170\text{ kW} \to 65\text{ kW}$).
- **Track H (DBMS)**: Reduced query workload latency by **15.9x** ($2,560\text{ ms} \to 161\text{ ms}$) using ILP index selection.
- **Track F (Compiler)**: Reduced runtime instruction count by **45.0%** ($20 \to 11$) with zero register spills.
- **Track J (Software Engineering)**: Reduced test suite size by **33.8%** ($80 \to 53$), delivering a **1.56x speedup** while retaining 100% statement coverage and 100% mutation detection.
- **Track A (Continuous)**: Momentum achieved **$500\times$ lower loss** on the Rosenbrock banana valley compared to vanilla GD.

### Question 5: What trade-offs, failures, or limitations did you discover?
**Answer**:
1. **The Coverage vs. Fault-Detection Gap (Track J)**: Minimizing test suites solely on code line coverage discarded edge-case tests, causing mutation score to drop from 100% to 86.7%. Adding fault/mutation constraints was necessary to prevent defect escape.
2. **Grover Over-Rotation on Small Search Spaces (Track I)**: Using the asymptotic formula $R \approx \frac{\pi}{4}\sqrt{N}$ caused severe over-rotation for $N=4$ (2 iterations instead of 1), dropping success probability to 23.9%. Exact trigonometric derivation was required.
3. **The Limits of Quantum Annealing / QAOA Under NISQ Noise (Tracks C & I)**: Two-qubit gate error rates ($1.2\%$) rapidly degrade quantum advantage as circuit depth increases, causing success probability to fall to $39.5\%$ at 6 qubits.
4. **Parallel Scaling Overhead (Track D)**: Python's GIL serializes CPU-bound pure Python threads, and Windows process `spawn` overhead caused an 80% slowdown for small tasks. Using NumPy C-extensions that release the GIL was essential to achieve $2.8\times$ multi-core speedup.

### Question 6: What evidence demonstrates that your results are correct and reproducible?
**Answer**:
1. **Automated Test Suite**: 151 independent unit tests across 10 tracks run via `py -3.12 -m pytest -v`, all passing with 100% success.
2. **Output Equivalence & Ground Truth**:
   - Compiler Track F: Interpreted outputs of unoptimized and optimized code match identically.
   - DBMS Track H: Query result hashes are bit-for-bit identical before and after index and join optimization.
   - Discrete Track B: Dynamic Programming, Branch-and-Bound, and ILP all converged to the exact same value ($V=280$).
   - Continuous Track A: MLP backpropagation gradients match finite-difference numerical gradients within $8.94 \times 10^{-11}$.
3. **Seed Control**: All stochastic experiments (Simulated Annealing, random baselines, mini-batch sampling) fix random seeds (`random_state=42`).
4. **Reproducible Scripts**: Every track has a dedicated `run_*_benchmarks.py` script that outputs raw CSV metrics and reproducible plots.

### Question 7: How could the optimization approach be adapted if the scale, data, constraints, or hardware changed?
**Answer**:
- *Scale Explosion (EV Fleet)*: If the fleet expands from 10 to 10,000 EVs, full centralized LP will encounter memory limits. The problem decomposes naturally using **Lagrangian Relaxation** or **ADMM (Alternating Direction Method of Multipliers)**, where the grid transformer capacity is dualized into a shadow price signal, allowing each EV to solve an independent local optimization.
- *Real-time Data Uncertainty*: If vehicle departure times or solar generation fluctuate stochastically, the deterministic LP adapts into a **Rolling-Horizon Model Predictive Control (MPC)** framework, re-solving an updated 4-hour window every 15 minutes.
- *Discrete Optimization at Scale*: For large instances where exact ILP times out, Chvátal's Greedy heuristic or Metaheuristics (Simulated Annealing / Genetic Algorithms) provide scalable near-optimal solutions.

### Question 8: For the compiler, xv6, and DBMS tracks, which optimization did you implement, what OS or database concept did you learn, and how was correctness preserved?
**Answer**:
- **Compiler (Track F)**: Implemented Constant Folding, DCE, CSE, LICM, and ILP Register Allocation. Learned that liveness analysis must construct precise interference graphs, and variables with overlapping live ranges cannot share registers. Correctness was verified by executing both AST representations against a test vector suite and asserting identical output.
- **xv6 OS (Track G)**: Implemented Stride Scheduling and Buffer Cache profiling syscalls. Learned that stride scheduling enforces exact proportional CPU sharing, but dynamically waking processes require pass reassignment to avoid monopolizing the CPU. Correctness was verified because xv6 successfully booted in QEMU, passed `usertests`, and preserved process isolation.
- **DBMS (Track H)**: Implemented 0-1 Knapsack ILP Index Selection and System-R Dynamic Programming for Join Ordering. Learned that relational query optimizers must balance disk I/O reduction against index maintenance storage overhead, and join trees must preserve relational algebra equivalences. Correctness was verified by computing SHA-256 hashes of query result sets across all 8 benchmark queries, ensuring zero row discrepancy.

### Question 9: What advantages and limits of QML, quantum algorithms, and parallel computing did your experiments reveal compared with classical computing?
**Answer**:
- **Quantum Algorithms (Grover & Shor)**: Offer provable asymptotic complexity speedups ($O(\sqrt{N})$ vs $O(N)$ for search, polynomial vs sub-exponential for factoring). However, on small problem sizes ($N \le 64$), classical linear search is thousands of times faster in wall-clock time due to quantum simulation overhead.
- **NISQ Noise & Barren Plateaus**: Hardware noise (two-qubit depolarizing error $>1\%$) erodes quantum fidelity exponentially with depth. Furthermore, gradient variance vanishes exponentially with qubit count ($\text{Var} \propto 2^{-cN}$), making deep QNNs untrainable without specialized local observables or identity-block initialization.
- **Quantum Machine Learning (QSVM)**: Successfully mapped non-linear data into quantum Hilbert space, matching classical RBF SVM accuracy (85.7%). However, classical SVM trained in 2 milliseconds, whereas computing the quantum fidelity kernel required $O(N^2)$ circuit evaluations.
- **Parallel Computing (Track D)**: Offers immediate, deterministic wall-clock speedups ($2.84\times$ on 4 CPU cores) for tensor operations, but is fundamentally bounded by Amdahl's serial fraction ($s = 0.12$) and communication/synchronization overhead.

### Question 10: Which AI tools did you use, how did their outputs differ, and which AI-generated errors did you find and fix?
**Answer**:
- **AI Tools Used**: Google DeepMind **Antigravity (Gemini 3.8 Flash)** for planning, test harness generation, algorithm implementation, and debugging; Anthropic **Claude 3.5/3.7 Sonnet** for dual-AI formulation evaluation.
- **Differences in Outputs**: Antigravity generated complete, runnable project architectures with integrated unit tests and zero-dependency scripts. Claude provided detailed analytical derivations of penalty multiplier lower bounds ($P_{\text{crit}}$).
- **AI-Generated Errors Caught and Fixed**:
  1. *Failure 01 (Track A)*: Dependency assumption on `tabulate` & relative import crashes. Fixed by writing native pure-Python Markdown generators and inserting `REPO_ROOT` into `sys.path`.
  2. *Failure 02 (Track A)*: Confirmation bias in text claiming Adam outperformed Momentum on Rosenbrock, contradicting measured numbers. Fixed by rewriting analysis to reflect empirical data truthfully.
  3. *Failure 03 (Track A)*: Inverted eigenvector column slice in Low-Rank Curvature Approximation causing $10^{58}$ explosion. Fixed by correcting the projection slice to `sorted_vecs[:, :r]`.
  4. *Failure 04 (Track C)*: Measuring an exact eigenstate in finite-shot quantum scaling yielding identically zero variance. Fixed by replacing with a superposition state $|\psi\rangle = R_y(\pi/3)|0\rangle$.
  5. *Failure 05 (Track D)*: Windows `spawn` process latency causing an 80% slowdown in parallel scaling. Fixed by switching to `ThreadPoolExecutor` with GIL-releasing NumPy BLAS matrix operations.
  6. *Failure 06 (Track I)*: Grover over-rotation on $N=4$ due to small-angle asymptotic formula approximation. Fixed by deriving exact trigonometric iteration count $R = \text{round}(\frac{\pi}{4\theta} - 0.5)$.
  7. *Failure 07 (Track J)*: Test suite minimization solely on code line coverage dropping mutation score from 100% to 86.7%. Fixed by stacking the mutation kill matrix into the constraint system.

---

## 4. Demonstration Guide for Viva Voce

Students can demonstrate the entire project live to examiners using the following sequence:

1. **Test Suite Verification**:
   ```powershell
   py -3.12 -m pytest -v
   ```
   *Viva Point*: Shows all 151 unit tests across Tracks A–J passing in ~15 seconds.
2. **Main Case Study Demonstration (Track E)**:
   ```powershell
   py -3.12 track_E_case_study/src/run_ev_benchmarks.py
   ```
   *Viva Point*: Show the terminal output showing 67.0% cost reduction and open `ev_fleet_charging_profiles.png` to explain peak-shaving and valley-filling.
3. **xv6 Kernel Scheduler (Track G)**:
   ```powershell
   # In WSL Ubuntu:
   wsl -d Ubuntu -- bash -c "cd /mnt/c/dev/ot-8/track_G_xv6/xv6-riscv && make qemu"
   ```
   *Viva Point*: Run `schedtest` in QEMU to demonstrate the 2:1 proportional stride execution between tickets 20 and 10.
4. **Software Engineering Test Suite Minimization (Track J)**:
   ```powershell
   py -3.12 track_J_software_engineering/src/run_track_j_benchmarks.py
   ```
   *Viva Point*: Show 0-1 ILP eliminating 27 redundant tests (1.56x speedup) with 100% statement coverage and 100% mutation score.
5. **AI Vibe-Coding Audit Trail**:
   Open [`AI_Prompt_Log.pdf`](file:///c:/dev/ot-8/AI_Prompt_Log.pdf) and [`AI_FAILURE_LOG.md`](file:///c:/dev/ot-8/AI_FAILURE_LOG.md) to walk through genuine debugging records and lessons learned.
