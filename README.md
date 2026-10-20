# Optimization Techniques Lab Assignment 8: Optimize Anything
### The 14-Day Vibe-Coding Challenge
*Discover a process. Formulate the problem. Apply optimization. Prove the improvement.*

---

## 1. Project Overview
This repository contains the complete 10-track optimization portfolio for **Optimization Techniques Lab Assignment 8**. Every track implements a concrete optimization problem, compares an unoptimized baseline against Chapter 6 methods, provides independent verification tests, and maintains a strict audit trail of AI-assisted development (vibe coding).

### Compulsory Tracks (A to J)
- **Track A — Continuous Optimization**: First-order (GD, SGD, Momentum, NAG, AdaGrad, RMSProp, Adam) and second-order (Newton, Gauss-Newton, curvature approximations, MLP backpropagation) over convex and non-convex landscapes.
- **Track B — Discrete Optimization**: Combinatorial optimization (Greedy, ILP, Branch-and-Bound, Dynamic Programming) and QUBO/Ising formulation with penalty terms for Max-Cut.
- **Track C — Quantum Optimization**: QAOA, VQE, PQCs, parameter-shift rule, SPSA, Quantum Natural Gradient, finite-shot variance, and barren plateaus study.
- **Track D — Submodular, Parallel & Distributed Optimization**: Diminishing returns greedy selection, multi-core parallel speedup (Amdahl & Gustafson laws), Federated Learning (FedAvg), and Multi-task optimization (MGDA & GradNorm).
- **Track E — Main Real-Process Case Study**: Joint EV Fleet Smart Charging Scheduling & Delivery Route Optimization.
- **Track F — Mini-Compiler Optimization**: Toy language compiler with 4 optimization passes (Constant Folding/Propagation, DCE, CSE, LICM) and ILP-based Register Allocation.
- **Track G — xv6 Operating-System Optimization**: Priority/Stride process scheduler and runtime profiling syscall under QEMU on xv6-riscv.
- **Track H — DBMS Optimization**: Automated Physical Design (0-1 Knapsack/ILP index selection) and Join-Order Dynamic Programming on TPC-H SF 0.1 schema.
- **Track I — Quantum Algorithms & QML Advantage**: Deutsch-Jozsa, Bernstein-Vazirani, Grover ($O(\sqrt{N})$ scaling vs linear search), QFT, QPE, Shor (15), and Variational Quantum Classifier vs Classical SVM.
- **Track J — Software-Engineering Optimization**: Regression Test Suite Minimization & Prioritization on a >1,000 LOC codebase.

---

## 2. Environment Setup (Windows 11 & PowerShell)

### Prerequisites
- Python 3.12 (or 3.11)
- Git for Windows
- Windows Subsystem for Linux (WSL2 Ubuntu) for xv6 RISC-V compilation

### Python Environment Setup
```powershell
# 1. Clone repository (or navigate to workspace)
cd c:\dev\ot-8

# 2. Install Python dependencies
py -3.12 -m pip install -r requirements.txt
```

### xv6 Toolchain Setup (Inside WSL2 Ubuntu)
For **Track G (xv6 OS)**, execute the following single command in PowerShell to prepare the RISC-V QEMU toolchain inside Ubuntu:
```powershell
wsl -d Ubuntu -- sudo apt update && wsl -d Ubuntu -- sudo apt install -y gcc-riscv64-unknown-elf qemu-system-misc build-essential git
```

---

## 3. Strict Vibe-Coding Documentation & Audit Trail

Per Section 8 of the assignment specification:
- **Master Prompt Index**: [`PROMPT_LOG.md`](PROMPT_LOG.md) lists every single interaction chronologically.
- **Compiled PDF**: [`AI_Prompt_Log.pdf`](AI_Prompt_Log.pdf) is automatically compiled via ReportLab from all Markdown files in `prompts/`.
- **AI Tools Register**: [`AI_TOOLS_REGISTER.md`](AI_TOOLS_REGISTER.md) lists tools (Antigravity & Claude), dates, and tasks.
- **Failure Log**: [`AI_FAILURE_LOG.md`](AI_FAILURE_LOG.md) documents genuine AI errors, independent tests that caught them, fixes, and lessons.
- **Chapter 6 Traceability Matrix**: [`CHAPTER_6_CHECKLIST.md`](CHAPTER_6_CHECKLIST.md) maps every required syllabus technique to its code and results.

To recompile the master prompt PDF at any time:
```powershell
py -3.12 scripts/generate_prompt_pdf.py
```

---

## 4. AI Tools Register Summary

| Tool Name | Model | Purpose | Dates Used |
| :--- | :--- | :--- | :--- |
| **Antigravity** | Gemini 3.8 Flash | Planning, implementation, testing, debugging, docs | 2026-10-20 to Present |
| **Claude** | Claude 3.5/3.7 Sonnet | Tool comparison on Track B QUBO formulation | 2026-10-20 to Present |

---

## 5. Verification & Testing
Every track includes an automated test suite under `tests/` asserting mathematical correctness and baseline comparison:
```powershell
# Run all unit and convergence tests
py -3.12 -m pytest
```
