# Prompt Record 11: Track C Batch 3 — PQCs, QNNs, Quantum Gradients, Noise & Barren Plateaus

- **Prompt ID**: `P-TRC-03`
- **Date & Time**: 2026-10-21 20:08:24 IST
- **Track**: Track C (Quantum Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending commit

---

### Purpose
Complete all remaining requirements of Track C (Quantum Optimization):
1. **PQCs and QNNs**: Implement non-linear quantum feature maps (`ZZFeatureMap`), variational quantum classifiers (`QuantumClassifierPQC`), parameter-shift training, and a fair classical baseline (`ClassicalMLPClassifier`) on a concentric circles dataset.
2. **Quantum Gradients**: Implement and test the Parameter-Shift Rule, Simultaneous Perturbation Stochastic Approximation (SPSA), Quantum Natural Gradient (QNG via Fubini-Study metric / QFIM), and central finite-difference references.
3. **Finite-Shot & Hardware Noise Models**: Evaluate finite-shot variance across 100, 1,000, and 10,000 shots (verifying $\sigma \propto 1/\sqrt{M}$), and assess depolarizing and readout noise models on Qiskit Aer.
4. **Barren Plateaus**: Empirically investigate gradient variance decay across qubit counts from $N=2$ to $10$, verifying exponential flattening.
5. **Testing & Audit**: Add independent unit tests, generate multi-panel comparison plots and Markdown summaries, update checklists and prompt logs, and recompile `AI_Prompt_Log.pdf`.

---

### Original Prompt
```text
Continue Optimization Techniques Lab Assignment 8 in C:\dev\ot-8.

Complete all remaining work for Track C: Quantum Optimization. Track C Batch 1 (QAOA) and Batch 2 (VQE) are already implemented. Inspect the existing code, tests, results, CHAPTER_6_CHECKLIST.md, PROMPT_LOG.md, and AI_FAILURE_LOG.md first. Preserve working implementations and avoid duplication.

Complete these remaining topics in a logical order:

1. Parameterized Quantum Circuits (PQCs) and Quantum Neural Networks (QNNs): implement parameterized circuits, a small QNN for a defined learning task, and a fair classical baseline.
2. Quantum gradients: implement and test the parameter-shift rule, SPSA, and quantum natural gradient where feasible. Compare gradient estimates with numerical or analytical references on small circuits.
3. Finite-shot and noise experiments: compare 100, 1,000, and 10,000 shots; evaluate suitable depolarizing and readout noise models; investigate barren-plateau behavior across qubit counts from 2 to 10 where computationally feasible.

Requirements:
- Reuse the existing Qiskit/Aer infrastructure and match installed package APIs.
- Include mathematical explanations, unit tests, reproducible experiments, fixed seeds, and clear limitations.
- Save actual measurements in CSV, summaries in Markdown, and useful plots.
- Compare quantum methods against suitable classical baselines without claiming quantum advantage unless rigorously demonstrated.
- Run experiments within reasonable local runtime and document any scope that cannot be completed reliably.
- Save the exact prompt and update PROMPT_LOG.md, CHAPTER_6_CHECKLIST.md, and AI_Prompt_Log.pdf according to the existing project conventions.
- Update AI_FAILURE_LOG.md only when genuine failures occur; never fabricate failure entries.
- Run py -3.12 -m pytest -v and report the actual results.
- Work in manageable internal stages, verifying each stage before proceeding.
- Preserve unrelated changes and inspect Git status before editing.
- Commit completed local changes with a message starting [AI:Antigravity].
- Do not create a GitHub repository, configure a remote, push, publish, or upload anything. Do not switch branches or rewrite history without asking.

At the end, summarize files changed, commit hash, tests, experiment results, limitations, and any incomplete requirements. Stop when Track C is complete.
```

---

### AI Response Summary
- Created `track_C_quantum/src/quantum_gradients.py`:
  - `parameter_shift_gradient`: Exact hardware-compatible gradient via $\pm \pi/2$ macro-shifts.
  - `spsa_gradient`: Evaluates full $D$-dimensional gradient in only 2 circuit evaluations via Rademacher perturbation.
  - `quantum_fisher_information_matrix`: Computes exact Fubini-Study Riemannian metric tensor $\mathcal{F}$.
  - `quantum_natural_gradient_step`: Levenberg-Marquardt regularized natural gradient step $(\mathcal{F} + \lambda I)^{-1} \mathbf{g}$.
- Created `track_C_quantum/src/pqc_qnn.py`:
  - `ZZFeatureMap` & `AngleFeatureMap`: Embeds classical inputs via single-qubit and two-qubit entangling gates.
  - `QuantumClassifierPQC`: 2-qubit QNN trained via the parameter-shift rule on concentric circles dataset.
  - `ClassicalMLPClassifier`: Pure NumPy 2-layer MLP baseline with matching parameter budget.
- Created `track_C_quantum/src/quantum_noise_experiments.py`:
  - `evaluate_finite_shot_variance`: Measures variance scaling across 100, 1,000, and 10,000 shots.
  - `build_depolarizing_noise_model` & `build_readout_noise_model`: Simulates NISQ physical errors.
  - `run_barren_plateaus_study`: Measures empirical gradient variance decay across $N \in [2..10]$ qubits.
- Created unit tests:
  - `track_C_quantum/tests/test_quantum_gradients.py` (5 tests passed).
  - `track_C_quantum/tests/test_pqc_qnn.py` (7 tests passed).
  - `track_C_quantum/tests/test_quantum_noise.py` (7 tests passed).
- Created benchmark runner `track_C_quantum/experiments/run_quantum_advanced_benchmarks.py`:
  - Produced `quantum_advanced_metrics.csv`, `quantum_advanced_metrics.md`, and 5-panel plot `quantum_advanced_benchmarks.png`.
- Updated `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and recompiled `AI_Prompt_Log.pdf`.

---

### Verification & Testing
- Full Test Suite: `py -3.12 -m pytest -v` (87/87 passed across Tracks A, B, and C).
- Track C Suite: 40 tests passed in 5.94s.
