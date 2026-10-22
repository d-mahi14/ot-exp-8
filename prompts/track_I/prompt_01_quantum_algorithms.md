# Prompt Log: Track I — Prompt 1

**Timestamp:** 2026-10-22T19:40:53+05:30  
**Prompt ID:** `P-TRI-01`  
**Track:** Track I — Quantum Algorithms and Quantum Machine Learning  
**Target:** Implementation of canonical quantum algorithms (Deutsch-Jozsa, Bernstein-Vazirani, QFT, QPE, Shor's period-finding for N=15, Grover's search), NISQ hardware noise modeling, and Quantum Machine Learning (Fidelity Quantum Kernel Estimation / QSVM vs Classical RBF and Linear SVM).

---

### User Prompt

```text
Continue Assignment 8 in `C:\dev\ot-8`. Complete Track I: Quantum Algorithms and Quantum Machine Learning.

Track C already covers QAOA, VQE, PQCs/QNNs, quantum gradients, finite-shot measurements, noise models, and barren-plateau experiments. Inspect existing implementations and reuse them where appropriate. Track I must add distinct work rather than simply repeat Track C.

### Required work

1. Implement and explain suitable additional quantum algorithms, such as Grover's search and the Quantum Fourier Transform, using Qiskit or another suitable simulator. Include correctness tests and small examples.
2. Where practical, evaluate an algorithm on an actual quantum device or documented hardware-noise model. Clearly distinguish simulator, noisy simulator, and real hardware results.
3. Compare a quantum algorithm against an appropriate classical baseline on the same problem instances.
4. Explain the algorithm's complexity, assumptions, resource requirements, and the problem sizes where it is useful.
5. Include a QML comparison using the existing QNN where suitable, or add a distinct quantum-kernel or quantum-classifier experiment. Compare against a classical model on the same dataset, preprocessing, and train/test split.
6. Discuss algorithmic speedup versus observed runtime. Include circuit evaluation counts, qubit requirements, shot budgets, or other relevant costs. Do not infer quantum advantage from small simulator experiments.

### Experiments and verification
Create reproducible scripts, fixed seeds, unit tests, CSV measurements, Markdown summaries, and plots where appropriate. Independently verify outputs for small instances and document limitations. Do not claim hardware execution unless it actually occurred.

Save the exact prompt under `prompts/track_I/`. Update `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and `AI_Prompt_Log.pdf`. Add only genuine failures to `AI_FAILURE_LOG.md`.

Run `py -3.12 -m pytest -v` and report the actual test results.

Commit local changes with a message beginning `[AI:Antigravity]`. Do not create a GitHub repository, configure a remote, push, or publish anything. Do not change branches or rewrite history without asking.

Stop after Track I and summarize the new algorithms, QML comparison, classical baselines, measured evidence, tests, limitations, and commit hash.
```
