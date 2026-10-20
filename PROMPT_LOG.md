# Master AI Prompt Log Index
**Course**: Optimization Techniques (Lab Assignment 8 — The 14-Day Vibe-Coding Challenge)  
**Student Project**: Optimize Anything — 10-Track Optimization Portfolio  
**Repository**: `ot-8`  
**Generated PDF Document**: [`AI_Prompt_Log.pdf`](file:///c:/dev/ot-8/AI_Prompt_Log.pdf)

---

## 1. Prompt Logging Policy & Rules
1. **Verbatim Preservation**: Every single user prompt is recorded exactly as entered.
2. **AI Disclosure**: Every use of AI (model, tool, date, purpose) is disclosed per Section 8.
3. **Commit Traceability**: AI-assisted commits use the prefix `[AI:<tool-name>]`, while human manual corrections use `[Manual:Fix]`.
4. **Independent Verification**: No experiment or claim is accepted without reproducible tests and verified metric outputs.
5. **No Secrets**: No API keys, passwords, or credentials may ever be saved in logs.

---

## 2. Master Prompt Registry

| Prompt ID | Date & Time (IST) | Track | Purpose | Markdown Log Link | Key Files Changed / Created | Verification Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `P-GEN-01` | 2026-10-20 20:12 | General | Initial Project Planning, Chapter 6 Syllabus Mapping, and Schedule Formulation | [`prompt_01_planning_and_architecture.md`](file:///c:/dev/ot-8/prompts/general/prompt_01_planning_and_architecture.md) | None (read-only system inspection) | Passed (Plan approved) |
| `P-GEN-02` | 2026-10-20 20:16 | General | Architectural Confirmation, xv6 Toolchain Diagnostics, Prompt Log Infrastructure & PDF Pipeline Setup | [`prompt_02_logging_and_approval.md`](file:///c:/dev/ot-8/prompts/general/prompt_02_logging_and_approval.md) | `PROMPT_LOG.md`, `AI_TOOLS_REGISTER.md`, `AI_FAILURE_LOG.md`, `scripts/generate_prompt_pdf.py`, `AI_Prompt_Log.pdf` | Passed (PDF generated & verified) |
| `P-TRA-01` | 2026-10-20 20:25 | Track A | Stage 1 First-Order Continuous Optimizers: GD, SGD, Mini-batch, Momentum, NAG, AdaGrad, RMSProp, Adam | [`prompt_01_continuous_optimization.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_01_continuous_optimization.md) | `track_A_continuous/src/`, `track_A_continuous/tests/`, `track_A_continuous/experiments/`, `track_A_continuous/results/` | Passed (13/13 unit tests passed, metrics & plots generated) |
| `P-TRA-02` | 2026-10-20 20:36 | Track A | Empirical Audit, Benchmark Findings Correction, Hyperparameter Sensitivity Rationale & Failure Log 02 | [`prompt_02_review_and_correction.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_02_review_and_correction.md) | `first_order_metrics.md`, `run_first_order_benchmarks.py`, `AI_FAILURE_LOG.md`, `AI_Prompt_Log.pdf` | Passed (Discrepancy resolved, Failure 02 recorded) |

---

## 3. Directory Structure of Prompt Logs
```text
prompts/
├── general/
│   ├── prompt_01_planning_and_architecture.md
│   └── prompt_02_logging_and_approval.md
├── track_A/        # Continuous Optimization
├── track_B/        # Discrete & QUBO Optimization
├── track_C/        # Quantum Optimization (QAOA, VQE)
├── track_D/        # Submodular, Parallel & Distributed
├── track_E/        # Main Case Study (EV Fleet Smart Charging)
├── track_F/        # Mini-Compiler Optimization
├── track_G/        # xv6 Operating System
├── track_H/        # DBMS Optimization
├── track_I/        # Quantum Algorithms & QML Advantage
└── track_J/        # Software Engineering Optimization
```

---

## 4. Re-Generating the Master PDF
To re-compile all Markdown logs into the official submission PDF `AI_Prompt_Log.pdf`, run:
```powershell
py -3.12 scripts/generate_prompt_pdf.py
```
