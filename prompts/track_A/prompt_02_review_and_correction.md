# Prompt Record 04: Track A — Empirical Audit, Benchmark Findings Correction & Failure Log 02

- **Prompt ID**: `P-TRA-02`
- **Date & Time**: 2026-10-20 20:36:26 IST
- **Track**: Track A (Continuous Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: `1f09854`

---

### Purpose
Review Track A Stage 1 benchmark documentation against the actual measured outputs in `track_A_continuous/results/first_order_metrics.md`. Correct inaccurate AI-generated claims regarding Rosenbrock optimization results, add rigorous explanations on the dependence of optimizer performance on learning rate and landscape geometry, ensure all textual conclusions match measured values, document Failure Record 02 in `AI_FAILURE_LOG.md`, update `PROMPT_LOG.md`, recompile `AI_Prompt_Log.pdf`, and create a local Git commit with prefix `[AI:Antigravity]`.

---

### Original Prompt
```text
Before Track A Stage 2, review the Track A Stage 1 benchmark documentation against the actual results in `track_A_continuous/results/first_order_metrics.md`.

1. Correct any inaccurate claims in the experiment findings, especially the Rosenbrock results.
2. Explain that optimizer performance depends on learning rate and objective function.
3. Ensure all reported conclusions match measured values.
4. Do not change benchmark results unless you rerun the experiments and identify a genuine issue.
5. Update the relevant documentation and prompt log, regenerate `AI_Prompt_Log.pdf` if needed, and create a local Git commit using the `[AI:Antigravity]` prefix.
6. Do not configure a remote or push to GitHub.

Afterward, report the exact files changed and the verification commands executed.
```

---

### AI Response Summary
1. Conducted an audit of `track_A_continuous/results/first_order_metrics.md` against the generated tabular values.
2. Identified an AI hallucination in Key Finding 2, which previously claimed *"Adam and RMSProp navigate the curved banana valley efficiently, achieving orders-of-magnitude lower final loss than GD."*
3. Confirmed from raw empirical data that **Momentum** ($0.00558$) and **NAG** ($0.00750$) actually achieved nearly $500\times$ lower loss than Vanilla GD ($2.726$), whereas **Adam** ($1.803$) and **AdaGrad** ($1.478$) achieved only modest improvements under un-tuned default hyperparameters.
4. Corrected the text in `first_order_metrics.md` and `run_first_order_benchmarks.py` to truthfully reflect measured metrics and explained the theoretical rationale (momentum accumulates directional kinetic energy along curved valleys, while adaptive second-moment scaling without tuning takes overly conservative steps).
5. Documented **Failure Record 02** in `AI_FAILURE_LOG.md` detailing the discrepancy, confirmation bias risk, verification method, fix, and viva takeaways.
6. Recompiled `AI_Prompt_Log.pdf` with ReportLab and committed the corrections locally under `[AI:Antigravity]`.

---

### Files Created / Modified
- [`prompts/track_A/prompt_02_review_and_correction.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_02_review_and_correction.md)
- [`PROMPT_LOG.md`](file:///c:/dev/ot-8/PROMPT_LOG.md)
- [`track_A_continuous/results/first_order_metrics.md`](file:///c:/dev/ot-8/track_A_continuous/results/first_order_metrics.md)
- [`track_A_continuous/experiments/run_first_order_benchmarks.py`](file:///c:/dev/ot-8/track_A_continuous/experiments/run_first_order_benchmarks.py)
- [`AI_FAILURE_LOG.md`](file:///c:/dev/ot-8/AI_FAILURE_LOG.md)
- [`AI_Prompt_Log.pdf`](file:///c:/dev/ot-8/AI_Prompt_Log.pdf)

---

### Verification Note
- All textual claims now precisely match the measured metrics in `first_order_metrics.csv` and `first_order_metrics.md`.
- No benchmark numerical data was fabricated or altered.
- `AI_Prompt_Log.pdf` successfully recompiled.

---

### Human Revisions / Instructions Received
Audit and corrections accepted; ready to proceed to Track A Stage 2 upon approval.
