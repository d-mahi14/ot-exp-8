# AI Failure & Correction Log
**Assignment 8 Requirement**: Document at least three genuine cases where AI-generated code was wrong, unsafe, inefficient, or misleading, along with the independent verification test that caught it, the fix applied, and the lesson learned.

---

## Failure Entry Template
```markdown
### Failure Record [01 | 02 | 03]: <Descriptive Title>
- **Track**: Track X (<Track Name>)
- **Date**: YYYY-MM-DD
- **AI Tool & Model**: <Tool Name> (<Model Version>)
- **Prompt Reference**: [Link to prompt log](file:///path/to/prompt)
- **The AI Mistake**: Describe the erroneous, unsafe, or hallucinated code.
- **Why It Failed / Subtle Bug**: Root cause analysis.
- **Verification Method (How It Was Caught)**: Unit test, compiler error, kernel panic, or analytical check.
- **The Fix**: Code diff or corrected logic.
- **Viva Lesson Learned**: What this teaches about relying on AI for optimization and systems code.
```

---

## Active Failure Records

*(This log will record at least three genuine failures encountered during implementation of Tracks A–J. Zero fabricated failures are permitted.)*

### Failure Record 01: Unintended Third-Party Dependency & Relative Import Resolution
- **Track**: Track A (Continuous Optimization)
- **Date**: 2026-10-20
- **AI Tool & Model**: Antigravity (Gemini 3.8 Flash)
- **Prompt Reference**: [`prompts/track_A/prompt_01_continuous_optimization.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_01_continuous_optimization.md)
- **The AI Mistake**:
  1. The benchmark script invoked `df.to_markdown(index=False)` assuming `tabulate` was part of standard `pandas`, throwing `ImportError: 'Import tabulate' failed`.
  2. The script also relied on direct execution from arbitrary directories without adding the repository root to `sys.path`, causing `ModuleNotFoundError: No module named 'track_A_continuous'`.
- **Why It Failed / Root Cause**: The model assumed optional pandas dependencies were globally installed and overlooked standalone CLI script execution semantics where `PYTHONPATH` isn't preset.
- **Verification Method (How It Was Caught)**:
  Executing `py -3.12 track_A_continuous/experiments/run_first_order_benchmarks.py` immediately crashed at runtime with the unhandled `ImportError`.
- **The Fix**:
  1. Replaced the `to_markdown()` call with a native, zero-dependency Markdown table generator in pure Python.
  2. Dynamically resolved `REPO_ROOT = Path(__file__).resolve().parents[2]` and inserted it into `sys.path[0]`.
- **Viva Lesson Learned**: Never rely on optional, convenience-wrapper third-party packages in automated evaluation scripts when standard library constructs provide the exact same functionality deterministically. Always enforce robust import path resolution for reproducible benchmarking.

---

### Failure Record 02: Discrepancy Between Measured Data and Textual Analysis (Hallucinated Adam Superiority on Rosenbrock)
- **Track**: Track A (Continuous Optimization)
- **Date**: 2026-10-20
- **AI Tool & Model**: Antigravity (Gemini 3.8 Flash)
- **Prompt Reference**: [`prompts/track_A/prompt_01_continuous_optimization.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_01_continuous_optimization.md)
- **The AI Mistake**:
  The summary section of `first_order_metrics.md` claimed: *"Adam and RMSProp navigate the curved banana valley efficiently, achieving orders-of-magnitude lower final loss than GD."*
  However, the actual experiment output in the table showed:
  - Vanilla GD: Final Loss = `2.726`
  - Adam ($\eta=0.02$): Final Loss = `1.803` (only modest improvement, NOT orders-of-magnitude)
  - Momentum ($\gamma=0.9$): Final Loss = `0.00558` (nearly $500\times$ lower than GD)
  - NAG ($\gamma=0.9$): Final Loss = `0.00750` (over $360\times$ lower than GD)
- **Why It Failed / Root Cause**:
  The AI relied on a common generic heuristic assumption ("Adam is universally better on complex non-convex landscapes") rather than strictly cross-referencing its own generated empirical metrics table. On Rosenbrock's parabolic valley, momentum-based methods retain directional kinetic energy along the narrow canyon floor, whereas un-tuned adaptive moment scaling in Adam can take timid steps along the bottom without specialized hyperparameter tuning.
- **Verification Method (How It Was Caught)**:
  Direct human and programmatic cross-verification of the markdown table column `Final Loss` against the textual bullet points.
- **The Fix**:
  1. Rewrote the experimental findings in `track_A_continuous/results/first_order_metrics.md` and `run_first_order_benchmarks.py` to truthfully report that Momentum and NAG achieved orders-of-magnitude lower loss ($0.00558$ and $0.00750$), while Adam and AdaGrad made only modest progress ($1.803$ and $1.478$).
  2. Added an explicit discussion highlighting that optimizer performance is fundamentally dependent on learning rate tuning and objective function geometry.
- **Viva Lesson Learned**:
  AI models frequently suffer from "confirmation bias" toward popular algorithms (like Adam) and will hallucinate claims that contradict their own generated data unless strictly checked against measured numbers. An honest negative or counter-intuitive result (e.g., standard momentum beating Adam on Rosenbrock) is scientifically valuable and demonstrates critical understanding.

---

### Failure Record 03: Inverted Eigenvector Indexing in Low-Rank Curvature Approximation
- **Track**: Track A (Continuous Optimization — Second-Order Methods)
- **Date**: 2026-10-20
- **AI Tool & Model**: Antigravity (Gemini 3.8 Flash)
- **Prompt Reference**: [`prompts/track_A/prompt_03_second_order_and_backprop.md`](file:///c:/dev/ot-8/prompts/track_A/prompt_03_second_order_and_backprop.md)
- **The AI Mistake**:
  In `LowRankCurvatureApproximation`, after sorting the eigenvalues and eigenvectors descending via `idx = np.argsort(eigvals)[::-1]`, the code executed:
  ```python
  sorted_vals = eigvals[idx]
  sorted_vecs = eigvecs[:, idx]  # Already sorted!

  top_vals = sorted_vals[:r]
  top_vecs = sorted_vecs[:, idx[:r]]  # Redundant re-indexing by idx!
```
  This redundant index operation re-indexed into an already-sorted matrix, pairing the largest eigenvalue $\lambda=10$ with the eigenvector of the smallest eigenvalue $\lambda=1$, and vice-versa.
- **Why It Failed / Root Cause**:
  Along the steep coordinate ($x_0$, eigenvalue 10), the step divided by 1.0 rather than 10.0, applying a step size 10 times too large ($x_{t+1} = x_t - 30$). Along the shallow coordinate ($x_1$, eigenvalue 1), the step divided by 10.0 rather than 1.0. This caused catastrophic exponential divergence ($10^{58}$ within 30 iterations) on the quadratic bowl.
- **Verification Method (How It Was Caught)**:
  Caught immediately when inspecting the generated metrics table `second_order_metrics.md`, where the final loss for `Low-Rank Curvature` was measured at `8.08e+58` instead of converging.
- **The Fix**:
  Corrected the column slice to `top_vecs = sorted_vecs[:, :r]`. Additionally, implemented robust estimation of the background curvature $\sigma^2$ from the average of discarded trailing eigenvalues $\frac{1}{d-r}\sum_{k=r+1}^d \lambda_k$ rather than dividing by raw damping.
- **Viva Lesson Learned**:
  When manipulating spectral decompositions (eigenspaces, SVD, PCA), subtle indexing mistakes can silently invert coordinate projections. Never assume second-order or approximation code works without verifying both the mathematical derivation and the empirical convergence numbers on controlled benchmarks.

---

### Log Status
- [x] **Failure 1**: Documented above (Track A: Script import resolution & dependency assumption).
- [x] **Failure 2**: Documented above (Track A: Discrepancy between empirical metrics and textual analysis).
- [x] **Failure 3**: Documented above (Track A: Inverted eigenvector indexing in Low-Rank Curvature).
