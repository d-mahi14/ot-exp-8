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

### Failure Record 04: Observable Eigenstate Zero-Variance in Finite-Shot Quantum Scaling Experiments
- **Track**: Track C (Quantum Optimization)
- **Date**: 2026-10-21
- **AI Tool & Model**: Antigravity (Gemini 3.8 Flash)
- **Prompt Reference**: [`prompts/track_C/prompt_03_qnn_gradients_and_noise.md`](file:///c:/dev/ot-8/prompts/track_C/prompt_03_qnn_gradients_and_noise.md)
- **The AI Mistake**:
  In `run_quantum_advanced_benchmarks.py`, the finite-shot scaling benchmark was initially configured to evaluate the observable $H = Z_0 Z_1$ on the Bell state $|\Phi^+\rangle = (|00\rangle + |11\rangle)/\sqrt{2}$. Because $|\Phi^+\rangle$ is an exact eigenstate of $Z_0 Z_1$ with eigenvalue $+1$, every projective measurement deterministically collapsed to $+1$. The empirical sample variance across all trials was identically zero ($\sigma_M^2 = 0.00$), producing a runtime warning in matplotlib (`UserWarning: Data has no positive values, and therefore cannot be log-scaled`) and failing to exhibit the Central Limit Theorem shot-noise scaling.
- **Why It Failed / Subtle Bug**:
  The AI overlooked basic quantum projective measurement physics: if $|\psi\rangle$ is an eigenstate of observable $O$ ($O|\psi\rangle = \lambda |\psi\rangle$), the measurement outcome is non-stochastic ($\Delta O = 0$). Projective quantum shot noise only manifests when $|\psi\rangle$ is a superposition across distinct eigenspaces.
- **Verification Method (How It Was Caught)**:
  Caught during terminal execution when `run_quantum_advanced_benchmarks.py` produced standard deviation outputs `Std Error: 0.0000` for all shot counts $M \in \{100, 1000, 10000\}$ and triggered matplotlib log-scale warnings.
- **The Fix**:
  Replaced the eigenstate with a non-eigenstate superposition state $|\psi\rangle = R_y(\pi/3)|0\rangle$ on observable $Z_0$, where $\langle Z_0 \rangle = \cos(\pi/3) = 0.5$ and the single-shot variance is $1 - 0.5^2 = 0.75 > 0$. This produced clean, empirical variance scaling matching $\sigma \propto 1/\sqrt{M}$ ($0.0903 \to 0.0253 \to 0.0097$).
- **Viva Lesson Learned**:
  When benchmarking quantum measurement variance and shot-noise statistics, always ensure the test state is in a genuine coherent superposition of the observable's eigenbasis. An exact eigenstate will collapse deterministically with zero variance regardless of shot budget.

---

### Failure Record 05: Python GIL and Windows Process Spawn Inversion in Parallel Scaling
- **Track**: Track D (Submodular, Parallel & Distributed Optimization)
- **Date**: 2026-10-21
- **AI Tool & Model**: Antigravity (Gemini 3.8 Flash)
- **Prompt Reference**: [`prompts/track_D/prompt_01_submodular_parallel_distributed.md`](file:///c:/dev/ot-8/prompts/track_D/prompt_01_submodular_parallel_distributed.md)
- **The AI Mistake**:
  1. The initial synthetic compute kernel used a pure Python loop of trigonometric scalars (`math.sin`, `math.cos`) executed via `threading.Thread` and `ProcessPoolExecutor`. Because pure Python bytecode does not release CPython's Global Interpreter Lock (GIL), multi-threaded workers were completely serialized on a single CPU core.
  2. To bypass the GIL, `ProcessPoolExecutor` was initially invoked per benchmark step. However, on Windows, multiprocessing defaults to `spawn` (which starts a brand-new Python interpreter process from scratch, importing packages and runtime). With a 1.0-second workload, the 1.5–2.0s per-worker spawn overhead caused 4 workers to take 5.3s (speedup = $0.19\times$, an 80% slowdown!).
- **Why It Failed / Subtle Bug**:
  The AI failed to account for CPython's execution model and operating system IPC differences:
  - Threads running pure Python bytecode cannot achieve multi-core parallelism due to the GIL.
  - Windows lacks POSIX `fork()`; spawning worker processes incurs significant cold-start process creation and module re-import latency.
- **Verification Method (How It Was Caught)**:
  Empirical benchmark run of `run_track_d_benchmarks.py` output:
  `Sequential T(1) = 1.019 s`, `Static p=2: T=4.841 s (Speedup: 0.21x)`, `Static p=4: T=5.299 s (Speedup: 0.19x)`, and Amdahl's fitted parallel fraction collapsed to `s = 0.000`.
- **The Fix**:
  1. Refactored `synthetic_compute_kernel` to use NumPy dense BLAS matrix operations (`np.dot(np.sin(mat), np.cos(mat))`). Under the hood, NumPy releases the GIL during C/BLAS matrix calculations, enabling true parallel execution across multiple CPU cores within threads.
  2. Updated parallel executors to leverage `ThreadPoolExecutor` with GIL-releasing kernels, eliminating the Windows process spawn latency entirely and achieving $2.17\times$ speedup on 2 workers and $>2.8\times$ on 4 workers ($T(1)=0.44\text{s} \to T(4)=0.15\text{s}$).
- **Viva Lesson Learned**:
  Parallel scaling is fundamentally constrained by runtime overheads: thread parallelism in Python requires explicit GIL release (via BLAS, Cython, or C extensions), while process parallelism on Windows must avoid repeated process instantiation in the critical path due to non-trivial `spawn` latency.

---

### Failure Record 06: Over-Rotation in Grover's Search for N=4 Due to Asymptotic Formula Approximation
- **Track**: Track I (Quantum Algorithms & Quantum Machine Learning)
- **Date**: 2026-10-22
- **AI Tool & Model**: Antigravity (Gemini 3.8 Flash)
- **Prompt Reference**: [`prompts/track_I/prompt_01_quantum_algorithms.md`](file:///c:/dev/ot-8/prompts/track_I/prompt_01_quantum_algorithms.md)
- **The AI Mistake**:
  In `build_grover_circuit`, the optimal iteration count was initially calculated using the asymptotic heuristic formula:
  ```python
  iterations = max(1, int(round((math.pi / 4.0) * math.sqrt(N))))
  ```
  For $N = 4$ ($n = 2$ qubits), $\sqrt{4} = 2$, yielding $\frac{\pi}{4} \times 2 = \frac{\pi}{2} \approx 1.571$. Rounding $1.571$ evaluated to **2 iterations**.
  Because each Grover iteration rotates the statevector in the 2D subspace by $2\theta = 2\arcsin(1/\sqrt{4}) = 2(\pi/6) = \pi/3$, after 1 iteration the state angle is exactly $\pi/6 + \pi/3 = \pi/2$ (100% target state probability). Applying a 2nd iteration rotated the state to $5\pi/6$ ($150^\circ$), causing severe over-rotation where target success probability collapsed to $\sin^2(5\pi/6) = 0.25$ (measured $0.238$ in simulation), no better than random guessing.
- **Why It Failed / Root Cause**:
  The asymptotic formula $R \approx \frac{\pi}{4}\sqrt{N}$ assumes small angles $\arcsin(1/\sqrt{N}) \approx 1/\sqrt{N}$, which holds accurately only as $N \to \infty$. For small search spaces ($N=4$), $\theta = 30^\circ$ is large, and the exact rotation condition $(2R + 1)\theta \approx \pi/2$ requires subtracting $0.5$:
  $R = \text{round}\left( \frac{\pi}{4\arcsin(1/\sqrt{N})} - 0.5 \right) = \text{round}(1.5 - 0.5) = 1$.
- **Verification Method (How It Was Caught)**:
  Executing `evaluate_grover_noise_impact([2, 3])` on the ideal simulator output a success probability of only $0.2387$ ($23.9\%$) on 2 qubits, immediately flagging the over-rotation anomaly.
- **The Fix**:
  Implemented `optimal_grover_iterations(n: int)` using the exact closed-form trigonometric formula:
  ```python
  def optimal_grover_iterations(n: int) -> int:
      N = 2 ** n
      theta = math.asin(1.0 / math.sqrt(N))
      return max(1, int(round((math.pi / (4.0 * theta)) - 0.5)))
  ```
  This immediately yielded $R = 1$ for $n=2$, restoring exact $100.0\%$ target state success on the ideal simulator.
- **Viva Lesson Learned**:
  Asymptotic approximations in quantum algorithm theory ($O(\sqrt{N})$, $O(n^2)$) often hide constant phase offsets and small-angle linearization assumptions. When implementing quantum algorithms on small NISQ or simulator instances ($n \in [2..6]$), always evaluate exact non-linear trigonometric boundaries to avoid destructive over-rotation.

---

### Log Status
- [x] **Failure 1**: Documented above (Track A: Script import resolution & dependency assumption).
- [x] **Failure 2**: Documented above (Track A: Discrepancy between empirical metrics and textual analysis).
- [x] **Failure 3**: Documented above (Track A: Inverted eigenvector indexing in Low-Rank Curvature).
- [x] **Failure 4**: Documented above (Track C: Observable eigenstate zero-variance in finite-shot scaling).
- [x] **Failure 5**: Documented above (Track D: Python GIL and Windows Process Spawn Inversion in Parallel Scaling).
- [x] **Failure 6**: Documented above (Track I: Over-Rotation in Grover's Search for N=4 Due to Asymptotic Formula Approximation).

