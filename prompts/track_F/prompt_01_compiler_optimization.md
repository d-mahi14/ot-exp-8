# Prompt Record 14: Track F — Mini-Compiler Optimization (TAC IR Passes & ILP Register Allocation)

- **Prompt ID**: `P-TRF-01`
- **Date & Time**: 2026-10-21 23:50:24 IST
- **Track**: Track F (Compiler Optimization)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: Pending commit

---

### Purpose
Complete Track F: Mini-Compiler Optimization:
1. **Three-Address Code (TAC) IR Architecture**:
   - Construct instruction representations (`Instruction`, `OpCode`), `BasicBlock`, and `ControlFlowGraph` (CFG).
   - Implement backward iterative liveness analysis ($In[B] = Use[B] \cup (Out[B] \setminus Def[B])$).
   - Implement dominator tree computation and natural loop detection.
2. **Five Optimization Passes**:
   - **Constant Folding & Constant Propagation**: Propagates constant scalars and evaluates compile-time expressions while strictly preserving runtime exception semantics (division by zero is never folded).
   - **Common-Subexpression Elimination (CSE)**: Detects duplicate commutative expressions and reuses precomputed values, invalidating entries upon variable mutation.
   - **Loop-Invariant Code Motion (LICM)**: Identifies loop-invariant computations inside natural loops and hoists them to pre-headers.
   - **Dead-Code Elimination (DCE)**: Removes dead variable assignments and prunes unreachable basic blocks.
   - **ILP-Based Register Allocation**: Formulates interference graph coloring as a 0-1 Integer Linear Program (ILP) with minimum spill cost, solved with HiGHS branch-and-cut.
3. **Deterministic Virtual Machine / Interpreter**:
   - Implements `IRInterpreter` measuring dynamic instruction path length, simulated microarchitectural cycle costs, and variable state.
   - Guarantees and asserts mathematical output equivalence between unoptimized and optimized code streams.
4. **Verification & Benchmarks**:
   - 8 unit tests in `track_F_compiler/tests/test_compiler.py`.
   - 5 benchmark programs in `run_compiler_benchmarks.py`.
   - CSV, native Markdown tables, and 4-panel visual plot `compiler_benchmarks.png`.

---

### Original Prompt
```text
Continue Assignment 8 in `C:\dev\ot-8`. Complete Track F: Compiler Optimization.

Inspect the assignment, repository, checklist, prompt logs, failure log, and Git status first. Build a small compiler or optimization pass for a toy language, or extend a suitable existing educational compiler. Keep the implementation understandable and demonstrable.

### Required work

Implement at least four of the following optimizations:
- Constant folding.
- Constant propagation.
- Dead-code elimination.
- Common-subexpression elimination.
- Loop-invariant code motion.
- Loop unrolling.
- Instruction/register allocation optimization.

Implement at least two optimizations in depth and explain their algorithms, correctness conditions, and limitations.

### Verification
- Create input programs that exercise each optimization.
- Verify that optimized and unoptimized programs produce identical outputs for supported inputs.
- Add tests for edge cases and invalid inputs.
- Measure instruction count, runtime, code size, or compile time before and after optimization.
- Include cases where an optimization does not help or must not be applied.
- Document the compiler pipeline and demonstrate the optimizations on sample programs.

### Deliverables
Save source code, tests, sample programs, benchmark scripts, CSV results, Markdown summaries, and useful plots where appropriate. Save the exact prompt under `prompts/track_F/`. Update `PROMPT_LOG.md`, `CHAPTER_6_CHECKLIST.md`, and `AI_Prompt_Log.pdf` according to project conventions. Record only genuine failures in `AI_FAILURE_LOG.md`.

Run `py -3.12 -m pytest -v` if the project uses Python tests; also run the compiler's own build and execution tests. Report actual results.

Commit local changes with `[AI:Antigravity]` at the start of the message. Do not create a GitHub repository, configure a remote, push, or publish anything. Do not change branches or rewrite history without asking. Preserve unrelated work.

Stop after Track F and summarize implemented optimizations, correctness evidence, measured improvements, tests, limitations, and commit hash.
```

---

### Execution & Verification Summary
1. **Source Code**:
   - `track_F_compiler/src/ir.py`: TAC Instructions, BasicBlock, ControlFlowGraph, Liveness Analysis, Dominator Tree, Natural Loops.
   - `track_F_compiler/src/interpreter.py`: Virtual Machine interpreter with microarchitectural cycle counting.
   - `track_F_compiler/src/optimizations.py`: ConstFold/Prop, CSE, LICM, DCE, and ILP Register Allocation.
   - `track_F_compiler/src/compiler.py`: MiniCompiler pipeline and IRBuilder.
   - `track_F_compiler/src/__init__.py`: Package exports.
2. **Unit Tests**:
   - `track_F_compiler/tests/test_compiler.py`: 8 tests passing.
3. **Benchmarks & Artifacts**:
   - `track_F_compiler/benchmarks/run_compiler_benchmarks.py`: 5 programs evaluated.
   - `track_F_compiler/results/compiler_metrics.csv`
   - `track_F_compiler/results/compiler_metrics.md`
   - `track_F_compiler/results/compiler_benchmarks.png`.
