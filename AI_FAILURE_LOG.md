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

### Log Status
- [x] **Failure 1**: Documented above (Track A: Script import resolution & dependency assumption).
- [ ] **Failure 2**: Pending implementation of Tracks F–H.
- [ ] **Failure 3**: Pending implementation of Tracks C, I, or J.
