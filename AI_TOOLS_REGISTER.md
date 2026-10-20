# AI Tools & Models Register
**Assignment 8**: Optimize Anything — The 14-Day Vibe-Coding Challenge  
**Section 8 Strict Disclosure**: All tools and models used in planning, code generation, debugging, testing, and documentation are registered below.

---

## 1. Registered Tools and Models

| Tool Name | Developer / Provider | Model Identifier | Active Tracks | Dates Used | Scope of Work |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Antigravity** | Google DeepMind | Gemini 3.8 Flash | General, Tracks A–J | 2026-10-20 to Present | System planning, architecture design, test harness generation, algorithm implementation, failure diagnosis, documentation. |
| **Claude** | Anthropic | Claude 3.5 Sonnet / Claude 3.7 Sonnet | General / Track B Comparison | 2026-10-20 to Present | Dual-AI evaluation comparison on identical combinatorial formulation task (Track B QUBO formulation & reduction). |

---

## 2. Commit Naming & Traceability Protocol
To ensure evaluators can immediately distinguish AI-generated contributions from student corrections:
- **`[AI:Antigravity]`**: Commits containing substantial code, tests, or docs generated via Antigravity prompts.
- **`[AI:Claude]`**: Commits containing components or comparative benchmarks generated via Claude.
- **`[Manual:Fix]`**: Commits containing student-authored manual corrections, debugging patches, and verification edits.
- **`[Manual:Doc]`**: Commits containing human explanations, viva answers, and self-authored analysis.

---

## 3. Tool Comparison Plan (Section 8 Mandate)
- **Target Task**: Formulating the Graph Maximum Cut (Max-Cut) Problem into an unconstrained QUBO / Ising Hamiltonian with constraint penalty terms, and implementing a verification baseline.
- **Comparison Dimensions**:
  1. Mathematical correctness of binary-to-spin variable substitution ($s_i = 1 - 2x_i$).
  2. Hallucinations or missing penalty bounds ($P > \sum |w_{ij}|$).
  3. Execution speed and solution optimality on a 6-node weighted graph.
  4. Number of prompt iterations required to produce passing unit tests.
- **Documentation Location**: `prompts/general/tool_comparison_antigravity_vs_claude.md` and technical report.
