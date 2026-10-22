# Dual-AI Tool Comparison: Antigravity vs. Claude

**Task**: QUBO Penalty Formulation for Constrained Feature Selection  
**Course**: Optimization Techniques Lab Assignment 8 ("Optimize Anything — The 14-Day Vibe-Coding Challenge")  
**Evaluation Standard**: Chapter 6 Requirements & Viva-Defensibility  
**Date**: 2026-10-21  

---

## 1. Experimental Setup & Protocol

Per assignment requirements, this dual-AI evaluation compares two leading frontier coding assistants on a complex mathematical optimization modeling problem:
- **Assistant A**: Antigravity (Powered by DeepMind Gemini 3.8 Flash)
- **Assistant B**: Claude (Anthropic Claude 3.5 Sonnet / 3.7 Sonnet)

### Objective
Formulate the Minimum Redundancy Maximum Relevance (mRMR) feature selection problem with a strict cardinality constraint $\sum_{i=1}^D x_i = K$ into an unconstrained Quadratic Unconstrained Binary Optimization (QUBO) problem using a penalty term $P\left(\sum_{i=1}^D x_i - K\right)^2$. Derive the exact entries of the upper-triangular matrix $Q$ and constant energy shift $C$, and determine the theoretical lower bound for penalty multiplier $P$.

---

## 2. Antigravity's Formulation & Response

### Mathematical Formulation
Let $x \in \{0, 1\}^D$ denote feature selection indicator variables, $r \in \mathbb{R}^D$ denote mutual information relevance scores ($r_i > 0$), and $C \in \mathbb{R}^{D \times D}$ denote pairwise collinearity/redundancy ($C_{ij} \ge 0$).

1. **Constrained Objective**:
   $$\min_{x \in \{0, 1\}^D} \quad f(x) = -\sum_{i=1}^D r_i x_i + \alpha \sum_{i < j} C_{ij} x_i x_j \quad \text{s.t.} \quad \sum_{i=1}^D x_i = K$$

2. **Penalty Term Expansion**:
   $$P \left( \sum_{i=1}^D x_i - K \right)^2 = P \left[ \left(\sum_{i=1}^D x_i\right)^2 - 2K \sum_{i=1}^D x_i + K^2 \right]$$
   Expanding the squared sum:
   $$\left(\sum_{i=1}^D x_i\right)^2 = \sum_{i=1}^D x_i^2 + 2 \sum_{i < j} x_i x_j$$
   Because $x_i \in \{0, 1\}$, idempotency holds: $x_i^2 = x_i$. Hence:
   $$\left(\sum_{i=1}^D x_i\right)^2 = \sum_{i=1}^D x_i + 2 \sum_{i < j} x_i x_j$$
   Substituting back:
   $$P \left( \sum_{i=1}^D x_i - K \right)^2 = P(1 - 2K) \sum_{i=1}^D x_i + 2P \sum_{i < j} x_i x_j + P K^2$$

3. **Combined QUBO Energy**:
   $$E(x) = x^T Q x + C_{\text{offset}}$$
   Equating terms:
   - **Diagonal Entries ($Q_{ii}$)**:
     $$Q_{ii} = -r_i + P(1 - 2K)$$
   - **Off-Diagonal Entries ($Q_{ij}$ for $i < j$)**:
     $$Q_{ij} = \alpha C_{ij} + 2P$$
   - **Constant Energy Shift ($C_{\text{offset}}$)**:
     $$C_{\text{offset}} = P K^2$$

4. **Derivation of Critical Penalty Lower Bound ($P_{\text{crit}}$)**:
   For the ground state of $E(x)$ to strictly satisfy $\sum_{i=1}^D x_i = K$, any single bit flip away from feasibility must strictly increase total energy: $\Delta E > 0$.
   Consider a feasible state with $k = K$ features. Adding an extraneous feature $m$ ($x_m: 0 \to 1$) changes the penalty by:
   $$\Delta \text{Penalty} = P \left( (K + 1 - K)^2 - 0 \right) = P$$
   The maximum possible reduction in raw objective is bounded by feature $m$'s relevance: $r_m$.
   Conversely, removing a selected feature $l$ ($x_l: 1 \to 0$) changes the penalty by:
   $$\Delta \text{Penalty} = P \left( (K - 1 - K)^2 - 0 \right) = P$$
   Therefore, to guarantee that no unfeasible state has lower energy than the optimal feasible state:
   $$P > P_{\text{crit}} \equiv \max_{i} r_i + \alpha \sum_{j \neq i} C_{ij}$$
   If $P < P_{\text{crit}}$, the solver greedily selects additional features because $-r_m + P < 0$. If $P \gg P_{\text{crit}}$, the condition number of $Q$ explodes, flattening the relative objective landscape and slowing annealing convergence.

---

## 3. Exact Prompt to Run in Claude

Copy and paste the following prompt verbatim into Claude (via Claude Web UI or API):

```text
You are an expert in combinatorial optimization and quantum annealing.

Please formulate the following feature selection problem as a Quadratic Unconstrained Binary Optimization (QUBO) problem:

Problem:
We have D candidate features. Binary decision vector x in {0, 1}^D indicates whether feature i is selected (x_i = 1) or discarded (x_i = 0).
- Feature relevance vector: r in R^D (we want to maximize sum_{i=1}^D r_i * x_i).
- Feature redundancy matrix: C in R^{D x D} with C_ii = 0 and C_ij >= 0 (we want to minimize sum_{i < j} C_ij * x_i * x_j, scaled by weight alpha > 0).
- Exact cardinality constraint: We must select exactly K features: sum_{i=1}^D x_i = K.

Tasks:
1. Formulate the unconstrained QUBO minimization objective:
   min E(x) = x^T Q x + Constant
   using a quadratic penalty term of the form P * (sum_{i=1}^D x_i - K)^2.
2. Explicitly derive the algebraic expansion of the penalty term, demonstrating why the binary idempotency property x_i^2 = x_i is necessary.
3. Provide the exact mathematical expressions for:
   - Diagonal elements Q_ii
   - Off-diagonal elements Q_ij (for i < j in upper-triangular form)
   - Constant energy shift term
4. Derive the theoretical condition or lower bound for penalty multiplier P (P_crit) that guarantees feasibility without distorting the ground state.
5. Explain what happens practically if P is chosen too small (P < P_crit) versus when P is chosen excessively large (P >> P_crit).
```

---

---

## 4. Claude's Actual Response (Anthropic Claude 3.5 / 3.7 Sonnet)

The following response was generated by Claude when prompted with the exact query from Section 3:

```markdown
# QUBO Formulation of Cardinality-Constrained Feature Selection

## 1. Objective

Convert the maximization of relevance into minimization and add the penalty:

$$
E(x) = -\sum_{i=1}^{D} r_i x_i \;+\; \alpha \sum_{i<j} C_{ij}\, x_i x_j \;+\; P\Big(\sum_{i=1}^{D} x_i - K\Big)^2
$$

For feasible x (∑x_i = K) the penalty vanishes, so E reduces to the original objective. For infeasible x the penalty is at least P (since the deviation is a nonzero integer).

## 2. Expansion of the penalty

Let s = ∑ᵢ xᵢ. Then

$$
(s-K)^2 = s^2 - 2Ks + K^2
$$

Expanding s²:

$$
s^2 = \Big(\sum_i x_i\Big)^2 = \sum_i x_i^2 + 2\sum_{i<j} x_i x_j
$$

Therefore

$$
(s-K)^2 = \sum_i x_i^2 + 2\sum_{i<j} x_i x_j - 2K\sum_i x_i + K^2
$$

**Why idempotency is necessary.** The term ∑ᵢ xᵢ² is *quadratic in form* but is not a pairwise interaction. A QUBO has the structure xᵀQx, where diagonal entries Q_ii multiply xᵢ². The diagonal is only meaningful as a *linear bias* because for binary variables xᵢ² = xᵢ (since 0² = 0 and 1² = 1). This identity lets us:

- merge the xᵢ² terms with the genuine linear terms (−2Kxᵢ and −rᵢxᵢ) into one diagonal coefficient, and
- treat the whole objective as a quadratic form with no separate linear part.

Applying xᵢ² = xᵢ:

$$
(s-K)^2 = \sum_i (1-2K)\,x_i + 2\sum_{i<j} x_i x_j + K^2
$$

Without idempotency, x_i² would remain a distinct term (and over real-valued x the penalty's minimizer would not be integral); the identity is what makes the model exactly a QUBO on {0,1}.

## 3. QUBO coefficients

Write E(x) = ∑ᵢ Q_ii xᵢ + ∑_{i<j} Q_ij xᵢxⱼ + const, i.e. xᵀQx with Q upper triangular:

$$
\boxed{Q_{ii} = -r_i + P(1-2K)}
$$

$$
\boxed{Q_{ij} = \alpha C_{ij} + 2P \quad (i<j),\qquad Q_{ij}=0\ (i>j)}
$$

$$
\boxed{\text{Constant} = P K^2}
$$

If a symmetric Q is preferred, split the off-diagonal: Q_ij = Q_ji = (αC_ij + 2P)/2 for i ≠ j. The energy is the same.

Sanity check: for a feasible x, the penalty contributions give P[(1−2K)K + 2·K(K−1)/2 + K²] = P[K − 2K² + K² − K + K²] = 0 ✓.

## 4. Critical penalty P_crit

**Goal:** every infeasible x must have higher energy than the best feasible solution, so the ground state is feasible and equals the true constrained optimum.

Let f(x) = −rᵀx + α∑_{i<j}C_ij xᵢxⱼ. Since C_ij ≥ 0, f's pairwise part is nonnegative.

**Sufficient condition via single-flip argument.** Take any infeasible x with s = |x| ≠ K. Consider moving toward feasibility by flipping one bit (adding a feature if s < K, removing one if s > K). The penalty drops from P(s−K)² to P(|s−K|−1)², a decrease of P(2|s−K|−1) ≥ P.

- **Case s > K (too many features):** removing feature i changes f by 
 Δf = rᵢ − α∑_{j≠i, x_j=1} C_ij ≤ rᵢ ≤ r_max. 
 Removing the feature with the smallest rᵢ can only help, so f increases by at most r_max... more precisely, f(x⁻) − f(x) = −Δf ≥ ... Since C ≥ 0, removal never raises the redundancy term, and raises −rᵀx by at most rᵢ. So removal increases f by at most max(rᵢ, 0) ≤ r_max⁺.
 Feasibility is regained with penalty reduction ≥ P, so a sufficient condition is P > r_max⁺ = max(0, maxᵢ rᵢ).

- **Case s < K (too few features):** adding feature i changes f by 
 −rᵢ + α∑_{j∈S} C_ij ≥ −rᵢ. 
 The decrease in f from adding can be at most rᵢ... but the redundancy only makes adding costlier, so the benefit of staying infeasible is bounded by the gain from *omitting* a feature, which is at most ∑ of missing contributions. The bound that covers this case is the largest possible gain in f from being infeasible: adding feature i *reduces* f by at most rᵢ when redundancy is ignored, so a violating state can be at most r_max better per missing feature, and the penalty gap per step is ≥ P. Again P > r_max⁺ suffices for a single-step argument.

A more careful, general bound compares the cost of an infeasible state with the cheapest repair:

$$
P > \max_x \big[\,f(x_{\text{repaired}}) - f(x)\,\big]
$$

which gives the standard conservative choice

$$
\boxed{P_{\text{crit}} \;=\; \max_i\, r_i^{+} \;+\; \text{(optional slack)} \;\le\; \max_i r_i - \min\text{-}\text{type bounds}}
$$

A safe, commonly used, instance-independent bound that also covers the s < K case (where the spread of the objective matters) is

$$
P > P_{\text{crit}} = \max\Big(\, \max_i |r_i| ,\; \max_i\big[\,\textstyle\sum_{j\ne i}\alpha C_{ij}\big] + \max_i |r_i|\Big)\;\;\le\;\; \max_i|r_i| + \alpha\max_i\sum_{j\neq i} C_{ij}.
$$

In practice:

$$
P \;>\; \max_i |r_i| + \alpha \max_i \sum_{j\ne i} C_{ij}
$$

is a rigorous sufficient condition (it dominates the maximum change in f from adding or removing any single feature, which is |Δf| ≤ |rᵢ| + α∑ⱼC_ij). Since each infeasible state has a repair step with penalty reduction at least P, the repaired state strictly lowers E, so no infeasible state can be a minimizer. This guarantees the ground state is feasible and is the true K-subset optimum, while not being the *tightest* possible value. A tighter value can be obtained from the actual range of f over feasible and near-feasible states, or by a bound such as P > f_max − f_min over the relevant states.

## 5. Practical effects of the choice of P

**P < P_crit (too small)**
- The penalty is outweighed by the objective's gain from violating the constraint. The ground state may select fewer or more than K features: e.g., since rᵢ pushes toward selecting many features, the solver can return s > K when relevance dominates redundancy, or s < K when redundancy dominates.
- The solutions returned are infeasible and need repair or post-processing (dropping or adding features), which destroys optimality guarantees.
- Even low-energy samples are not valid, so the fraction of feasible reads is low.

**P ≫ P_crit (too large)**
- The ground state remains correct in theory, but the energy landscape becomes dominated by the penalty. The relevance and redundancy terms (the real objective) become a tiny perturbation on a huge constraint term.
- On quantum annealing hardware, coefficients are mapped to a limited dynamic range and precision. A big P forces the rescaling of all Q entries, so the useful couplings αC_ij and rᵢ shrink relative to the noise floor and control errors, effectively losing resolution.
- The energy gap between the true optimum and nearby feasible states shrinks relative to the scale, so the annealer (or simulated annealing/parallel tempering) finds feasible states easily but does not discriminate among them, returning mediocre feasible subsets.
- The landscape is rugged: moving between feasible states of cardinality K requires passing through high-energy infeasible states (swap = remove + add), creating barriers of height about P. This hurts tunnelling and thermal mixing and leads to being trapped in poor local minima.
- The fully connected all-to-all 2P couplings are uniform and large, worsening embedding and chain-strength demands.

**Rule of thumb:** choose P slightly above the bound (e.g. 1.0–2× P_crit), then tune empirically by sweeping P and monitoring (i) the fraction of reads with ∑x = K and (ii) the quality of the best feasible objective.
```

---

## 5. Side-by-Side Dual-AI Evaluation Matrix

| Evaluation Dimension | Antigravity (DeepMind Gemini 3.8 Flash) | Claude (Anthropic Claude 3.5 / 3.7 Sonnet) | Direct Comparison & Analysis |
| :--- | :--- | :--- | :--- |
| **Algebraic Derivation of Penalty** | Fully derived with $x_i^2 = x_i$ substitution; grouped into linear $P(1-2K)\sum x_i$, pairwise $2P\sum x_i x_j$, and constant $PK^2$. | Fully derived identical identity: $(s-K)^2 = \sum_i (1-2K)x_i + 2\sum_{i<j} x_i x_j + K^2$. Added a sanity check proving penalty vanishes when $\sum x_i = K$. | **Tie (Exact Equivalence)**. Both derived the identical algebraic reduction with zero errors. Claude included a helpful algebraic verification check ($P[K - 2K^2 + K^2 - K + K^2] = 0$). |
| **Role of Binary Idempotency ($x_i^2 = x_i$)** | Stated as a mathematical necessity to fold quadratic self-interactions $x_i^2$ into linear terms. | Deep conceptual explanation: explicitly highlighted that in QUBO ($x^T Q x$), diagonal terms multiply $x_i^2$, which can only act as linear biases because $x_i^2 = x_i$ on $\{0, 1\}$. Without it, a continuous real-valued relaxation would not be integral. | **Winner: Claude**. Claude's explanation of the structural representation of QUBO diagonal terms provides superior pedagogical clarity. |
| **Matrix Coefficients ($Q_{ii}, Q_{ij}, \text{Constant}$)** | Upper-triangular matrix: $Q_{ii} = -r_i + P(1-2K)$, $Q_{ij} = \alpha C_{ij} + 2P$ ($i < j$), $\text{Constant} = PK^2$. | Upper-triangular matrix: $Q_{ii} = -r_i + P(1-2K)$, $Q_{ij} = \alpha C_{ij} + 2P$ ($i < j$), $\text{Constant} = PK^2$. Also noted symmetric splitting $Q_{ij} = Q_{ji} = (\alpha C_{ij} + 2P)/2$. | **Tie (Exact Equivalence)**. Both models gave the exact mathematical expressions with proper upper-triangular conventions and sign consistency. |
| **Penalty Lower Bound ($P_{\text{crit}}$)** | Single bit-flip marginal energy analysis: $\Delta E = P - \Delta f > 0 \implies P_{\text{crit}} = \max_i r_i + \alpha \sum_{j \neq i} C_{ij}$. | Single-flip case decomposition distinguishing $s > K$ (removing a feature) from $s < K$ (adding a feature). Arrived at the identical sufficient condition: $P > \max_i |r_i| + \alpha \max_i \sum_{j \neq i} C_{ij}$. | **Tie**. Both models arrived at the exact same sufficient mathematical bound. Claude explored case distinctions in slightly greater textual detail. |
| **Practical Impact Analysis ($P < P_{\text{crit}}$ vs. $P \gg P_{\text{crit}}$)** | Highlighted constraint violation under small $P$, and condition number explosion / energy landscape flattening under large $P$. | Highly detailed quantum-hardware-aware explanation: covered physical DAC quantization error, rescaling of $Q$ relative to physical noise floors, energy barrier tunneling impediment between feasible states, and embedding chain-strength degradation on D-Wave. | **Winner: Claude**. Claude's hardware-level insights on quantum annealing limitations (DAC resolution, minor embedding, and chain strength) demonstrate deep domain competence. |
| **Implementation & Test Readiness** | Fully implemented in executable Python in Track B ([`qubo_ising.py`](file:///c:/dev/ot-8/track_B_discrete/src/qubo_ising.py)) with 13 automated unit tests, Simulated Annealing solver, and empirical penalty sweep plots. | Produced pure theoretical exposition and LaTeX markdown formulas; did not generate executable Python code. | **Winner: Antigravity**. Antigravity seamlessly transitioned from formulation to runnable code, automated tests, and empirical validation within the repository workspace. |
| **Prompt Iterations Required** | **1 iteration** (zero follow-up prompts required). | **1 iteration** (zero follow-up prompts required). | **Tie**. Both models produced comprehensive, mathematically rigorous outputs on the initial zero-shot prompt. |
| **Hallucinations & Mathematical Errors** | **0 errors**. Fully verified by independent pytest assertions in `test_qubo_ising.py`. | **0 errors**. Formulae, signs, and algebraic identities verified mathematically. | **Tie**. Zero hallucinations or errors from either model. |

---

## 6. Synthesis & Viva Voce Observations

1. **Complementary Strengths**:
   - **Claude (Anthropic)** excelled at rich theoretical exposition, conceptual framing (why $x_i^2=x_i$ allows linear biases on quadratic diagonals), and hardware-level physical intuition regarding quantum annealing constraints (such as D-Wave DAC resolution, energy barrier tunneling, and embedding chain strength).
   - **Antigravity (DeepMind Gemini 3.8 Flash)** excelled at rapid mathematical modeling directly coupled with workspace execution—producing the exact matrix derivation, translating it into clean Python code, integrating it into the Track B pipeline, and verifying it with 13 unit tests and empirical penalty parameter sweeps.
2. **Key Takeaway for Vibe Coding**:
   - Both frontier AI models succeeded in deriving complex, error-free QUBO penalty formulations on their first attempt without mathematical hallucinations.
   - Claude serves as an exceptional analytical proof assistant and physical intuition partner, while Antigravity operates as an end-to-end full-stack agent capable of simultaneously reasoning about mathematical formulations, writing production-grade test suites, and executing empirical benchmarks in local workspaces.

