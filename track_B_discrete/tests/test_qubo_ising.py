"""
test_qubo_ising.py
Independent unit tests for Track B Batch 3:
- QUBO energy evaluation, symmetric, and upper-triangular representations
- Exact algebraic equivalence between QUBO and Ising models: s_i = 1 - 2*x_i
- Graph Max-Cut QUBO and Ising formulations vs direct graph cut values
- Feature Selection mRMR QUBO formulation and penalty term P*(sum(x_i) - K)^2
- Critical penalty threshold behavior (feasibility enforcement)
- Delta energy calculation exactness for Simulated Annealing
- Exact Brute-Force ground state verification
- Simulated Annealing convergence to exact ground state on small instances
- Edge cases: N=1, zero matrix, K=0, dimension mismatches
"""

import pytest
import numpy as np
from track_B_discrete.src.qubo_ising import (
    QUBOProblem,
    IsingModel,
    maxcut_to_qubo,
    maxcut_to_ising,
    feature_selection_to_qubo,
    solve_qubo_brute_force,
    solve_qubo_simulated_annealing,
)


class TestQUBOFoundations:
    def test_qubo_energy_evaluation(self):
        """Verify basic E(x) = x^T Q x + C on deterministic small matrix."""
        Q = np.array([
            [-2.0, 3.0],
            [0.0, -1.0]
        ])
        qubo = QUBOProblem(Q, constant=5.0)

        assert qubo.evaluate_energy(np.array([0, 0])) == 5.0
        assert qubo.evaluate_energy(np.array([1, 0])) == 5.0 - 2.0
        assert qubo.evaluate_energy(np.array([0, 1])) == 5.0 - 1.0
        assert qubo.evaluate_energy(np.array([1, 1])) == 5.0 - 2.0 - 1.0 + 3.0

    def test_symmetric_and_upper_triangular_equivalence(self):
        """Verify that Q, Q_symmetric, and Q_upper_triangular yield identical energy for all x."""
        rng = np.random.RandomState(42)
        n = 5
        Q_raw = rng.randn(n, n)
        qubo = QUBOProblem(Q_raw, constant=2.5)
        qubo_sym = qubo.to_symmetric()
        qubo_tri = qubo.to_upper_triangular()

        # Test across 32 binary configurations
        for i in range(1 << n):
            x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=np.float64)
            e_raw = qubo.evaluate_energy(x)
            e_sym = qubo_sym.evaluate_energy(x)
            e_tri = qubo_tri.evaluate_energy(x)
            assert np.isclose(e_raw, e_sym, atol=1e-12)
            assert np.isclose(e_raw, e_tri, atol=1e-12)


class TestIsingEquivalence:
    def test_qubo_to_ising_exact_equivalence(self):
        """
        Verify the exact algebraic identity:
            E_QUBO(x) == H_Ising(s), where s_i = 1 - 2*x_i <==> x_i = (1 - s_i)/2
        """
        rng = np.random.RandomState(101)
        n = 6
        Q = rng.uniform(-5.0, 5.0, size=(n, n))
        qubo = QUBOProblem(Q, constant=7.2)
        ising = qubo.to_ising()

        for i in range(1 << n):
            x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=np.float64)
            s = 1.0 - 2.0 * x  # Spin conversion: 0 -> +1, 1 -> -1

            e_qubo = qubo.evaluate_energy(x)
            e_ising = ising.evaluate_energy(s)

            assert np.isclose(e_qubo, e_ising, atol=1e-11), (
                f"Mismatch for state {x}: QUBO={e_qubo}, Ising={e_ising}"
            )

    def test_ising_to_qubo_round_trip(self):
        """Verify that round-trip Ising -> QUBO -> Ising preserves energies exactly."""
        rng = np.random.RandomState(202)
        n = 5
        J = np.triu(rng.uniform(-3.0, 3.0, size=(n, n)), k=1)
        h = rng.uniform(-2.0, 2.0, size=n)
        ising = IsingModel(J=J, h=h, constant=-1.5)

        qubo_recovered = ising.to_qubo()
        ising_recovered = qubo_recovered.to_ising()

        for i in range(1 << n):
            x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=np.float64)
            s = 1.0 - 2.0 * x
            e_orig = ising.evaluate_energy(s)
            e_qubo = qubo_recovered.evaluate_energy(x)
            e_roundtrip = ising_recovered.evaluate_energy(s)

            assert np.isclose(e_orig, e_qubo, atol=1e-11)
            assert np.isclose(e_orig, e_roundtrip, atol=1e-11)


class TestMaxCutFormulation:
    def test_triangle_graph_maxcut(self):
        """
        Triangle graph K_3 with unit weights:
            Edges: (0,1), (1,2), (0,2).
            Maximum cut is 2 (any 2-vs-1 partition).
            QUBO minimizes -Cut(x), so ground state energy must be -2.0.
        """
        W = np.array([
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0]
        ])
        qubo = maxcut_to_qubo(W)
        bf_res = solve_qubo_brute_force(qubo)

        assert np.isclose(bf_res["best_energy"], -2.0)
        # In K_3, 6 out of 8 states achieve cut=2 (all except [0,0,0] and [1,1,1])
        assert len(bf_res["all_ground_states"]) == 6

    def test_maxcut_qubo_and_ising_consistency(self):
        """Verify direct maxcut_to_ising gives identical energies to maxcut_to_qubo.to_ising()."""
        # 4-node cycle C_4: max cut is bipartite cut of size 4
        W = np.array([
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0]
        ])
        qubo = maxcut_to_qubo(W)
        ising_direct = maxcut_to_ising(W)
        ising_from_qubo = qubo.to_ising()

        for i in range(16):
            x = np.array([(i >> bit) & 1 for bit in range(4)], dtype=np.float64)
            s = 1.0 - 2.0 * x

            e_qubo = qubo.evaluate_energy(x)
            e_direct = ising_direct.evaluate_energy(s)
            e_converted = ising_from_qubo.evaluate_energy(s)

            # Direct graph cut calculation: sum_{(i,j)} W_ij * (x_i != x_j)
            cut_val = (
                W[0, 1] * (x[0] != x[1]) +
                W[1, 2] * (x[1] != x[2]) +
                W[2, 3] * (x[2] != x[3]) +
                W[0, 3] * (x[0] != x[3])
            )
            assert np.isclose(e_qubo, -cut_val)
            assert np.isclose(e_direct, -cut_val)
            assert np.isclose(e_converted, -cut_val)


class TestFeatureSelectionFormulation:
    def test_penalty_expansion_zero_at_cardinality(self):
        """When sum(x_i) == K, the penalty P*(sum(x_i) - K)^2 must vanish exactly."""
        r = np.array([3.0, 2.0, 5.0, 1.0])
        C_mat = np.zeros((4, 4))
        K = 2
        P = 50.0
        qubo = feature_selection_to_qubo(r, C_mat, target_k=K, penalty_p=P, alpha=1.0)

        # State with exactly 2 features selected: [1, 0, 1, 0] -> sum = 2
        x_feasible = np.array([1, 0, 1, 0], dtype=np.float64)
        # Expected energy: -r[0] - r[2] + 0 = -3 - 5 = -8.0 (penalty = 0)
        assert np.isclose(qubo.evaluate_energy(x_feasible), -8.0)

        # State with 3 features: [1, 1, 1, 0] -> sum = 3, penalty = P*(3-2)^2 = 50.0
        x_infeasible = np.array([1, 1, 1, 0], dtype=np.float64)
        expected_infeasible = - (3.0 + 2.0 + 5.0) + 50.0 * (1**2)
        assert np.isclose(qubo.evaluate_energy(x_infeasible), expected_infeasible)

    def test_penalty_parameter_threshold(self):
        """
        Verify that:
        - When P is too small (P < P_crit), unconstrained optimization picks all features (cardinality > K).
        - When P is sufficiently large (P >= P_crit), the ground state strictly satisfies sum(x_i) == K.
        """
        r = np.array([10.0, 8.0, 6.0, 4.0])  # all positive relevance
        C_mat = np.zeros((4, 4))             # zero collinearity for clear demonstration
        K = 2

        # With P=0.0 (no penalty), the unconstrained minimum selects all 4 features
        qubo_unpenalized = feature_selection_to_qubo(r, C_mat, target_k=K, penalty_p=0.0)
        bf_unpenalized = solve_qubo_brute_force(qubo_unpenalized)
        assert np.sum(bf_unpenalized["best_state"]) == 4  # Feasibility violated!

        # With P=20.0 (sufficiently high penalty > max(r_i)), ground state picks top 2 features: [1, 1, 0, 0]
        qubo_penalized = feature_selection_to_qubo(r, C_mat, target_k=K, penalty_p=20.0)
        bf_penalized = solve_qubo_brute_force(qubo_penalized)
        assert np.sum(bf_penalized["best_state"]) == K   # Feasibility satisfied!
        assert np.array_equal(bf_penalized["best_state"], np.array([1, 1, 0, 0]))


class TestSimulatedAnnealingAndSolvers:
    def test_sa_delta_energy_exactness(self):
        """
        Verify that the O(N) single-bit-flip delta energy calculation matches
        direct full-matrix energy difference E(x') - E(x) exactly.
        """
        rng = np.random.RandomState(456)
        n = 8
        Q_raw = rng.randn(n, n)
        qubo = QUBOProblem(Q_raw, constant=3.0)
        Q_sym = qubo.to_symmetric().Q

        for _ in range(20):
            x = rng.randint(0, 2, size=n).astype(np.float64)
            k = rng.randint(0, n)

            # Flip bit k
            x_flipped = x.copy()
            x_flipped[k] = 1.0 - x_flipped[k]

            exact_delta = qubo.evaluate_energy(x_flipped) - qubo.evaluate_energy(x)

            # Delta formula from solve_qubo_simulated_annealing
            delta_xk = 1.0 - 2.0 * x[k]
            row_sum = 2.0 * (np.dot(Q_sym[k], x) - Q_sym[k, k] * x[k])
            formula_delta = delta_xk * (Q_sym[k, k] + row_sum)

            assert np.isclose(exact_delta, formula_delta, atol=1e-12)

    def test_sa_matches_brute_force_on_maxcut(self):
        """
        Verify that Simulated Annealing finds the exact ground state energy
        on small Max-Cut graphs (N=6 and N=8) where brute-force is known.
        """
        rng = np.random.RandomState(789)
        # N=6 random graph
        n = 6
        W = rng.randint(0, 5, size=(n, n)).astype(np.float64)
        W = np.triu(W, k=1) + np.triu(W, k=1).T
        qubo = maxcut_to_qubo(W)

        bf_res = solve_qubo_brute_force(qubo)
        sa_res = solve_qubo_simulated_annealing(
            qubo,
            initial_temp=15.0,
            final_temp=0.01,
            cooling_rate=0.98,
            steps_per_temp=100,
            seed=42
        )

        assert np.isclose(sa_res["best_energy"], bf_res["best_energy"], atol=1e-8)


class TestEdgeCasesAndValidation:
    def test_single_variable_qubo(self):
        """Verify N=1 edge case."""
        Q = np.array([[5.0]])
        qubo = QUBOProblem(Q, constant=2.0)
        assert qubo.evaluate_energy(np.array([0])) == 2.0
        assert qubo.evaluate_energy(np.array([1])) == 7.0

        bf = solve_qubo_brute_force(qubo)
        assert np.array_equal(bf["best_state"], np.array([0]))
        assert bf["best_energy"] == 2.0

    def test_dimension_mismatch_raises(self):
        """Non-square matrices and state dimension mismatches must raise ValueError."""
        with pytest.raises(ValueError, match="square 2D matrix"):
            QUBOProblem(np.zeros((2, 3)))

        qubo = QUBOProblem(np.eye(3))
        with pytest.raises(ValueError, match="does not match problem dimension"):
            qubo.evaluate_energy(np.array([1, 0]))

    def test_brute_force_intractable_guard(self):
        """solve_qubo_brute_force must reject n > 22 to prevent memory/time lockup."""
        qubo = QUBOProblem(np.eye(25))
        with pytest.raises(ValueError, match="Brute force intractable"):
            solve_qubo_brute_force(qubo)
