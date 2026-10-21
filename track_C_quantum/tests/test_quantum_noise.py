"""
test_quantum_noise.py
Independent unit tests for finite-shot variance, quantum hardware noise models,
and barren-plateau gradient variance scaling.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from track_C_quantum.src.quantum_noise_experiments import (
    build_depolarizing_noise_model,
    build_readout_noise_model,
    build_combined_nisq_noise_model,
    evaluate_finite_shot_variance,
    evaluate_noise_impact,
    run_barren_plateaus_study,
    explain_noise_and_barren_plateaus,
)


class TestNoiseModels:
    """Tests for Qiskit Aer noise model construction."""

    def test_depolarizing_noise_model(self):
        nm = build_depolarizing_noise_model(p1=0.01, p2=0.05)
        instructions = nm.noise_instructions
        assert "rx" in instructions
        assert "cx" in instructions

    def test_readout_noise_model(self):
        nm = build_readout_noise_model(p_meas=0.04)
        assert "measure" in nm.noise_instructions

    def test_combined_nisq_model(self):
        nm = build_combined_nisq_noise_model(p1=0.005, p2=0.02, p_meas=0.03)
        instructions = nm.noise_instructions
        assert "measure" in instructions
        assert "cx" in instructions
        assert "ry" in instructions


class TestShotVarianceAndNoise:
    """Tests for finite-shot statistics and noise degradation."""

    def test_finite_shot_variance_scaling(self):
        qc = QuantumCircuit(2)
        qc.h([0, 1])
        qc.cx(0, 1)
        H = SparsePauliOp.from_list([("ZZ", 1.0)])

        res = evaluate_finite_shot_variance(qc, H, shots_list=[100, 1000], n_trials=20, seed=42)
        stats = res["results"]
        assert len(stats) == 2

        var_100 = stats[0]["variance"]
        var_1000 = stats[1]["variance"]

        # 100 shots should have strictly larger variance than 1000 shots
        assert var_100 > var_1000

    def test_noise_impact_shifts_energy(self):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        H = SparsePauliOp.from_list([("ZZ", 1.0)])

        res = evaluate_noise_impact(qc, H, shots=1000, p1=0.02, p2=0.08, p_meas=0.05, seed=42)
        envs = {e["environment"]: e for e in res["environments"]}

        assert "Ideal (Noiseless)" in envs
        assert "Combined NISQ Model" in envs
        # Combined noise should exhibit non-zero energy shift
        assert envs["Combined NISQ Model"]["energy_shift"] > 0.0


class TestBarrenPlateaus:
    """Tests for barren-plateau gradient variance exponential decay."""

    def test_barren_plateaus_gradient_variance_decay(self):
        # Quick study on N in [2, 3, 4] with 15 samples
        study = run_barren_plateaus_study(qubit_range=[2, 3, 4], n_samples=15, seed=42)
        assert len(study["results"]) == 3
        # Negative slope verifies exponential decay of gradient variance with qubit count
        assert study["slope"] < 0.0

        var_2 = study["results"][0]["gradient_variance"]
        var_4 = study["results"][2]["gradient_variance"]
        assert var_2 > var_4

    def test_explanation_text(self):
        text = explain_noise_and_barren_plateaus()
        assert "Finite-Shot Measurement Scaling" in text
        assert "Hardware Noise" in text
        assert "Barren Plateau Phenomenon" in text
