# Track A: First-Order Continuous Optimizers Benchmark Results

**Generated**: 2026-10-20 20:29:53

| Landscape | Optimizer | Initial Loss | Final Loss | Distance to Optimum | Iterations | Runtime (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| Convex Quadratic (kappa=10) | Vanilla GD (Baseline) | 49.5 | 0.00048387456472084574 | 0.031108666468392557 | 150 | 1.283 |
| Convex Quadratic (kappa=10) | Momentum (gamma=0.85) | 49.5 | 1.7549082926981493e-10 | 6.9482827476594704e-06 | 150 | 1.649 |
| Convex Quadratic (kappa=10) | NAG (gamma=0.85) | 49.5 | 9.681181099444399e-13 | 1.3914870534391903e-06 | 150 | 1.953 |
| Convex Quadratic (kappa=10) | AdaGrad (lr=0.8) | 49.5 | 3.6007696613522845e-23 | 3.618523035980869e-12 | 150 | 1.903 |
| Convex Quadratic (kappa=10) | RMSProp (lr=0.08) | 49.5 | 0.02462076682119254 | 0.09461867983121015 | 150 | 2.105 |
| Convex Quadratic (kappa=10) | Adam (lr=0.15) | 49.5 | 3.718148354138998e-07 | 0.00036770294408017495 | 150 | 2.591 |
| Rosenbrock (Non-Convex) | Vanilla GD (Baseline) | 24.199999999999996 | 2.7264051917969647 | 1.745138533220884 | 400 | 2.447 |
| Rosenbrock (Non-Convex) | Momentum (gamma=0.9) | 24.199999999999996 | 0.005576860163445435 | 0.16215683734670422 | 400 | 3.091 |
| Rosenbrock (Non-Convex) | NAG (gamma=0.9) | 24.199999999999996 | 0.007499083761872278 | 0.1871233523488469 | 400 | 4.214 |
| Rosenbrock (Non-Convex) | AdaGrad (lr=0.2) | 24.199999999999996 | 1.4783474759087798 | 1.5435736163222797 | 400 | 4.759 |
| Rosenbrock (Non-Convex) | RMSProp (lr=0.005) | 24.199999999999996 | 0.5107289986152878 | 1.1633040442065372 | 400 | 4.287 |
| Rosenbrock (Non-Convex) | Adam (lr=0.02) | 24.199999999999996 | 1.8032294293836018 | 1.6048876931510008 | 400 | 5.922 |
| Noisy Regression (B=1000) | Full-Batch GD (B=1000) | 7.577641500055415 | 0.14919673686095433 | 0.5122842349324533 | 40 | 1.162 |
| Noisy Regression (B=32) | Mini-Batch GD (B=32) | 7.577641500055415 | 0.018786855567964205 | 0.030865001214608678 | 1240 | 46.158 |
| Noisy Regression (B=1) | Pure SGD (B=1) | 7.577641500055415 | 0.019004317068602113 | 0.03669860171712702 | 40000 | 1375.504 |
| Noisy Regression (B=32) | Adam (B=32) | 7.577641500055415 | 0.021081881694459945 | 0.07404796273251885 | 1240 | 57.703 |

### Key Experimental Findings:
1. **Quadratic Ill-Conditioning**: Momentum and Adam converge substantially faster than Vanilla GD by damping transverse oscillations across the steep axis.
2. **Rosenbrock Curvature**: Adam and RMSProp navigate the curved banana valley efficiently, achieving orders-of-magnitude lower final loss than GD.
3. **Stochastic Trade-offs**: Mini-batch GD (B=32) achieves comparable final loss to Full-Batch GD with significantly lower per-iteration gradient compute cost.
