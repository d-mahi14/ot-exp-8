# Parallel Optimization Scaling Benchmarks

## Benchmark Summary

- Workload: 48 compute-intensive tasks (80000 iterations/task)
- Sequential Baseline T(1): 1.302 s
- Fitted Amdahl's Law: Parallelizable fraction $s = 0.430$, Serial fraction $1-s = 0.570$
- Fitted Gustafson's Law: Serial fraction $\alpha = 0.891$

### Empirical Scaling Data Across Worker Counts:

| scheduler | workers | time_sec | speedup | efficiency | overhead_sec | throughput_tasks_sec |
| --- | --- | --- | --- | --- | --- | --- |
| Static_ThreadPool | 1 | 1.3016588003374636 | 1.0 | 1.0 | 0.0 | 36.87602310801854 |
| Static_ThreadPool | 2 | 1.401948899962008 | 0.9284637980547921 | 0.46423189902739603 | 1.5022389995865524 | 34.23805247202717 |
| Static_ThreadPool | 4 | 0.8810830996371806 | 1.4773394256154397 | 0.36933485640385993 | 2.2226735982112586 | 54.4784027973818 |
| Static_ThreadPool | 8 | 0.7542543997988105 | 1.7257556610669658 | 0.21571945763337072 | 4.73237639805302 | 63.63900563629924 |
| Dynamic_WorkQueue | 1 | 1.3016588003374636 | 1.0 | 1.0 | 0.0 | 36.87602310801854 |
| Dynamic_WorkQueue | 2 | 1.5322537003085017 | 0.8495060576948775 | 0.42475302884743876 | 1.7628486002795398 | 31.326405013958034 |
| Dynamic_WorkQueue | 4 | 1.3339832001365721 | 0.9757685105810934 | 0.24394212764527334 | 4.034274000208825 | 35.982462144265234 |
| Dynamic_WorkQueue | 8 | 1.0849835998378694 | 1.1997036642138852 | 0.14996295802673565 | 7.378209998365492 | 44.24030004432574 |
| WorkStealing_Simulation | 1 | 2.892701900098473 | 0.4499802763268323 | 0.4499802763268323 | 1.5910430997610092 | 16.593483067980834 |
| WorkStealing_Simulation | 2 | 1.5101536996662617 | 0.8619379607685796 | 0.4309689803842898 | 1.7186485989950597 | 31.78484415898052 |
| WorkStealing_Simulation | 4 | 0.8493357999250293 | 1.5325608557326333 | 0.38314021393315834 | 2.0956843993626535 | 56.51474953044126 |
| WorkStealing_Simulation | 8 | 0.723602999933064 | 1.798857661532459 | 0.22485720769155737 | 4.487165199127048 | 66.33471669470715 |


### Analytical Comparison: Data Parallel vs Pipeline Parallel

| Model | Overhead Mechanism | Scaling Behavior |
|---|---|---|

| **Data Parallel (DDP)** | Ring All-Reduce communication $\frac{2(P-1)}{P} \times \text{ModelSize}$ | Efficient for large batches; bottlenecked by network bandwidth when compute/comm ratio is low. |

| **Pipeline Parallel (1F1B)** | Pipeline bubble fraction $\frac{K-1}{M + K - 1}$ | Memory-efficient for models exceeding single-device capacity; bubble overhead vanishes as microbatch count $M \gg K$. |
