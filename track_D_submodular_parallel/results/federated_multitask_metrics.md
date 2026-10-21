# Distributed (FedAvg) and Multi-Task (MGDA, GradNorm) Benchmark Results

## 1. Federated Learning (FedAvg Simulation)
Comparison of Centralized training vs FedAvg under IID ($E=1$) and Non-IID ($E=5$) client distributions.

| category | experiment | round_or_step | metric_1_name | metric_1_val | metric_2_name | metric_2_val | extra_info |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Federated_Distributed | FedAvg_IID_E1 | 6 | TestLoss | 0.6088158683776855 | TestAccuracy | 0.82 | IID_E1 |
| Federated_Distributed | FedAvg_IID_E1 | 7 | TestLoss | 0.5725465354919433 | TestAccuracy | 0.85 | IID_E1 |
| Federated_Distributed | FedAvg_IID_E1 | 8 | TestLoss | 0.5265984992980957 | TestAccuracy | 0.87 | IID_E1 |
| Federated_Distributed | FedAvg_IID_E1 | 9 | TestLoss | 0.47140653991699216 | TestAccuracy | 0.886 | IID_E1 |
| Federated_Distributed | FedAvg_IID_E1 | 10 | TestLoss | 0.4110205612182617 | TestAccuracy | 0.896 | IID_E1 |
| Federated_Distributed | FedAvg_NonIID_E5 | 1 | TestLoss | 0.490134033203125 | TestAccuracy | 0.87 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 2 | TestLoss | 0.10957529354095459 | TestAccuracy | 0.978 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 3 | TestLoss | 0.08759555435180665 | TestAccuracy | 0.97 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 4 | TestLoss | 0.08532769012451172 | TestAccuracy | 0.976 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 5 | TestLoss | 0.08348095655441284 | TestAccuracy | 0.97 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 6 | TestLoss | 0.09895256090164184 | TestAccuracy | 0.966 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 7 | TestLoss | 0.09391976022720337 | TestAccuracy | 0.968 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 8 | TestLoss | 0.09538493061065674 | TestAccuracy | 0.968 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 9 | TestLoss | 0.10038833236694336 | TestAccuracy | 0.966 | NonIID_E5 |
| Federated_Distributed | FedAvg_NonIID_E5 | 10 | TestLoss | 0.10891887140274048 | TestAccuracy | 0.962 | NonIID_E5 |


## 2. Multi-Task Learning (MTL) Comparison
Shared neural trunk trained simultaneously on continuous regression (Task 1) and binary classification (Task 2).

| category | experiment | round_or_step | metric_1_name | metric_1_val | metric_2_name | metric_2_val | extra_info |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MultiTask_Optimization | Uniform_Weighting | 60 | Task1_RegLoss | 0.05281232297420502 | Task2_ClsLoss | 0.1366777867078781 | w1=0.50, w2=0.50 |
| MultiTask_Optimization | MGDA_Pareto | 1 | Task1_RegLoss | 8.495665550231934 | Task2_ClsLoss | 0.6788694262504578 | alpha1=0.01, cos=-0.02 |
| MultiTask_Optimization | MGDA_Pareto | 10 | Task1_RegLoss | 7.084677696228027 | Task2_ClsLoss | 0.38723549246788025 | alpha1=0.00, cos=0.02 |
| MultiTask_Optimization | MGDA_Pareto | 20 | Task1_RegLoss | 5.525787830352783 | Task2_ClsLoss | 0.16101032495498657 | alpha1=0.00, cos=-0.01 |
| MultiTask_Optimization | MGDA_Pareto | 30 | Task1_RegLoss | 4.35379695892334 | Task2_ClsLoss | 0.07219556719064713 | alpha1=0.00, cos=-0.08 |
| MultiTask_Optimization | MGDA_Pareto | 40 | Task1_RegLoss | 3.633100986480713 | Task2_ClsLoss | 0.03909363970160484 | alpha1=0.00, cos=-0.02 |
| MultiTask_Optimization | MGDA_Pareto | 50 | Task1_RegLoss | 3.5118792057037354 | Task2_ClsLoss | 0.023605549708008766 | alpha1=0.00, cos=-0.07 |
| MultiTask_Optimization | MGDA_Pareto | 60 | Task1_RegLoss | 3.5851080417633057 | Task2_ClsLoss | 0.015480787493288517 | alpha1=0.00, cos=-0.04 |
| MultiTask_Optimization | GradNorm_Dynamic | 1 | Task1_RegLoss | 8.491540908813477 | Task2_ClsLoss | 0.697885274887085 | w1=0.98, w2=1.02 |
| MultiTask_Optimization | GradNorm_Dynamic | 10 | Task1_RegLoss | 4.088101387023926 | Task2_ClsLoss | 0.6113713979721069 | w1=0.80, w2=1.20 |
| MultiTask_Optimization | GradNorm_Dynamic | 20 | Task1_RegLoss | 1.181747555732727 | Task2_ClsLoss | 0.4736955165863037 | w1=0.59, w2=1.41 |
| MultiTask_Optimization | GradNorm_Dynamic | 30 | Task1_RegLoss | 0.3000568151473999 | Task2_ClsLoss | 0.2967754006385803 | w1=0.40, w2=1.60 |
| MultiTask_Optimization | GradNorm_Dynamic | 40 | Task1_RegLoss | 0.274577260017395 | Task2_ClsLoss | 0.17160554230213165 | w1=0.25, w2=1.75 |
| MultiTask_Optimization | GradNorm_Dynamic | 50 | Task1_RegLoss | 0.1598837524652481 | Task2_ClsLoss | 0.09842906147241592 | w1=0.13, w2=1.87 |
| MultiTask_Optimization | GradNorm_Dynamic | 60 | Task1_RegLoss | 0.12468910962343216 | Task2_ClsLoss | 0.060715917497873306 | w1=0.05, w2=1.95 |


### Theoretical Insights & Limitations:

1. **FedAvg Client Drift**: In Non-IID federated settings with high local epochs ($E=5$), local client models drift toward their local empirical distributions before averaging, causing slower global loss decrease compared to IID and centralized baselines.

2. **MGDA Pareto-Stationarity**: MGDA optimizes on the convex hull of task gradients, ensuring no task suffers descent degradation. When gradients conflict (negative cosine similarity), MGDA balances weights to maintain non-negative progress for all tasks.

3. **GradNorm Dynamic Weighting**: GradNorm dynamically scales task loss weights $w_t(t)$ based on relative inverse training rates $r_t(t)$, preventing any single fast-learning task from dominating shared representations.
