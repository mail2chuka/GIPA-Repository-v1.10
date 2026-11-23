# GIPA Seed-Level Statistical Analysis Report

## Reproduction check

The modified experiment completed all 30 matched runs for both workload regimes, all four fleet densities, all principal policies, and all GIPA ablations. The regenerated aggregate file has MD5 fingerprint `b4b3c084c3b8c97562c6c58097b6b923`, which is identical to the archived principal result. Retaining counts and seed-level observations therefore did not alter the simulation stream or the reported aggregate means.

Confidence intervals use 10,000 seed-level bootstrap resamples with bootstrap seed 20260804. Policy comparisons use paired differences under common seed indices.

## Principal GIPA result at Regime A and N = 160

| Measure | Mean | Seed-level 95% bootstrap CI |
|---|---:|---:|
| Mean latency | 18.377 ms | [18.150, 18.611] ms |
| p95 latency | 24.003 ms | [23.777, 24.232] ms |
| Overall offload ratio | 48.142% | [47.392%, 48.895%] |
| Safety offload ratio | 0.000% | [0.000%, 0.000%] |
| Background offload ratio | 68.974% | [68.782%, 69.169%] |
| Jain fairness | 0.97615 | [0.97514, 0.97719] |
| Stylized energy | 2.532 | [2.526, 2.538] |
| Maximum RSU load | 29.936 | [29.481, 30.390] |

No safety or background miss was observed in any of the 30 runs. Across all evaluated slots, the experiment observed 65,250 safety tasks and 150,750 background tasks. Under a direct binomial interpretation, the one-sided 95% Clopper-Pearson upper bounds are 0.00459% for safety misses and 0.00199% for background misses. These are descriptive task-level bounds. The manuscript should continue to state that no misses were observed rather than claiming a mathematically zero miss probability.

## GIPA versus all-local execution

The paired mean-latency difference, defined as GIPA minus all-local, is 0.0024 ms with a 95% bootstrap interval of [-0.1071, 0.1149] ms. The paired TOST with a pre-specified equivalence margin of 1 ms gives a 90% confidence interval of [-0.0940, 0.0988] ms. Both one-sided tests reject differences outside the equivalence region, with p-values below 3 x 10^-17. Mean-latency equivalence within 1 ms is therefore supported.

GIPA increases the overall offload ratio by 48.142 percentage points, with a 95% interval of [47.373, 48.882] percentage points. GIPA also reduces the stylized energy value by 2.022 units, with a 95% interval of [-2.053, -1.990]. Jain fairness cannot be compared with all-local execution because fairness is undefined when no RSU receives a task.

GIPA's p95 latency is 2.003 ms higher than all-local execution, with a 95% interval of [1.774, 2.224] ms. The supported claim is therefore mean-latency parity with substantial infrastructure use, not tail-latency equivalence.

## GIPA versus reactive best response

GIPA reduces mean latency by 8.825 ms, with a 95% paired interval of [8.678, 8.972] ms, and reduces p95 latency by 7.340 ms, with a 95% interval of [7.093, 7.583] ms. GIPA also reduces the safety miss rate by 51.11 percentage points and the background miss rate by 9.25 percentage points. The maximum RSU load is lower by 0.573 tasks on average, with a 95% interval of [0.085, 1.065] tasks.

Reactive best response has a numerically higher Jain index by approximately 0.0114, but that fairness measure occurs alongside nonzero deadline failures. Fairness must therefore be interpreted jointly with latency and miss rates.

## GIPA versus masked softmax

GIPA reduces mean latency by 4.413 ms, with a 95% interval of [4.241, 4.589] ms, and reduces p95 latency by 6.794 ms, with a 95% interval of [6.633, 6.953] ms. Both policies record no observed misses. GIPA offloads 21.65 percentage points less overall and reduces maximum RSU load by 12.515 tasks, with a 95% interval of [12.265, 12.770] tasks.

The stylized energy value is 0.909 units higher for GIPA. This trade-off should be discussed only within the stated stylized energy model.

## GIPA versus local-dominant matched alternatives

GIPA and prior-adjusted masked softmax have virtually identical mean latency. The paired difference is 0.0025 ms with a 95% interval of [-0.1084, 0.1148] ms. GIPA offloads 48.141 percentage points more workload, but its p95 latency is 2.003 ms higher. Fairness should not be compared because prior-adjusted masked softmax has only one seed with a defined seed-level fairness summary.

GIPA and entropy-regularized masked selection also have similar mean latency. The paired difference is 0.0624 ms with a 95% interval of [-0.0488, 0.1701] ms. GIPA offloads 47.709 percentage points more workload, while its p95 latency is 2.003 ms higher.

These comparisons support the claim that GIPA achieves substantially greater RSU participation at similar mean latency. They do not support a tail-latency advantage over local-dominant policies.

## Ablation results

Full GIPA has mean latency 0.1778 ms higher than fixed-weight GIPA, with a 95% interval of [0.1659, 0.1898] ms. Full GIPA also has mean latency 0.1823 ms higher than binary-weight GIPA, with a 95% interval of [0.1707, 0.1948] ms. These differences are statistically resolved under the paired design but remain below 0.2 ms. The fixed and binary variants offload approximately 1.29 and 1.33 percentage points less overall.

Removing diffusion produces mean latency nearly identical to full GIPA but reduces offloading by 48.141 percentage points. Diffusion is therefore the component that sustains nontrivial RSU participation in the evaluated configuration.

Removing the prior increases mean latency by 1.023 ms, with a 95% interval of [0.975, 1.070] ms, increases p95 latency by 1.929 ms, and increases overall offloading by 6.454 percentage points. It also raises maximum RSU load by 3.667 tasks. The prior therefore moderates aggressive background offloading and reduces peak concentration and latency.

The ablation evidence does not establish a latency benefit from annealing or graded amplification. It does establish strong operational roles for the congestion prior and diffusion floor.

## Manuscript-ready statistical statement

Values are means over 30 matched simulation runs. Error bars show seed-level 95% percentile-bootstrap confidence intervals based on 10,000 resamples. Policy comparisons use paired differences obtained from common environment and action-seed indices. For Regime A at N = 160, paired two one-sided tests support equivalence between GIPA and all-local mean latency within the pre-specified 1 ms margin. No deadline miss was observed for GIPA across 65,250 safety tasks and 150,750 background tasks. These zero empirical rates are reported as observed outcomes rather than proof of zero population miss probability.
