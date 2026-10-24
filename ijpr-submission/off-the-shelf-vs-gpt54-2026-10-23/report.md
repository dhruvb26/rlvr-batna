# Revised results against GPT-5.4

Revised trained-model rows (Threshold, Surplus and Uniform) average seeds 42, 43 and 44. Off-the-shelf models have one evaluation per scenario. BR cells show mean clipped bargained ratio (deal %); — indicates unavailable results.

## Table 3 — mean clipped bargained ratio (deal %)

| Learner | AHP | CA | CRA | DnD | JI |
|---|---:|---:|---:|---:|---:|
| Threshold (Ours) | 0.51 (80) | 0.62 (95) | 0.44 (87) | 0.75 (98) | 0.68 (95) |
| Surplus | 0.41 (88) | 0.59 (97) | 0.32 (92) | 0.75 (97) | 0.66 (98) |
| Uniform (new) | 0.51 (82) | 0.61 (97) | 0.44 (86) | 0.76 (93) | 0.70 (99) |
| Qwen3-30B | -0.12 (73) | 0.52 (89) | -0.26 (75) | 0.60 (97) | 0.56 (93) |
| Qwen3-30B-T | — | — | — | — | — |
| Qwen3-235B | 0.20 (68) | 0.53 (91) | -0.03 (71) | 0.64 (96) | 0.64 (88) |
| Qwen3-235B-T | 0.13 (78) | 0.52 (95) | 0.04 (74) | 0.70 (99) | 0.66 (89) |
| Qwen3.6-35B | 0.14 (67) | 0.50 (98) | 0.03 (69) | 0.64 (93) | 0.66 (90) |
| GPT-5.4 | 0.24 (82) | 0.52 (96) | 0.07 (83) | 0.69 (96) | 0.68 (85) |
| GPT-5.4-mini | 0.14 (67) | 0.47 (99) | -0.13 (75) | 0.65 (97) | 0.65 (92) |
| DeepSeek-V3.1 | 0.13 (74) | 0.48 (85) | 0.03 (66) | 0.69 (89) | 0.72 (100) |
| Kimi-K2-T | 0.20 (72) | 0.51 (98) | 0.05 (67) | 0.70 (100) | 0.68 (87) |
| Llama-4-Maverick | 0.01 (65) | 0.50 (100) | -0.17 (71) | 0.64 (100) | 0.69 (95) |

## Table 4 — Pareto efficiency and first-bid ratios

| Learner | Pareto CA ↑ | Pareto DnD ↑ | First bid AHP ↓ | First bid CRA ↓ |
|---|---:|---:|---:|---:|
| Threshold (Ours) | 0.4490 | 0.7609 | 0.4392 | 0.4068 |
| Surplus | 0.4659 | 0.6769 | 0.5217 | 0.4972 |
| Uniform (new) | 0.4665 | 0.7469 | 0.4543 | 0.4282 |
| Qwen3-30B | 0.5000 | 0.4330 | 0.9828 | 0.9834 |
| Qwen3-30B-T | — | — | — | — |
| Qwen3-235B | 0.4425 | 0.6458 | 0.8022 | 0.8355 |
| Qwen3-235B-T | 0.4661 | 0.6970 | 0.7730 | 0.7972 |
| Qwen3.6-35B | 0.4628 | 0.5806 | 0.7708 | 0.7693 |
| GPT-5.4 | 0.4706 | 0.6458 | 0.7141 | 0.7458 |
| GPT-5.4-mini | 0.3577 | 0.6082 | 0.8359 | 0.8892 |
| DeepSeek-V3.1 | 0.4381 | 0.6517 | 0.7797 | 0.8138 |
| Kimi-K2-T | 0.6198 | 0.7100 | 0.8021 | 0.8103 |
| Llama-4-Maverick | 0.4597 | 0.5600 | 0.8801 | 0.8881 |
