## Training configs

| Dataset | Opponent | Configuration | Replacements | Episodes |
| --- | --- | --- | --- | --- |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Surplus | 1,823 | 30,456 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Selective τ=.2 | 1,610 | 30,456 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Selective τ=.4 | 1,637 | 30,456 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Uniform τ=.4 | 1,628 | 30,456 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Selective τ=.6 | 1,733 | 30,456 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Surplus | 542 | 30,968 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Selective τ=.2 | 666 | 30,968 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Selective τ=.4 | 575 | 30,968 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Uniform τ=.4 | 582 | 30,968 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Selective τ=.6 | 536 | 30,968 |

## Eval configs

| Dataset | Opponent | Configuration | Replacements | Episodes |
| --- | --- | --- | --- | --- |
| Amazon | GPT-5.4 | Surplus | 4 | 279 |
| Amazon | GPT-5.4 | Selective τ=.4 | 0 | 279 |
| Amazon | GPT-5.4 | Uniform τ=.4 | 0 | 279 |
| Amazon | GPT-5.4 | Selective τ=.2 | 1 | 279 |
| Amazon | GPT-5.4 | Selective τ=.6 | 1 | 279 |
| Amazon | GPT-5.4 | Untrained Qwen3.6 | 0 | 93 |
| Amazon | GPT-5.4 | Surplus — regulation Off | 0 | 150 |
| Amazon | GPT-5.4 | Selective τ=.4 — regulation Off | 0 | 150 |
| Amazon | GPT-5.4 | Surplus — regulation On (seed42) | 1 | 50 |
| Amazon | GPT-5.4 | Selective τ=.4 — regulation On (seed42) | 1 | 50 |
| Amazon | GPT-5.4-mini | Surplus | 9 | 279 |
| Amazon | GPT-5.4-mini | Selective τ=.4 | 21 | 279 |
| Amazon | GPT-5.4-mini | Uniform τ=.4 | 16 | 279 |
| Amazon | GPT-5.4-mini | Selective τ=.2 | 30 | 279 |
| Amazon | GPT-5.4-mini | Selective τ=.6 | 38 | 279 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Surplus | 15 | 279 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Selective τ=.4 | 25 | 279 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Uniform τ=.4 | 16 | 279 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Selective τ=.2 | 14 | 279 |
| Amazon | Qwen3.6-35B-A3B (Tinker) | Selective τ=.6 | 9 | 279 |
| Amazon | Qwen3-30B Instruct | Surplus | 60 | 279 |
| Amazon | Qwen3-30B Instruct | Selective τ=.4 | 61 | 279 |
| Amazon | Qwen3-30B Instruct | Uniform τ=.4 | 49 | 279 |
| Amazon | Qwen3-30B Thinking | Surplus | 0 | 279 |
| Amazon | Qwen3-30B Thinking | Selective τ=.4 | 0 | 279 |
| Amazon | Qwen3-30B Thinking | Uniform τ=.4 | 0 | 279 |
| Amazon | Qwen3-235B Instruct | Surplus | 16 | 279 |
| Amazon | Qwen3-235B Instruct | Selective τ=.4 | 31 | 279 |
| Amazon | Qwen3-235B Instruct | Uniform τ=.4 | 18 | 279 |
| Amazon | Qwen3-235B Thinking | Surplus | 3 | 279 |
| Amazon | Qwen3-235B Thinking | Selective τ=.4 | 4 | 279 |
| Amazon | Qwen3-235B Thinking | Uniform τ=.4 | 7 | 279 |
| Amazon | Llama 4 Maverick | Surplus | 67 | 279 |
| Amazon | Llama 4 Maverick | Selective τ=.4 | 68 | 279 |
| Amazon | Llama 4 Maverick | Uniform τ=.4 | 68 | 279 |
| Amazon | Kimi K2 Thinking | Surplus | 0 | 279 |
| Amazon | Kimi K2 Thinking | Selective τ=.4 | 1 | 279 |
| Amazon | Kimi K2 Thinking | Uniform τ=.4 | 1 | 279 |
| Craigslist | GPT-5.4 | Surplus | 0 | 300 |
| Craigslist | GPT-5.4 | Selective τ=.4 | 0 | 300 |
| Craigslist | GPT-5.4 | Uniform τ=.4 | 0 | 300 |
| Craigslist | GPT-5.4 | Selective τ=.2 | 0 | 300 |
| Craigslist | GPT-5.4 | Selective τ=.6 | 0 | 300 |
| Craigslist | GPT-5.4 | Untrained Qwen3.6 | 0 | 100 |
| Craigslist | GPT-5.4 | Surplus — regulation Off | 0 | 150 |
| Craigslist | GPT-5.4 | Selective τ=.4 — regulation Off | 0 | 150 |
| Craigslist | GPT-5.4 | Surplus — regulation On (seed42) | 0 | 50 |
| Craigslist | GPT-5.4 | Selective τ=.4 — regulation On (seed42) | 0 | 50 |
| Craigslist | GPT-5.4-mini | Surplus | 1 | 300 |
| Craigslist | GPT-5.4-mini | Selective τ=.4 | 0 | 300 |
| Craigslist | GPT-5.4-mini | Uniform τ=.4 | 1 | 300 |
| Craigslist | GPT-5.4-mini | Selective τ=.2 | 0 | 300 |
| Craigslist | GPT-5.4-mini | Selective τ=.6 | 7 | 300 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Surplus | 5 | 300 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Selective τ=.4 | 10 | 300 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Uniform τ=.4 | 1 | 300 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Selective τ=.2 | 8 | 300 |
| Craigslist | Qwen3.6-35B-A3B (Tinker) | Selective τ=.6 | 14 | 300 |
| Craigslist | Qwen3-30B Instruct | Surplus | 18 | 300 |
| Craigslist | Qwen3-30B Instruct | Selective τ=.4 | 32 | 300 |
| Craigslist | Qwen3-30B Instruct | Uniform τ=.4 | 34 | 300 |
| Craigslist | Qwen3-30B Thinking | Surplus | 0 | 300 |
| Craigslist | Qwen3-30B Thinking | Selective τ=.4 | 0 | 300 |
| Craigslist | Qwen3-30B Thinking | Uniform τ=.4 | 0 | 300 |
| Craigslist | Qwen3-235B Instruct | Surplus | 2 | 300 |
| Craigslist | Qwen3-235B Instruct | Selective τ=.4 | 3 | 300 |
| Craigslist | Qwen3-235B Instruct | Uniform τ=.4 | 4 | 300 |
| Craigslist | Qwen3-235B Thinking | Surplus | 2 | 300 |
| Craigslist | Qwen3-235B Thinking | Selective τ=.4 | 0 | 300 |
| Craigslist | Qwen3-235B Thinking | Uniform τ=.4 | 2 | 300 |
| Craigslist | Llama 4 Maverick | Surplus | 6 | 300 |
| Craigslist | Llama 4 Maverick | Selective τ=.4 | 16 | 300 |
| Craigslist | Llama 4 Maverick | Uniform τ=.4 | 19 | 300 |
| Craigslist | Kimi K2 Thinking | Surplus | 1 | 300 |
| Craigslist | Kimi K2 Thinking | Selective τ=.4 | 0 | 300 |
| Craigslist | Kimi K2 Thinking | Uniform τ=.4 | 0 | 300 |
