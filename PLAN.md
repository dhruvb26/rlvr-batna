# Walking Away from Bad Deals: BATNA-Aware Reward Design for Negotiation

Negotiation is hard for language models, even frontier ones (see [Zhan et al., 2024](https://arxiv.org/abs/2402.01097) for a survey of the field). We want to use reinforcement learning with verifiable rewards to train a local negotiation agent that performs well across tasks. There have been primarily two kinds of negotiation scenarios in the literature - distributive (single-issue price) and integrative, following the [Multi-Issue Bargaining Task (MIBT) framework (Fershtman, 1990)](https://doi.org/10.1016/0899-8256(90)90024-O) for multi-issue resource allocation.

For single-issue price we have AmazonPriceHistory and CraigslistBargains. For multi-issue resource allocation we have CaSiNo and DnD. There is also a hybrid dataset that combines both - Job Interview. We want to show whether training across these scenario types produces an agent that generalizes.

The goal is simple - have a hypothesis and do extensive validation/testing to see if it holds. Since we're using reinforcement learning, choosing a local model to train makes sense. From the literature, `Qwen3-30B-A3B` is the best model for this. First step is to get baseline results of how different models perform across all these datasets.

## Evaluation Metrics

These metrics are computed across all datasets:

- **[Bargained ratio](https://en.wikipedia.org/wiki/Zone_of_possible_agreement)**: Fraction of maximum possible utility captured by the learner, normalized to [0, 1]. Answers "how much of what you could have gotten did you actually get?" Computed differently per scenario type (see table below) but semantically identical.
- **[Deal rate](https://aclanthology.org/2024.findings-acl.213.pdf)**: Fraction of negotiations ending in agreement.

These metrics are scenario-specific:

- **First bid ratio** *(single-issue only)*: The learner's opening price as a fraction of their budget (buyer) or as a multiple of their cost (seller). Measures [anchoring](https://doi.org/10.1037/0022-3514.81.4.657); first offers predict final prices (Galinsky & Mussweiler, 2001). Identified as a key failure mode in [Are LLMs Effective Negotiators?](https://arxiv.org/abs/2402.15813) where LLMs open too close to their reservation price, losing surplus.
- **[Pareto efficiency](https://en.wikipedia.org/wiki/Pareto_efficiency)** *(multi-item only)*: % of deals where no reallocation of items could improve one side without hurting the other. Measures whether the agent discovers integrative (win-win) trades rather than naive 50/50 splits.
- **Joint surplus** *(multi-item only)*: Sum of both parties' points as a fraction of the maximum possible joint score. Higher = less value destroyed.

## Datasets & Scenarios

| Dataset | Scenario | Example | Bargained Ratio |
|---|---|---|---|
| [**CaSiNo**](https://convokit.cornell.edu/documentation/casino-corpus.html) (1,030 scenarios) | Multi-item (integrative)  - Two campsite neighbors split 3 packages each of food, water, firewood. Private priority orderings: High (5 pts), Medium (4 pts), Low (3 pts). Max possible = 36 pts. | Learner values food=High(5), water=Med(4), firewood=Low(3). Deal gives learner food:2, water:2, firewood:1. Points = 2×5 + 2×4 + 1×3 = 21. Ratio = 21/36 = **0.58**. | `learner_points / 36` |
| [**DnD**](https://github.com/facebookresearch/end-to-end-negotiator) (10,095 scenarios) | Multi-item (integrative)  - Two parties split books, hats, balls. Variable item counts and private point values per scenario. | Learner has 3 books(value=2), 1 hat(value=4), 2 balls(value=1). Max = 3×2+1×4+2×1 = 12. Deal gives learner 2 books, 1 hat, 0 balls. Points = 2×2+1×4+0 = 8. Ratio = 8/12 = **0.67**. | `learner_points / max_possible` |
| [**AmazonHistoryPrice**](https://github.com/TianXiaSJTU/AmazonPriceHistory) (~515 products) | Buyer/seller (distributive)  - Buyer (budget B = current market price) and seller (cost C = historical low) negotiate a single price for a product. 18 product categories. | Product: headphones. B=$90 (current), C=$60 (lowest). Surplus = $30. Deal at P=$72. Buyer gain = 90-72 = $18. Ratio = 18/30 = **0.60**. | `(B - P) / \|B - C\|` |
| [**Craigslist Bargains**](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains) (6,682 dialogues) | Buyer/seller (distributive)  - Buyer and seller negotiate price of a Craigslist listing. Different product domain from Amazon. | Listing: used laptop at $500. B=$400, C=$250. Surplus = $150. Deal at P=$310. Buyer gain = 400-310 = $90. Ratio = 90/150 = **0.60**. | `(B - P) / \|B - C\|` |
| [**Job Interview**](https://github.com/gucci-j/negotiation-breakdown-detection) (2,639 dialogues) - *held-out, eval only* | Hybrid (multi-attribute)  - Recruiter and applicant negotiate salary ($20-50/hr), position (4 options), company (4 options), workplace (4 cities), weekly holiday (2-6 days). Position×Company have interdependent utilities → non-linear scoring. ~9,920 possible deals. | Applicant weights: Salary=0.24, Position+Company=0.19, Holiday=0.27, Workplace=0.15, Company=0.24. Deal: $39/hr, Manager, Apple, Sydney, 4 days. U(deal) computed via weighted sum with interdependency bias. U_max found by brute-force over all 9,920 combinations. Ratio = U(deal)/U_max = **0.72**. | `U(deal) / U_max` |

## Unified Reward (Training)

The reward is **dataset-aware**: multi-item scenarios (Casino, DnD, JI) use a BATNA threshold to penalize bad deals, while price environments (Amazon, Craigslist) use pure surplus with a floor at 0. This split was motivated by reward hacking observed in price environments — thin-margin scenarios (buyer_budget ≈ seller_cost) allowed the model to propose unrealistically low prices and achieve BR >> 1.0 against a lenient instruct opponent. The structural floor from the seller's reservation price was insufficient; the threshold=0.0 design lets gradient flow naturally without creating an exploitable cliff.

| Scenario type | Outcome | Reward |
|---|---|---|
| Multi-item (Casino, DnD, JI) | Deal with BR ≥ τ (0.4) | `bargained_ratio` (0.4–1.0) |
| Multi-item | Deal with BR < τ | `-0.5` (BATNA penalty) |
| Price (Amazon, Craigslist) | Deal | `bargained_ratio` (0.0–1.0) |
| All | Walk away / no deal / max turns | `0.0` |
| All | Format violation | `-1.0` |

Note: Qwen3-30B-A3B has 95-99% format compliance out of the box (observed in baseline evals), so the format violation penalty rarely fires. The reward signal is entirely focused on deal quality.

## Dataset Distribution

The natural scenario counts are heavily skewed, causing training signal imbalance:

| Dataset | Raw scenarios | Natural % | Recommended weight |
|---|---|---|---|
| DnD | ~5,000 | 48% | 0.25 |
| Craigslist | ~4,000 | 38% | 0.25 |
| Casino | ~900 | 9% | 0.25 |
| Amazon | ~515 | 5% | 0.25 |

With natural distribution, Amazon receives ~3 samples per batch (batch_size=64), making it highly volatile and susceptible to reward hacking from a single outlier scenario. Uniform weighting (0.25 each) ensures each dataset gets ~16 samples per batch, providing more stable learning across all domains. Dataset weights are configurable in `train/config.yaml`.

## Baselines

Evaluate all models on all 5 datasets (100 scenarios each) to establish pre-training performance. The opponent is always Qwen3-30B-A3B-Instruct (fixed, unmodified) for consistency across runs.

- [x] Qwen3-235B-A22B-Instruct  
- [x] Qwen3-235B-A22B-Thinking  
- [x] Qwen3-30B-A3B-Instruct  
- [x] Qwen3-30B-A3B-Thinking  
- [x] Qwen3-8B-Instruct  
- [x] GPT-5.4  
- [x] GPT-5.4-mini  
- [x] Kimi-K2-Thinking  
- [x] Llama-4-Maverick  
- [x] DeepSeek-V3.1

### Baseline Results

Bargained ratio (deal rate) per dataset. All models play as buyer/learner against Qwen3-30B-A3B-Instruct, 100 scenarios each.

| Model | Amazon | CaSiNo | Craigslist | DnD | JI (held-out) | Avg BR |
|---|---|---|---|---|---|---|
| **GPT-5.4** | **0.87** (0.93) | 0.52 (0.55) | **0.33** (0.80) | **0.66** (0.90) | 0.69 (0.72) | **0.61** |
| DeepSeek-V3.1 | 0.71 (0.94) | **0.45** (0.67) | 0.24 (0.89) | 0.61 (0.91) | **0.75** (0.83) | 0.55 |
| Qwen3-235B-A22B-T | 0.71 (0.94) | 0.49 (0.61) | 0.13 (0.86) | 0.59 (0.77) | 0.74 (0.73) | 0.53 |
| Qwen3-235B-A22B | 0.46 (0.89) | 0.53 (0.55) | 0.21 (0.79) | 0.63 (0.82) | 0.74 (0.72) | 0.51 |
| Kimi-K2-Thinking | 0.26 (0.93) | 0.50 (0.80) | 0.16 (0.85) | 0.56 (0.80) | 0.73 (0.87) | 0.44 |
| Qwen3-30B-A3B | -0.12 (0.64) | 0.53 (0.45) | 0.03 (0.57) | 0.64 (0.65) | 0.70 (0.52) | 0.36 |
| Qwen3-30B-A3B-T | -0.16 (0.81) | 0.47 (0.69) | 0.02 (0.70) | 0.49 (0.71) | 0.78 (0.46) | 0.32 |
| Qwen3-8B | -0.36 (0.72) | 0.44 (0.85) | 0.04 (0.54) | 0.52 (0.79) | 0.72 (0.74) | 0.27 |
| Llama-4-Maverick | -0.49 (0.88) | 0.49 (0.74) | -0.09 (0.76) | 0.53 (0.87) | 0.75 (0.90) | 0.24 |
| GPT-5.4-mini | 1.62† (0.84) | 0.49 (0.56) | 0.17 (0.76) | 0.64 (0.80) | 0.70 (0.79) | 0.72† |

†GPT-5.4-mini Amazon BR >1.0 indicates reward hacking — the model exploits thin-margin scenarios by proposing prices below the seller's cost, which the instruct opponent sometimes accepts. This is exactly the failure mode our BATNA threshold is designed to address.

Key observations:
- **Price scenarios are hard**: Most models have negative or near-zero BR on Amazon/Craigslist as buyers. The Qwen opponent as seller anchors strongly, and smaller models cave to the seller's price.
- **Multi-item scenarios are easier**: CaSiNo/DnD/JI BRs cluster around 0.45–0.75 across models, suggesting the cooperative opponent accepts reasonable splits.
- **Deal rate vs quality tradeoff**: Qwen3-30B-A3B-Instruct (our base model) has the lowest deal rates (0.45–0.65) but decent BR when deals happen. Thinking mode improves deal rate but tanks BR on price scenarios.
- **Frontier gap**: GPT-5.4 leads with 0.61 avg BR. Our base model (Qwen3-30B-A3B) sits at 0.36 — a 0.25 gap to close with training.

## Experimental Plan

Fixed opponent across all baselines and training: `Qwen3-30B-A3B-Instruct`. This keeps the reward landscape stationary and results comparable across experiments. For additional analysis, we may test trained models against harder opponents (GPT-5.4, GPT-5.4-mini) to measure robustness.

**Step 1: Baselines** - Run all models above on all 5 datasets. No training, just measure how well each model negotiates out of the box. This gives us the "before" numbers. **Done.**


**Step 2: Surplus reward** - Train Qwen3-30B-A3B with GRPO using simple surplus as the reward (`bargained_ratio` for deals, 0 for no deal), following [Liu et al. (2026)](https://arxiv.org/abs/2604.09855). Applied across all 4 training datasets with uniform weighting. 50 steps, batch_size=64, group_size=8. **Done.**

**Step 3: BATNA-aware reward (our contribution)** - Train Qwen3-30B-A3B with GRPO using the dataset-aware combined reward explained in the Unified Reward section. 70 steps, batch_size=64, group_size=8. **Done.**

### Training Results

Comparison of base Qwen3-30B-A3B-Instruct vs. surplus-only reward vs. BATNA-aware reward. All evaluated against same Qwen3-30B-A3B-Instruct opponent, 100 scenarios each.

| Model | Amazon | CaSiNo | Craigslist | DnD | JI (held-out) | Avg BR |
|---|---|---|---|---|---|---|
| Qwen3-30B base | -0.12 (0.64) | 0.53 (0.45) | 0.03 (0.57) | 0.64 (0.65) | 0.70 (0.52) | 0.36 |
| + Surplus reward | 0.58 (0.73) | 0.53 (0.90) | 0.53 (0.89) | 0.48 (0.92) | 0.59 (0.85) | 0.54 |
| + **BATNA reward** | **0.64** (0.82) | **0.59** (0.67) | **0.61** (0.96) | **0.71** (0.83) | **0.69** (0.74) | **0.65** |

**Multi-item metrics (CaSiNo + DnD):**

| Model | CaSiNo Pareto % | CaSiNo Joint Surplus | DnD Pareto % | DnD Joint Surplus |
|---|---|---|---|---|
| Qwen3-30B base | 51.1% | 0.956 | 33.9% | 0.815 |
| + Surplus reward | 43.3% | 0.945 | 31.5% | 0.745 |
| + BATNA reward | **53.7%** | **0.955** | **47.0%** | 0.824 |

**Price scenario anchoring (first bid ratio):**

| Model | Amazon 1st Bid | Craigslist 1st Bid |
|---|---|---|
| Qwen3-30B base | 0.927 | 0.911 |
| + Surplus reward | 0.579 | 0.613 |
| + BATNA reward | **0.491** | **0.551** |

Key findings:
- **BATNA closes the frontier gap**: Base model averaged 0.36 BR → BATNA achieves 0.65, surpassing GPT-5.4 (0.61). A +0.29 absolute improvement.
- **BATNA > Surplus everywhere**: BATNA outperforms surplus reward on all 5 datasets including the held-out JI scenario (+0.10 on JI), confirming the BATNA threshold provides a better learning signal.
- **Surplus reward hurts multi-item quality**: Surplus-trained model has lower BR than base on DnD (0.48 vs 0.64) and JI (0.59 vs 0.70) — it learns to close deals quickly at any price. BATNA avoids this by penalizing bad deals.
- **Deal rates improve across the board**: Both trained models dramatically improve deal rates over base (0.57 avg → 0.80 BATNA, 0.86 surplus). Surplus achieves higher deal rates but at the cost of deal quality.
- **Anchoring improves**: Both trained models learn to open with more aggressive first bids (lower ratio = buyer opening further from budget). BATNA anchors even more aggressively (0.49 vs 0.58 on Amazon).
- **Pareto efficiency**: BATNA improves Pareto efficiency on both CaSiNo (+2.6pp) and DnD (+13.1pp), while surplus degrades it. BATNA-trained agents discover better integrative trades.
- **Generalization to held-out JI**: BATNA achieves 0.69 BR on JI (a novel hybrid scenario never seen in training) vs 0.70 for the base model, while surplus drops to 0.59. BATNA preserves generalization; surplus overfits to deal-closing behavior.

### Why BATNA-aware rewards

The BATNA threshold is grounded in negotiation theory and supported by concurrent findings in the literature:

- **Negotiation theory**: BATNA (Best Alternative to Negotiated Agreement) is the foundational concept in principled negotiation ([Fisher, Ury & Patton, 1981](https://en.wikipedia.org/wiki/Getting_to_Yes)). A rational agent should reject any deal worse than their outside option. Our threshold encodes this directly into the reward.
- **[Bergemann et al. (2026)](https://arxiv.org/abs/2604.16472)** diagnose exactly this problem in bilateral trade: RLVR-trained agents learn to close deals at any price rather than walk away, because simple surplus rewards provide a positive signal for bad deals. They propose a BATNA-like threshold as future work - we implement it.
- **[Chawla et al. (2023)](https://arxiv.org/abs/2310.14404)** show in EMNLP 2023 that a purely selfish reward outperforms Fehr-Schmidt fairness-aware rewards when paired with a tough opponent. This supports our choice of a selfish surplus reward with a quality floor rather than a fairness-weighted objective.
- **Structural asymmetry**: In buyer/seller scenarios, the opponent's reservation price naturally prevents terrible deals (a seller won't accept below cost). In multi-item scenarios no such structural floor exists - a cooperative opponent will accept any split, so the agent has no negative signal to learn from. The threshold provides this missing signal.

## Analysis & Findings
Throughout training and evaluation, document interesting emergent behaviors as done in [Liu et al. (2026)](https://arxiv.org/abs/2604.09855).

[.] How do negotiation strategies evolve over training steps?
[.] Does the model learn to anchor more aggressively (first bid ratio)?
[.] How do concession patterns change with training?
[.] Does deal quality distribution shift (fewer bad deals with BATNA)?
[.] Are strategies different across scenario types (integrative vs distributive)?
[.] Does the model learn to walk away from bad deals?
[.] Is there evidence of log-rolling in multi-item scenarios?
[.] Does Pareto efficiency improve with training?
[.] How does the model behave on JI zero-shot (novel scenario type)?
[.] Collect qualitative examples of interesting negotiation behaviors