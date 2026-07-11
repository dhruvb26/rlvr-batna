# rlvr-batna

BATNA-aware reinforcement learning from verifiable rewards for negotiation. Trains `Qwen3-30B-A3B` with GRPO across four negotiation datasets (evaluating on a fifth held-out set) using a dataset-aware reward that penalizes deals below a quality threshold — encoding the negotiation principle of [BATNA](https://en.wikipedia.org/wiki/Best_alternative_to_a_negotiated_agreement) directly into the reward signal.

See [`PLAN.md`](./PLAN.md) for the full experimental plan, baseline results, and training outcomes.

> **Note:** the original base model (`Qwen3-30B-A3B-Instruct-2507`) was retired by Tinker in June 2026. Supplementary experiments (e.g. the multi-item-only transfer ablation) use its replacement, `Qwen3.6-35B-A3B`, a hybrid thinking model — the `renderer_name` config field pins the non-thinking renderer for it.

## Papers

- **`emnlp-submission/`** — ACL/ARR short paper (+ rebuttal changelog in `files/`)
- **`ijpr-submission/`** — IJPR journal version

Both build with `make` (requires a LaTeX toolchain); `make clean` removes build artifacts.

## Stack

- **[Tinker](https://tinker.thinkingmachines.dev)** ([docs](https://tinker-docs.thinkingmachines.ai/)) — GRPO training and serving trained model checkpoints
- **[Weights & Biases](https://wandb.ai)** — experiment tracking (`wandb_project: rlvr-batna`)
- **[OpenRouter](https://openrouter.ai)** — frontier model inference for baselines and opponent

## Setup

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dhruvb26/rlvr-batna.git
cd rlvr-batna
uv sync
```

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
TINKER_API_KEY=...
WANDB_API_KEY=...
OPENROUTER_API_KEY=...
HF_TOKEN=...        # optional, for gated HF models
```

## Usage

### Evaluation

Run negotiation episodes for all matchups defined in `configs/eval.yaml`:

```bash
uv run python main.py eval --config configs/eval.yaml
```

Re-score existing episode logs without re-running them:

```bash
uv run python main.py eval --score-only ./logs/eval/run_YYYYMMDD_HHMMSS
```

Retry failed episodes and merge results back:

```bash
uv run python main.py eval --retry ./logs/eval/run_YYYYMMDD_HHMMSS
```

### Training

Run GRPO training with the config in `configs/train.yaml`:

```bash
uv run python main.py train --config configs/train.yaml
```

Training is handled by Tinker. Set `resume_run` in the config to continue from a checkpoint.

The transfer ablation (multi-item-only training, evaluated zero-shot on price tasks) is documented as commented blocks in `configs/train.yaml` and `configs/eval.yaml`.

## Configuration

**`configs/eval.yaml`** — key fields:

| Field | Description |
|---|---|
| `datasets` | Which datasets to evaluate (`casino`, `dnd`, `amazon`, `craigslist`, `ji`) |
| `num_episodes` | Episodes per dataset per matchup |
| `max_turns` | Max negotiation turns per episode |
| `max_concurrent` | Parallel episode workers |
| `matchups` | List of learner/opponent model pairs with API endpoints |

**`configs/train.yaml`** — key fields:

| Field | Description |
|---|---|
| `model_name` | Base model to train |
| `renderer_name` | Optional renderer override (e.g. `qwen3_5_disable_thinking` for hybrid thinking models) |
| `datasets` | Training datasets with sampling weights |
| `batch_size` / `group_size` | GRPO batch and group size |
| `batna_threshold` | BR threshold below which multi-item deals are penalized; default `0.0` (surplus reward), set to `0.4` for the threshold agent |
| `train_temperature` | Learner sampling temperature during training |
| `max_steps` | Training steps |
| `resume_run` | Tinker run ID to resume from |
| `wandb_project` | W&B project name |

## Datasets

| Dataset | Type | Split |
|---|---|---|
| [CaSiNo](https://convokit.cornell.edu/documentation/casino-corpus.html) | Multi-item (integrative) | train / eval (standard) |
| [DnD](https://github.com/facebookresearch/end-to-end-negotiator) | Multi-item (integrative) | train / eval (standard) |
| [Amazon Price History](https://github.com/TianXiaSJTU/AmazonPriceHistory) | Buyer/seller (distributive) | train / eval (product-disjoint, `scripts/split_ahp.py`) |
| [Craigslist Bargains](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains) | Buyer/seller (distributive) | train / eval (standard) |
| [Job Interview](https://github.com/gucci-j/negotiation-breakdown-detection) | Hybrid (held-out) | eval only |

All datasets are pre-processed and included in `data/`. AHP has no canonical split, so `scripts/split_ahp.py` creates an 80/20 product-level split (stratified by category) in `data/ahp/train/` and `data/ahp/test/`.
