# rlvr-batna

BATNA-aware reinforcement learning from verifiable rewards for negotiation. Trains `Qwen3-30B-A3B` with GRPO across five negotiation datasets using a dataset-aware reward that penalizes deals below a quality threshold — encoding the negotiation principle of [BATNA](https://en.wikipedia.org/wiki/Best_alternative_to_a_negotiated_agreement) directly into the reward signal.

See [`PLAN.md`](./PLAN.md) for the full experimental plan, baseline results, and training outcomes.

## Stack

- **[Tinker](https://tinker.thinkingmachines.dev)** — GRPO training and serving trained model checkpoints
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
OPENAI_API_KEY=...  # optional
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
| `datasets` | Training datasets with sampling weights |
| `batch_size` / `group_size` | GRPO batch and group size |
| `batna_threshold` | BR threshold below which multi-item deals are penalized (default `0.4`) |
| `max_steps` | Training steps |
| `resume_run` | Tinker run ID to resume from |
| `wandb_project` | W&B project name |

## Datasets

| Dataset | Type | Split |
|---|---|---|
| [CaSiNo](https://convokit.cornell.edu/documentation/casino-corpus.html) | Multi-item (integrative) | train / eval |
| [DnD](https://github.com/facebookresearch/end-to-end-negotiator) | Multi-item (integrative) | train / eval |
| [Amazon Price History](https://github.com/TianXiaSJTU/AmazonPriceHistory) | Buyer/seller (distributive) | train / eval |
| [Craigslist Bargains](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains) | Buyer/seller (distributive) | train / eval |
| [Job Interview](https://github.com/gucci-j/negotiation-breakdown-detection) | Hybrid (held-out) | eval only |

All datasets are pre-processed and included in `data/`.
