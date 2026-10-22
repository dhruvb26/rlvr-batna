# rlvr-batna

BATNA-aware reinforcement learning from verifiable rewards for negotiation. The
code trains language models with GRPO across four negotiation datasets and
evaluates generalization on a fifth held-out dataset. Reward conditions compare
surplus maximization with a minimum deal-quality threshold.

See [`PLAN.md`](./PLAN.md) for the historical experimental plan and results.
The original experiments used `Qwen3-30B-A3B-Instruct-2507`; revision training
supports `Qwen/Qwen3.6-35B-A3B` with the non-thinking renderer
`qwen3_5_disable_thinking`.

## Papers

- **`emnlp-submission/`** — ACL/ARR short paper and submission history.
- **`ijpr-submission/`** — IJPR journal version.

Both build with `make` (requires a LaTeX toolchain); `make clean` removes build artifacts.

## Stack and setup

- **[Tinker](https://tinker.thinkingmachines.dev)** ([docs](https://tinker-docs.thinkingmachines.ai/)) — GRPO training and checkpoint inference.
- **[Weights & Biases](https://wandb.ai)** — experiment tracking.
- **[OpenRouter](https://openrouter.ai)** — inference through OpenAI-compatible endpoints. Direct OpenAI endpoints are also supported.

Requires Python 3.13+ and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/dhruvb26/rlvr-batna.git
cd rlvr-batna
uv sync --locked
cp .env.example .env
```

The project requires Tinker SDK 0.30.4 or newer; `uv.lock` pins 0.30.4.
Fill in the environment variables for the providers you use:

```env
TINKER_API_KEY=...
WANDB_API_KEY=...
OPENROUTER_API_KEY=...
OPENAI_API_KEY=...  # for direct OpenAI inference
HF_TOKEN=...        # optional, for gated Hugging Face models
```

`main.py` loads `.env`. Training outputs go under `logs/negotiation/`; evaluation
output locations are configurable. Keep local configs and working documents in
repository-root `files/`. Both `logs/` and `files/`, plus `.env` and `.venv`, are
ignored by Git.

## Revision implementation and purpose

| Change | Why it is needed |
|---|---|
| Shared response validation and explicit allocation ownership | Prevent malformed actions, fractional item quantities, and stale offers from becoming scored deals; make each participant's share unambiguous. |
| DnD loader correction and repaired Craigslist splits | Recover omitted DnD scenarios, avoid weighting duplicate DnD contexts, and remove checked Craigslist overlap across splits. |
| Frozen evaluation manifests and shared-panel scoring | Compare models on the same scenarios, roles, personas, and first movers, including when run sizes differ. |
| Selective/uniform reward controls and training seeds | Compare reward conditions with reproducible local scenario schedules and group roles. |
| Regulation controls and intervention evidence | Measure the effect of the opponent safeguard and retain the original and replacement responses for analysis. |
| Run manifests, checkpointed execution, and session tracking | Detect incompatible resumes and connect local logs, Tinker checkpoints, and W&B histories. |

Two consolidated recipes cover the completed experiment matrix:
[`configs/train.yaml`](configs/train.yaml) and
[`configs/eval.yaml`](configs/eval.yaml). Each has one active run and documented,
commented alternatives. Replace the relevant active fields when switching runs;
these files do not launch sweeps or interpret commented profiles automatically.
This avoids maintaining separate files for every condition, seed, and opponent.

The original per-run configs are preserved locally under the Git-ignored
`files/config-archive/2026-10-22/`, with hashes in `archive-index.json`.
Checkpoint IDs, provider routes, token caps, and historical protocol differences
are documented in the consolidated files. Historical training resumes still
require an identical saved manifest; a recipe alone cannot restore old code.

## Training

`configs/train.yaml` defaults to fresh Surplus training, seed 42, with equal
weights across CaSiNo, DnD, Amazon, and repaired Craigslist data: 80 updates,
64 groups per update, and 8 trajectories per group (40,960 episodes).
Commented replacements cover Surplus, Selective thresholds 0.2/0.4/0.6, Uniform
threshold 0.4, and the two smoke recipes. Each main condition uses seeds 42/43/44.
The multi-item-only ablation is explicitly marked planned, with no trained
checkpoint claimed.

Use the recipe directly or copy it into `files/` for local experiments. Set
`tinker_project_id` to your project UUID; the checked-in recipe records the
original project. When omitted, the resolved `TINKER_PROJECT_ID` environment
setting or SDK default is used. Optional `run_label` prefixes the readable name.

Validate configuration and data offline before training:

```bash
uv run --locked python main.py train --config configs/train.yaml --dry-run
```

To train, omit `--dry-run`. Override `--seed 43` or `--seed 44` for fresh independent
replicates with both resume fields null. The seed controls weighted resampling,
scenario ordering, and learner roles shared within each GRPO group. Remote model
initialization and both participants' generations remain stochastic.

| Condition | `threshold_scope` | `batna_threshold` |
|---|---|---:|
| Surplus | `selective` (inactive at zero) | 0.0 |
| Selective Threshold | `selective` | 0.4 |
| Uniform Threshold | `uniform` | 0.4 |

Selective applies the threshold to CaSiNo/DnD; Uniform also applies it to
Amazon/Craigslist. Commented Selective variants also cover thresholds 0.2 and 0.6.
JI is rejected as a training dataset. Equal dataset weights give equal expected task exposure.

### Pause, inspect, and resume

Keep the full budget in YAML and limit only the current launch:

```bash
uv run --locked python main.py train --config configs/train.yaml --run-steps 5
uv run --locked python main.py train --config configs/train.yaml --seed 42 --resume-run RUN_DIRECTORY_NAME
```

The first command trains five updates and saves model and optimizer state. The
second continues the same run to its declared budget; adding `--run-steps 5`
limits a resume to five additional updates. `--dry-run` also works with these
options. `resume_run` is a local directory name under `logs/negotiation/`, not a
Tinker checkpoint URI. `load_checkpoint_path` is a separate initialization option;
use `--resume-run` for an optimizer-preserving continuation.

`training_manifest.json` records the full configuration, training-pool hashes,
source hashes, dependency versions, and seed limitations. Resume requires an
identical manifest. `executions.jsonl` records each launch's boundaries and
status. A checkpoint called `final` can mark a paused launch: check its `batch`
number in `checkpoints.jsonl`. Use `sampler_path` for evaluation and `state_path`
for training-state restoration.

CLI training records Tinker session IDs, roles, metadata, and console links in
`tinker_sessions.jsonl`. W&B identity is saved in `wandb_run.json` and reused on
resume; ambiguous identity or history ahead of the saved checkpoint is rejected.
`src/training_audit.py` supplies optional diagnostic assertions on learner loss
masking, advantages, and token/log-probability alignment; ordinary training does
not enable them.

## Evaluation

Evaluation accepts YAML or TOML. `configs/eval.yaml` defaults to one matchup:
Surplus seed 43 at step 80 against direct OpenAI GPT-5.4, regulation enabled,
on all 517 main-panel episodes. Its commented alternatives record all 15 trained
learners, the untrained baseline, nine opponent models (including the historical
GPT-5.4 provider route), and regulation sensitivity panels. The run matrix shows
which combinations were completed. Switch the documented blocks and update
`storage_dir`, or copy the recipe into `files/` for local experiments.

For example, this dataset entry runs Amazon's main panel and also scores its
nested baseline panel:

```yaml
datasets:
  - name: amazon
    data_path: data/ahp/test
    manifest_path: data/evaluation_manifests/revision/main/amazon.json
    comparison_manifest_path: data/evaluation_manifests/revision/baseline/amazon.json
max_turns: 12
temperature: 0.7
top_p: 1.0
seed: 42
economic_regulation: true
```

Keep one desired `matchups` block active. Agent blocks specify
`label`, `model`, `max_tokens`, and either API `base_url`/`api_key_env` or
`provider: tinker` with an optional renderer. Tinker checkpoint access can use
`tinker_project_id` at the top level or on an agent, falling back to
`TINKER_PROJECT_ID` or the SDK default project.

```bash
uv run --locked python main.py eval --config configs/eval.yaml
uv run --locked python main.py eval --score-only logs/eval/RUN_DIRECTORY_NAME
uv run --locked python main.py eval --config configs/eval.yaml --retry logs/eval/RUN_DIRECTORY_NAME
```

The first command launches inference; score-only aggregates saved episode
metrics without model calls. Retry reruns execution failures, preserving their
scenario, persona, roles, first mover, regulation setting, and saved top-level
generation settings. Use the original agent configuration. Invalid model
responses count as model behavior rather than execution failures.

### Main and baseline panels

The committed schedules are indexed in
[`data/evaluation_manifests/revision/index.json`](data/evaluation_manifests/revision/index.json).
They contain inputs and assignments, not model responses or training data.

| Dataset | Main | Baseline |
|---|---:|---:|
| Amazon | 93 | 50 |
| CaSiNo | 124 | 50 |
| DnD | 100 | 50 |
| Craigslist | 100 | 50 |
| JI | 100 | 50 |
| **Total per matchup** | **517** | **250** |

Each baseline is a subset of main with identical roles, persona, and first mover.
Use main for the larger evaluation, or baseline for a smaller run. Setting
`comparison_manifest_path` produces `shared_metrics.json`, `shared_summary.json`,
and `shared_comparison.csv` from the shared scenarios without extra inference.
Missing required episodes mark shared results incomplete rather than silently
changing the comparison subset. Full-panel results remain in `metrics.json`,
`summary.json`, and `comparison.csv`.

Saved manifests validate loaded-pool hashes and override scenario-selection
settings. Without a saved manifest, the harness builds one schedule per dataset,
samples unique scenarios without replacement, and reuses assignments across all
matchups. `num_episodes` requests unique scenarios, capped at the available pool;
`repeats` explicitly schedules trials per scenario (default 1). Role and
first-mover balance is across a panel, not a full factorial per scenario.

Join panels using `scenario_id` and `repeat`; `episode_id` is panel-local. Compare
scores within a matching dataset, comparison-manifest hash, and
`comparison_group`, which captures opponent configuration, protocol, regulation,
turn limit, temperature, and top-p. Model responses remain stochastic.

Compatible saved runs can also be scored on a shared panel offline:

```bash
uv run --locked python -m src.eval.shared \
  --log-dirs logs/eval/RUN_A logs/eval/RUN_B \
  --manifest-dir data/evaluation_manifests/revision/baseline \
  --output-dir logs/eval/shared-comparison
```

## Response protocol and regulation

Training and live evaluation share the `explicit-allocation-v3` protocol:
exactly one nonempty `Thought:`, `Talk:`, and `Action:` section, in that order.
Actions must be fully parsed and domain-valid; fractional indivisible-item
quantities and trailing text are rejected. Outgoing item proposals specify the
speaker's share. Incoming observations preserve speech and label `You receive`
and `Partner receives`; prices/job offers display `Proposed terms`. Private
thoughts are not sent to the partner.

Malformed turns and explicit rejections clear the pending offer. An acceptance
requires a valid active partner proposal. Invalid learner responses terminate
training episodes with reward -1; invalid opponent responses consume a turn.
Evaluation consumes invalid turns and continues. Raw failures and their reasons
are retained. Training's `max_rounds` includes the opponent reply to the final
learner proposal. Evaluation ends on agreement, walk-away, or `max_turns`;
repeated offers are diagnostic events and do not terminate it early.

`economic_regulation` controls the opponent's negative-utility guard in both
training and evaluation (default true). It replaces valid opponent proposals or
acceptances yielding negative opponent utility with rejection. It does not guard
the learner or relax action validation. Logs retain original and replacement
responses; evaluation events also retain structured original/final turns and
conversation indices. These events show interventions, not counterfactual final
outcomes. Training terminal logs retain scenario, role, conversation, and reward
settings in the Cookbook rollout exports.

Live evaluation records protocol `explicit-allocation-v3-turn-limit-v1`. Saved
historical results can still be aggregated, but retries require the current
protocol. Changing regulation or protocol requires a fresh evaluation; rescoring
cannot reconstruct a different conversation. Historical training exposure and
v2/v3 results remain distinct from fresh revision runs.

## Datasets and split repairs

| Dataset | Type | Revision use |
|---|---|---|
| [CaSiNo](https://convokit.cornell.edu/documentation/casino-corpus.html) | Multi-item | Train / test |
| [DnD](https://github.com/facebookresearch/end-to-end-negotiator) | Multi-item | Train / test; distinct allocation contexts |
| [Amazon Price History](https://github.com/TianXiaSJTU/AmazonPriceHistory) | Price | Train / test; product-disjoint AHP split |
| [Craigslist Bargains](https://huggingface.co/datasets/stanfordnlp/craigslist_bargains) | Price | Repaired train / validation / test |
| [Job Interview](https://github.com/gucci-j/negotiation-breakdown-detection) | Hybrid | Evaluation only |

AHP's existing category-stratified 80/20 product split reproduces with
`scripts/split_ahp.py`: 422 usable train and 93 test scenarios. It needs no rewrite.

DnD now reads every row and deduplicates by item counts and both agents'
valuations, including swapped seats: 1,884 unique train and 200 test contexts.
The old alternate-row assumption omitted distinct scenarios. This changes DnD
sampling weights and invalidates manifests made against the old loaded pool.

`src/craigslist_data.py` shares scenario parsing and content-based identities
between the loader and data-preparation tooling. Revision splits under
`data/craigslist/revision/` remove 57 raw training rows and four validation rows
from overlapping listing/context groups, prioritizing test, then validation,
then training. Seller-cost and filtering rules are retained. Scenario UUIDs are
excluded from repair grouping because unrelated source listings reuse them.

Original Craigslist files remain available, and the revision test file is
byte-for-byte unchanged. `repair-manifest.json` records exclusions and hashes.
Repaired usable counts are 3,973 train, 462 validation, and 637 test. Configure
revision training with `data/craigslist/revision/train.json`; evaluation's default
Craigslist path already selects the revision test copy. Fixed data and manifests
permit comparisons on the checked split identities; they do not undo exposure
in old checkpoints or rule out every possible near-duplicate.


## IJPR revision figures

The [October 4 revision figure snapshot](ijpr-submission/figures/revision-2026-10-04/README.md)
contains 12 PNG/vector-PDF pairs and saved numeric inputs for the training and
`bar_*` plots. These preserve the revision results for manuscript preparation;
the original manuscript still references the earlier figures. The figure README
records seed aggregation, the retained Craigslist outlier, historical protocol
and provider differences, and the snapshot's tau .4/.6 coverage. The later tau .2
experiments are not included in these plots.


## Offline data and figure tools

These tools support the committed split repairs, frozen evaluation panels, and
revision figures. Run the commands below from the repository root. They use
local files and make no model calls.

| Script | Purpose |
|---|---|
| `scripts/audit_splits.py` | Read-only source/context overlap checks, AHP split reproduction, and DnD loader coverage checks. `--strict` fails on blocking issues. |
| `scripts/repair_craigslist_splits.py` | Rebuild repaired Craigslist splits and the exclusion/hash manifest; preserve original sources and the test file's bytes. |
| `scripts/generate_eval_manifests.py` | Audit the splits, then regenerate the 517-episode main panel and nested 250-episode baseline panel with their index. |
| `scripts/gen_revision_handoff_figures.py` | Rebuild nine revision PNG/PDF pairs and numeric inputs from saved local results, without editing a handoff or manuscript. |

```bash
uv run --locked python -m scripts.audit_splits --strict --output files/split-audit.json
```

The following preparation commands rewrite derived files under `data/`. Use them
when intentionally rebuilding the frozen artifacts, rather than before each run:

```bash
uv run --locked python -m scripts.repair_craigslist_splits
uv run --locked python -m scripts.generate_eval_manifests --seed 42 --config configs/eval.yaml
```

Manifest generation reads only `persona_weights` from the config; the committed
panels use `none: 1`. It preserves baseline scenario/role/persona/first-mover
assignments from the main panels. Changing seed or persona weights changes the
frozen comparison and requires an intentional new evaluation panel.

To render the nine original-style revision plots into a local preview directory:

```bash
uv run --locked python -m scripts.gen_revision_handoff_figures --output files/revision-figure-preview
```

Figure generation requires the saved training inventory, evaluation aggregates,
and training metrics under `logs/`; these evidence inputs remain local and are
not included in this scripts commit. `--root`, `--inventory`, `--analysis`, and
`--output` allow another checkout or evidence location. Relative arguments resolve
under `--root`, which defaults to this checkout regardless of the working
directory. Historical source paths are rebased onto the current root; newly
written provenance uses repository-relative paths. Missing inputs produce an
explicit error. Importing the generator does not render figures. The three
separate overview/sensitivity plots in the snapshot are not generated by this
script; its original nine-plot scope and the saved snapshot remain unchanged.
