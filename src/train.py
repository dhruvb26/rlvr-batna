import asyncio
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

import chz
import tinker
from tinker import ModelInput
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.completers import (
    MessageCompleter,
    StopCondition,
    TinkerMessageCompleter,
)
from tinker_cookbook.renderers import Message, Renderer, get_renderer, get_text_content
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .envs import NegotiateEnv as EvalEnv
from .envs import get_env

OUTCOMES = ("deal", "walk_away", "max_turns", "format_violation")


def _outcome_metrics(
    outcome: str, dataset: str = "", **extra: float
) -> dict[str, float]:
    """Build metrics dict for episode outcome logging.

    Args:
        outcome: One of 'deal', 'walk_away', 'max_turns', 'format_violation'.
        dataset: Dataset name for per-dataset metric tags.
        **extra: Additional float metrics to include.

    Returns:
        Dict of metric name to float value.
    """
    m = {f"outcome_{o}": float(o == outcome) for o in OUTCOMES}
    if dataset:
        m["dataset"] = hash(dataset) % 1000
        m[f"outcome_deal/{dataset}"] = float(outcome == "deal")
        m[f"outcome_walk_away/{dataset}"] = float(outcome == "walk_away")
    return m | extra


def extract_section(text: str, label: str) -> str | None:
    """Extract a labeled section from structured model output.

    Args:
        text: Full model output text.
        label: Section label (e.g. 'Thought', 'Talk', 'Action').

    Returns:
        Section content string or None if not found.
    """
    m = re.search(
        rf"^{label}:\s*(.+?)(?=^(?:Thought|Talk|Action):|\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def strip_thought(text: str) -> str:
    """Remove the Thought: section from model output before forwarding."""
    return re.sub(
        r"^Thought:.*?(?=^(?:Talk|Action):|\Z)",
        "",
        text,
        flags=re.MULTILINE | re.DOTALL,
    ).strip()


@dataclass(frozen=True)
class Scenario:
    """A single negotiation scenario with its environment and metadata."""

    env: EvalEnv
    data: dict
    agent_ids: list[str]
    dataset_name: str = ""


def load_all_scenarios(
    datasets: list[dict], target_size: int | None = None
) -> list[Scenario]:
    """Load scenarios from multiple datasets with optional weighted resampling.

    When datasets have a 'weight' key, scenarios are resampled so each dataset
    contributes proportionally to its weight. Without weights (or all equal),
    uses the natural distribution.

    Args:
        datasets: List of {name, data_path, weight?} dicts.
        target_size: Total number of scenarios to produce. If None, uses the
            sum of all raw scenarios (natural size).

    Returns:
        Flat list of Scenario objects, resampled according to weights.
    """
    per_dataset: dict[str, list[Scenario]] = {}
    weights: dict[str, float] = {}

    for ds in datasets:
        name = ds["name"]
        data_path = ds["data_path"]
        weight = ds.get("weight", 1.0)
        env = get_env(name)
        raw_scenarios = env.load_scenarios(data_path)
        scenarios = []
        for s in raw_scenarios:
            agent_ids = s.get("agent_ids", s.get("agents", {}).keys())
            scenarios.append(
                Scenario(env=env, data=s, agent_ids=list(agent_ids), dataset_name=name)
            )
        per_dataset[name] = scenarios
        weights[name] = weight

    total_raw = sum(len(s) for s in per_dataset.values())
    if target_size is None:
        target_size = total_raw

    total_weight = sum(weights.values())
    all_scenarios: list[Scenario] = []
    for name, scenarios in per_dataset.items():
        n_target = round(target_size * weights[name] / total_weight)
        if n_target <= 0 or not scenarios:
            continue
        resampled = random.choices(scenarios, k=n_target)
        all_scenarios.extend(resampled)

    return all_scenarios


class NegotiationEnv(Env):
    """One negotiation episode as a Tinker RL environment.

    Reward:
      deal with bargained_ratio >= threshold -> bargained_ratio
      deal with bargained_ratio <  threshold -> -0.5 (BATNA penalty)
      no deal / walk away / max turns        -> 0.0
      format violation                       -> -1.0
    """

    def __init__(
        self,
        opponent: MessageCompleter,
        scenario: Scenario,
        renderer: Renderer,
        max_rounds: int = 6,
        batna_threshold: float = 0.0,
    ):
        """Initialize a negotiation environment for one episode.

        Args:
            opponent: Message completer for the opponent model.
            scenario: Scenario data with env, agent_ids, dataset_name.
            renderer: Renderer for building generation prompts.
            max_rounds: Maximum dialogue rounds before timeout.
            batna_threshold: Minimum bargained_ratio threshold for reward.
        """
        self.opponent = opponent
        self.scenario = scenario
        self.env = scenario.env
        self.renderer = renderer
        self.max_rounds = max_rounds
        self.batna_threshold = batna_threshold

        ids = scenario.agent_ids
        fixed = self.env.fixed_learner_role()
        if fixed and fixed in ids:
            self.learner_id = fixed
            self.opponent_id = [a for a in ids if a != fixed][0]
        else:
            self.learner_id, self.opponent_id = (
                (ids[0], ids[1]) if random.random() < 0.5 else (ids[1], ids[0])
            )

        self.learner_sys: Message = {
            "role": "system",
            "content": self.env.build_system_prompt(scenario.data, self.learner_id),
        }
        self.opponent_sys: Message = {
            "role": "system",
            "content": self.env.build_system_prompt(scenario.data, self.opponent_id),
        }

        self.learner_turns: list[Message] = []
        self.opponent_turns: list[Message] = []
        self.round = 0
        self.last_submit_deal: dict | None = None
        self.last_submit_by: str | None = None
        self.learner_first_offer: dict | None = None
        self.turns_taken = 0

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _deal_result(self, learner_alloc: dict) -> StepResult:
        """Compute reward and metrics when a deal is reached.

        Args:
            learner_alloc: The learner's allocation in the deal.

        Returns:
            StepResult with reward and episode metrics.
        """
        ds = self.scenario.dataset_name
        points = self.env.compute_points(
            learner_alloc, self.scenario.data, self.learner_id
        )
        max_pts = self.env.max_points(self.scenario.data, self.learner_id)
        bargained_ratio = points / max_pts if max_pts > 0 else 0.0

        extra: dict[str, float] = {
            "learner_points": float(points),
            "bargained_ratio": bargained_ratio,
            f"bargained_ratio/{ds}": bargained_ratio,
            "episode_length": float(self.turns_taken),
        }

        if ds in ("amazon", "craigslist") and self.learner_first_offer is not None:
            budget = self.scenario.data.get("buyer_budget", 0)
            first_price = self.learner_first_offer.get("price", 0)
            if budget > 0:
                extra["first_bid_ratio"] = first_price / budget
                extra[f"first_bid_ratio/{ds}"] = extra["first_bid_ratio"]
            deal_price = learner_alloc.get("price", 0)
            extra["price_overshoot"] = float(deal_price > budget) if budget > 0 else 0.0

        if ds in ("casino", "dnd"):
            opp_alloc = self.env.invert_alloc(learner_alloc, self.scenario.data)
            opp_points = self.env.compute_points(
                opp_alloc, self.scenario.data, self.opponent_id
            )
            max_joint = self.env.max_joint_points(self.scenario.data)
            if max_joint > 0:
                extra["joint_surplus"] = (points + opp_points) / max_joint
                extra[f"joint_surplus/{ds}"] = extra["joint_surplus"]
            pareto = self.env.is_pareto_efficient(
                learner_alloc, self.scenario.data, self.learner_id
            )
            extra["pareto_efficient"] = float(pareto)
            extra[f"pareto_efficient/{ds}"] = float(pareto)

            if self.learner_first_offer is not None and max_pts > 0:
                first_pts = self.env.compute_points(
                    self.learner_first_offer, self.scenario.data, self.learner_id
                )
                extra["first_offer_greed"] = first_pts / max_pts
                extra[f"first_offer_greed/{ds}"] = extra["first_offer_greed"]

        threshold = self.batna_threshold if ds in ("casino", "dnd", "ji") else 0.0
        if threshold > 0 and bargained_ratio < threshold:
            reward = -0.5
        else:
            reward = max(-1.0, min(1.0, bargained_ratio))

        return StepResult(
            next_observation=self.renderer.build_generation_prompt(
                self._learner_convo()
            ),
            next_stop_condition=self.stop_condition,
            episode_done=True,
            reward=reward,
            metrics=_outcome_metrics("deal", ds, **extra),
        )

    def _learner_convo(self) -> list[Message]:
        if not self.learner_turns:
            opening = self.env.opening_prompt(self.learner_id)
            return [self.learner_sys, {"role": "user", "content": opening}]
        return [self.learner_sys] + self.learner_turns

    def _opponent_convo(self) -> list[Message]:
        return [self.opponent_sys] + self.opponent_turns

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        """Return the initial prompt for the learner to generate from."""
        return (
            self.renderer.build_generation_prompt(self._learner_convo()),
            self.stop_condition,
        )

    async def step(
        self, action: Action, *, extra: ActionExtra | None = None
    ) -> StepResult:
        """Process one round: learner spoke, now opponent responds.

        Args:
            action: The learner's generated action (tokenized response).
            extra: Optional extra action metadata.

        Returns:
            StepResult with reward, done flag, and next observation.
        """
        (learner_msg, _) = self.renderer.parse_response(action)
        learner_text = get_text_content(learner_msg)
        self.turns_taken += 1

        if not (
            extract_section(learner_text, "Thought")
            and extract_section(learner_text, "Talk")
            and extract_section(learner_text, "Action")
        ):
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=-1.0,
                metrics=_outcome_metrics(
                    "format_violation",
                    self.scenario.dataset_name,
                    episode_length=float(self.turns_taken),
                ),
            )

        self.learner_turns.append({"role": "assistant", "content": learner_text})
        visible = self.env.flip_deal(strip_thought(learner_text), self.scenario.data)
        self.opponent_turns.append({"role": "user", "content": visible})

        learner_action = extract_section(learner_text, "Action") or ""
        learner_action_upper = learner_action.strip().upper()

        parsed = self.env.parse_deal(learner_action)
        if parsed is not None:
            if self.learner_first_offer is None:
                self.learner_first_offer = parsed
            self.last_submit_deal = parsed
            self.last_submit_by = "learner"

        if (
            "[ACCEPT_DEAL]" in learner_action_upper
            and self.last_submit_deal is not None
            and self.last_submit_by == "opponent"
        ):
            learner_alloc = self.env.invert_alloc(
                self.last_submit_deal, self.scenario.data
            )
            return self._deal_result(learner_alloc)

        if "[WALK_AWAY]" in learner_action_upper:
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
                metrics=_outcome_metrics(
                    "walk_away",
                    self.scenario.dataset_name,
                    episode_length=float(self.turns_taken),
                ),
            )

        self.round += 1
        if self.round >= self.max_rounds:
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
                metrics=_outcome_metrics(
                    "max_turns",
                    self.scenario.dataset_name,
                    episode_length=float(self.turns_taken),
                ),
            )

        opponent_msg = await self.opponent(self._opponent_convo())
        opponent_text = get_text_content(opponent_msg)

        # Regulate: intercept any opponent action that violates cost constraint.
        opp_action_pre = extract_section(opponent_text, "Action") or ""
        opp_action_pre_upper = opp_action_pre.strip().upper()
        if (
            "[ACCEPT_DEAL]" in opp_action_pre_upper
            and self.last_submit_deal is not None
            and self.last_submit_by == "learner"
        ):
            opp_alloc = self.env.invert_alloc(self.last_submit_deal, self.scenario.data)
            opp_pts = self.env.compute_points(
                opp_alloc, self.scenario.data, self.opponent_id
            )
            if opp_pts < 0:
                opponent_text = (
                    "Thought: That price is too low for me.\n"
                    "Talk: I can't accept that offer.\n"
                    "Action: [REJECT]"
                )
        elif "[SUBMIT_DEAL]" in opp_action_pre_upper:
            opp_deal = self.env.parse_deal(opp_action_pre)
            if opp_deal is not None:
                opp_alloc = self.env.invert_alloc(opp_deal, self.scenario.data)
                opp_pts = self.env.compute_points(
                    opp_alloc, self.scenario.data, self.opponent_id
                )
                if opp_pts < 0:
                    opponent_text = (
                        "Thought: I need to protect my margins.\n"
                        "Talk: I can't go that low.\n"
                        "Action: [REJECT]"
                    )

        self.opponent_turns.append({"role": "assistant", "content": opponent_text})
        visible_opp = self.env.flip_deal(
            strip_thought(opponent_text), self.scenario.data
        )
        self.learner_turns.append({"role": "user", "content": visible_opp})

        opp_action = extract_section(opponent_text, "Action") or ""
        opp_action_upper = opp_action.strip().upper()

        opp_parsed = self.env.parse_deal(opp_action)
        if opp_parsed is not None:
            self.last_submit_deal = opp_parsed
            self.last_submit_by = "opponent"

        if (
            "[ACCEPT_DEAL]" in opp_action_upper
            and self.last_submit_deal is not None
            and self.last_submit_by == "learner"
        ):
            return self._deal_result(self.last_submit_deal)

        if "[WALK_AWAY]" in opp_action_upper:
            return StepResult(
                next_observation=self.renderer.build_generation_prompt(
                    self._learner_convo()
                ),
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
                metrics=_outcome_metrics(
                    "walk_away",
                    self.scenario.dataset_name,
                    episode_length=float(self.turns_taken),
                ),
            )

        return StepResult(
            next_observation=self.renderer.build_generation_prompt(
                self._learner_convo()
            ),
            next_stop_condition=self.stop_condition,
            episode_done=False,
            reward=0.0,
        )


@dataclass(frozen=True)
class NegotiationGroupBuilder(EnvGroupBuilder):
    """Builds a group of identical NegotiationEnv instances for GRPO."""

    opponent: MessageCompleter
    scenario: Scenario
    renderer: Renderer
    num_envs: int
    max_rounds: int = 6
    batna_threshold: float = 0.0

    async def make_envs(self) -> Sequence[Env]:
        return [
            NegotiationEnv(
                self.opponent,
                self.scenario,
                self.renderer,
                self.max_rounds,
                self.batna_threshold,
            )
            for _ in range(self.num_envs)
        ]


@dataclass(frozen=True)
class NegotiationDataset(RLDataset):
    """Dataset that yields batches of negotiation scenario groups."""

    opponent: MessageCompleter
    scenarios: Sequence[Scenario]
    renderer: Renderer
    batch_size: int
    group_size: int
    max_rounds: int = 6
    batna_threshold: float = 0.0

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        start = (index * self.batch_size) % len(self.scenarios)
        batch_scenarios = [
            self.scenarios[(start + i) % len(self.scenarios)]
            for i in range(self.batch_size)
        ]
        return [
            NegotiationGroupBuilder(
                opponent=self.opponent,
                scenario=s,
                renderer=self.renderer,
                num_envs=self.group_size,
                max_rounds=self.max_rounds,
                batna_threshold=self.batna_threshold,
            )
            for s in batch_scenarios
        ]

    def __len__(self) -> int:
        return len(self.scenarios) // self.batch_size


@chz.chz
class NegotiationDatasetBuilder(RLDatasetBuilder):
    """Builds the negotiation RL dataset with opponent completer."""

    datasets: str = "casino:data/casino/ca.train.csv"
    batch_size: int = 2
    group_size: int = 2
    max_rounds: int = 6
    model_name_for_tokenizer: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    num_epochs: int = 1
    opponent_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    opponent_temperature: float = 0.7
    eval_group_size: int = 4
    batna_threshold: float = 0.0

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        service_client = tinker.ServiceClient()

        opp_renderer_name = model_info.get_recommended_renderer_name(
            self.opponent_model
        )
        opp_tokenizer = get_tokenizer(self.opponent_model)
        opp_renderer = get_renderer(opp_renderer_name, opp_tokenizer)
        opp_client = service_client.create_sampling_client(
            base_model=self.opponent_model
        )
        opponent = TinkerMessageCompleter(
            sampling_client=opp_client,
            renderer=opp_renderer,
            max_tokens=300,
            temperature=self.opponent_temperature,
        )

        learner_renderer_name = model_info.get_recommended_renderer_name(
            self.model_name_for_tokenizer
        )
        learner_tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        learner_renderer = get_renderer(learner_renderer_name, learner_tokenizer)

        ds_specs = _parse_datasets_str(self.datasets)
        scenarios = load_all_scenarios(ds_specs)
        random.shuffle(scenarios)
        scenarios = scenarios * self.num_epochs

        dataset = NegotiationDataset(
            opponent=opponent,
            scenarios=scenarios,
            renderer=learner_renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            max_rounds=self.max_rounds,
            batna_threshold=self.batna_threshold,
        )
        return dataset, None


def _parse_datasets_str(datasets_str: str) -> list[dict]:
    """Parse 'name:path:weight,name:path:weight,...' into list of dicts.

    Args:
        datasets_str: Comma-separated dataset specifications.
            Format: 'name:path[:weight]' where weight is optional (default 1.0).

    Returns:
        List of {name, data_path, weight} dicts.
    """
    specs = []
    for entry in datasets_str.split(","):
        entry = entry.strip()
        parts = entry.split(":")
        if len(parts) >= 3:
            name, path, weight = parts[0].strip(), parts[1].strip(), float(parts[2])
            specs.append({"name": name, "data_path": path, "weight": weight})
        elif len(parts) == 2:
            name, path = parts[0].strip(), parts[1].strip()
            specs.append({"name": name, "data_path": path, "weight": 1.0})
        else:
            specs.append(
                {"name": entry, "data_path": _default_path(entry), "weight": 1.0}
            )
    return specs


DEFAULT_TRAIN_PATHS: dict[str, str] = {
    "casino": "data/casino/ca.train.csv",
    "dnd": "data/dnd/dnd.train.csv",
    "amazon": "data/ahp",
    "craigslist": "data/craigslist/train.json",
    "ji": "data/ji/ji.test.json",
}


def _default_path(name: str) -> str:
    """Get the default training data path for a dataset name.

    Args:
        name: Dataset name.

    Returns:
        Default file path string.

    Raises:
        ValueError: If no default path is known.
    """
    if name not in DEFAULT_TRAIN_PATHS:
        raise ValueError(f"No default train path for {name!r}")
    return DEFAULT_TRAIN_PATHS[name]


@chz.chz
class CLIConfig:
    """CLI configuration for training runs."""

    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    datasets: str = "casino:data/casino/ca.train.csv"
    group_size: int = 2
    batch_size: int = 2
    num_epochs: int = 2
    learning_rate: float = 3e-5
    lora_rank: int = 32
    max_tokens: int = 300
    max_rounds: int = 6

    opponent_temperature: float = 0.7
    kl_penalty_coef: float = 0.0
    batna_threshold: float = 0.0
    max_steps: int | None = None
    save_every: int = 10
    eval_every: int = 0
    resume_run: str | None = None
    load_checkpoint_path: str | None = None
    wandb_project: str | None = "agent-rlvr"
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


def build_config(cli: CLIConfig) -> tuple[train.Config, str]:
    """Build a Tinker training Config from CLI parameters.

    Args:
        cli: Parsed CLI configuration.

    Returns:
        Tuple of (train.Config, run_name).
    """
    from datetime import datetime

    renderer_name = model_info.get_recommended_renderer_name(cli.model_name)

    if cli.resume_run:
        run_name = cli.resume_run
        log_path = f"logs/negotiation/{run_name}"
    else:
        run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        log_path = f"logs/negotiation/{run_name}"

    dataset_builder = NegotiationDatasetBuilder(
        datasets=cli.datasets,
        batch_size=cli.batch_size,
        group_size=cli.group_size,
        max_rounds=cli.max_rounds,
        model_name_for_tokenizer=cli.model_name,
        num_epochs=cli.num_epochs,
        opponent_model=cli.model_name,
        opponent_temperature=cli.opponent_temperature,
        batna_threshold=cli.batna_threshold,
    )

    config = train.Config(
        model_name=cli.model_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        kl_penalty_coef=cli.kl_penalty_coef,
        max_steps=cli.max_steps,
        save_every=cli.save_every,
        eval_every=cli.eval_every,
        load_checkpoint_path=cli.load_checkpoint_path,
        wandb_project=cli.wandb_project,
        wandb_name=run_name if cli.wandb_project else None,
    )

    return config, run_name


def main(config_path: str = "configs/train.yaml"):
    """CLI entry point for negotiation training.

    Args:
        config_path: Path to YAML training config.
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if "datasets" in cfg and isinstance(cfg["datasets"], list):
        parts = []
        for ds in cfg["datasets"]:
            if isinstance(ds, dict):
                name = ds["name"]
                path = ds.get("data_path", _default_path(name))
                weight = ds.get("weight", 1.0)
                parts.append(f"{name}:{path}:{weight}")
            else:
                parts.append(ds)
        cfg["datasets"] = ",".join(parts)

    cfg.pop("csv_path", None)
    cfg.pop("eval_csv_path", None)

    cli = CLIConfig(**{k: v for k, v in cfg.items() if hasattr(CLIConfig, k)})
    config, run_name = build_config(cli)
    if not cli.resume_run:
        cli_utils.check_log_dir(
            config.log_path, behavior_if_exists=cli.behavior_if_log_dir_exists
        )
    asyncio.run(train.main(config))
