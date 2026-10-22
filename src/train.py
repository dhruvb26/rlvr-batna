from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

import chz
from tinker import ModelInput
from tinker_cookbook import checkpoint_utils, cli_utils, model_info
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
    Logs,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .envs import NegotiateEnv as EvalEnv
from .envs import get_env
from .protocol import PROTOCOL_VERSION, parse_response, partner_observation
from .scenarios import digest, scenario_id, serialize_scenario
from .tinker_tracking import (
    create_service_client,
    record_tinker_sessions,
    track_tinker_sessions,
)

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
        m["dataset"] = int(digest(dataset)[:8], 16) % 1000
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
    datasets: list[dict], target_size: int | None = None, *, seed: int = 42
) -> list[Scenario]:
    """Load scenarios from multiple datasets with optional weighted resampling.

    Each dataset is resampled in proportion to its weight (default 1.0).
    Equal weights give equal task exposure, not the natural row distribution.

    Args:
        datasets: List of {name, data_path, weight?} dicts.
        target_size: Total number of scenarios to produce. If None, uses the
            sum of all raw scenarios (natural size).

    Returns:
        Flat list of Scenario objects, resampled according to weights.
    """
    rng = random.Random(seed)
    per_dataset: dict[str, list[Scenario]] = {}
    weights: dict[str, float] = {}

    for ds in datasets:
        name = ds["name"]
        data_path = ds["data_path"]
        weight = ds.get("weight", 1.0)
        if name == "ji":
            raise ValueError("JI is held out and cannot be included in revision training")
        if name in per_dataset or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Datasets must be unique and have finite positive weights")
        env = get_env(name)
        raw_scenarios = env.load_scenarios(data_path)
        if not raw_scenarios:
            raise ValueError(f"No usable training scenarios for {name}")
        scenarios = []
        for s in raw_scenarios:
            agent_ids = s.get("agent_ids", s.get("agents", {}).keys())
            scenarios.append(
                Scenario(env=env, data=s, agent_ids=list(agent_ids), dataset_name=name)
            )
        per_dataset[name] = scenarios
        weights[name] = weight

    if not per_dataset:
        raise ValueError("At least one training dataset is required")
    total_raw = sum(len(s) for s in per_dataset.values())
    if target_size is None:
        target_size = total_raw

    total_weight = sum(weights.values())
    all_scenarios: list[Scenario] = []
    for name, scenarios in per_dataset.items():
        n_target = round(target_size * weights[name] / total_weight)
        if n_target <= 0 or not scenarios:
            continue
        resampled = rng.choices(scenarios, k=n_target)
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
        *,
        learner_id: str | None = None,
        seed: int = 42,
        threshold_scope: str = "selective",
        economic_regulation: bool = True,
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
        self.threshold_scope = threshold_scope
        self.economic_regulation = economic_regulation
        self.seed = seed
        self.regulation_interventions: list[dict] = []
        self.format_failures: list[dict] = []
        if threshold_scope not in {"selective", "uniform"}:
            raise ValueError("threshold_scope must be selective or uniform")
        if type(economic_regulation) is not bool or type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("Invalid regulation flag or round limit")
        if not math.isfinite(batna_threshold) or not 0 <= batna_threshold <= 1:
            raise ValueError("batna_threshold must be in [0, 1]")

        ids = scenario.agent_ids
        fixed = self.env.fixed_learner_role()
        if len(ids) != 2 or len(set(ids)) != 2:
            raise ValueError("Training requires two distinct agent IDs")
        self.learner_id = learner_id or fixed or random.Random(seed).choice(ids)
        if self.learner_id not in ids or (fixed and self.learner_id != fixed):
            raise ValueError("Invalid learner role")
        self.opponent_id = next(a for a in ids if a != self.learner_id)

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
        if not self.env.validate_deal(learner_alloc, self.scenario.data):
            raise ValueError("Invalid allocation reached reward calculation")
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
            if max_joint is not None and max_joint > 0:
                extra["joint_surplus"] = (points + opp_points) / max_joint
                extra[f"joint_surplus/{ds}"] = extra["joint_surplus"]
            pareto = self.env.is_pareto_efficient(
                learner_alloc, self.scenario.data, self.learner_id
            )
            if pareto is not None:
                extra["pareto_efficient"] = float(pareto)
                extra[f"pareto_efficient/{ds}"] = float(pareto)

            if self.learner_first_offer is not None and max_pts > 0:
                first_pts = self.env.compute_points(
                    self.learner_first_offer, self.scenario.data, self.learner_id
                )
                extra["first_offer_greed"] = first_pts / max_pts
                extra[f"first_offer_greed/{ds}"] = extra["first_offer_greed"]

        threshold = self.batna_threshold if (self.threshold_scope == "uniform" or ds in ("casino", "dnd")) else 0.0
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
            logs=self._logs(learner_alloc),
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

    def _logs(self, deal=None) -> Logs:
        """Retain full terminal evidence in Cookbook rollout JSON logs."""
        return {"negotiation": json.dumps({
            "scenario_id": scenario_id(self.scenario.dataset_name, self.scenario.data),
            "scenario": serialize_scenario(self.scenario.dataset_name, self.scenario.data),
            "dataset": self.scenario.dataset_name,
            "learner_agent_id": self.learner_id,
            "opponent_agent_id": self.opponent_id,
            "episode_seed": self.seed,
            "threshold_scope": self.threshold_scope,
            "batna_threshold": self.batna_threshold,
            "economic_regulation": self.economic_regulation,
            "protocol_version": PROTOCOL_VERSION,
            "learner_messages": [self.learner_sys] + self.learner_turns,
            "opponent_messages": [self.opponent_sys] + self.opponent_turns,
            "regulation_interventions": self.regulation_interventions,
            "format_failures": self.format_failures,
            "deal": deal,
        }, indent=2)}

    def _terminal(self, outcome, reward=0.0):
        return StepResult(
            next_observation=self.renderer.build_generation_prompt(self._learner_convo()),
            next_stop_condition=self.stop_condition,
            episode_done=True, reward=reward,
            metrics=_outcome_metrics(outcome, self.scenario.dataset_name,
                                     episode_length=float(self.turns_taken)),
            logs=self._logs(),
        )

    def _clear_offer(self):
        self.last_submit_deal = None
        self.last_submit_by = None

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        """One learner turn and one opponent turn; validate before state changes."""
        learner_msg, _ = self.renderer.parse_response(action)
        learner_text = get_text_content(learner_msg)
        self.turns_taken += 1
        self.learner_turns.append({"role": "assistant", "content": learner_text})
        learner_response = parse_response(
            learner_text, self.env, self.scenario.data, self.last_submit_by == "opponent"
        )
        learner_action = learner_response.action
        if learner_response.error:
            self.format_failures.append({
                "turn_index": self.turns_taken - 1, "speaker": "learner",
                "reason": learner_response.error, "raw_response": learner_text,
                "stop_reason": (extra or {}).get("stop_reason"),
            })
            self._clear_offer()
            return self._terminal("format_violation", -1.0)
        kind = learner_response.kind
        if kind == "ACCEPT_DEAL":
            assert self.last_submit_deal is not None
            return self._deal_result(self.env.invert_alloc(self.last_submit_deal, self.scenario.data))
        if kind == "WALK_AWAY":
            return self._terminal("walk_away")
        if kind == "SUBMIT_DEAL":
            parsed = self.env.parse_deal(learner_action)
            assert parsed is not None
            if self.learner_first_offer is None:
                self.learner_first_offer = parsed
            self.last_submit_deal, self.last_submit_by = parsed, "learner"
        else:
            self._clear_offer()
        self.opponent_turns.append({"role": "user", "content": partner_observation(
            learner_response, self.env, self.scenario.data)})

        # Even the final learner proposal receives its allotted opponent reply.
        opponent_msg = await self.opponent(self._opponent_convo())
        opponent_text = get_text_content(opponent_msg)
        self.turns_taken += 1
        opponent_response = parse_response(
            opponent_text, self.env, self.scenario.data, self.last_submit_by == "learner"
        )
        opponent_action = opponent_response.action
        valid = opponent_response.error is None
        if not valid:
            self.format_failures.append({
                "turn_index": self.turns_taken - 1, "speaker": "opponent",
                "reason": opponent_response.error, "raw_response": opponent_text,
            })
        if valid:
            kind = opponent_response.kind
            # An opponent proposal is already expressed in the opponent's seat.
            if kind == "SUBMIT_DEAL":
                allocation = self.env.parse_deal(opponent_action)
                assert allocation is not None
            elif kind == "ACCEPT_DEAL":
                assert self.last_submit_deal is not None
                allocation = self.env.invert_alloc(self.last_submit_deal, self.scenario.data)
            else:
                allocation = None
            if self.economic_regulation and allocation is not None:
                points = self.env.compute_points(allocation, self.scenario.data, self.opponent_id)
                if points < 0:
                    replacement = "Thought: This offer leaves me with too little.\nTalk: I can't accept that offer.\nAction: [REJECT_DEAL]"
                    self.regulation_interventions.append({
                        "turn_index": self.turns_taken - 1,
                        "opponent_utility": points,
                        "original_response": opponent_text,
                        "replacement_response": replacement,
                    })
                    opponent_text, kind = replacement, "REJECT_DEAL"
                    opponent_response = parse_response(replacement, self.env, self.scenario.data, False)
        self.opponent_turns.append({"role": "assistant", "content": opponent_text})
        self.learner_turns.append({"role": "user", "content": partner_observation(
            opponent_response, self.env, self.scenario.data)})
        if valid:
            if kind == "ACCEPT_DEAL":
                assert self.last_submit_deal is not None
                return self._deal_result(self.last_submit_deal)
            if kind == "WALK_AWAY":
                return self._terminal("walk_away")
            if kind == "SUBMIT_DEAL":
                self.last_submit_deal = self.env.parse_deal(opponent_action)
                self.last_submit_by = "opponent"
            else:
                self._clear_offer()
        else:
            # Invalid opponent actions consume a turn but cannot create a deal.
            self._clear_offer()
        self.round += 1
        if self.round >= self.max_rounds:
            return self._terminal("max_turns")
        return StepResult(
            next_observation=self.renderer.build_generation_prompt(self._learner_convo()),
            next_stop_condition=self.stop_condition, episode_done=False, reward=0.0,
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
    seed: int = 42
    threshold_scope: str = "selective"
    economic_regulation: bool = True

    async def make_envs(self) -> Sequence[Env]:
        # All completions within one GRPO group must start from the same prompt.
        learner_id = self.scenario.env.fixed_learner_role() or random.Random(
            derive_seed(self.seed, "role")).choice(self.scenario.agent_ids)
        return [
            NegotiationEnv(
                self.opponent, self.scenario, self.renderer, self.max_rounds,
                self.batna_threshold, learner_id=learner_id,
                seed=derive_seed(self.seed, "trajectory", i),
                threshold_scope=self.threshold_scope,
                economic_regulation=self.economic_regulation,
            ) for i in range(self.num_envs)
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
    seed: int = 42
    threshold_scope: str = "selective"
    economic_regulation: bool = True

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
                seed=derive_seed(self.seed, "group", index, i),
                threshold_scope=self.threshold_scope,
                economic_regulation=self.economic_regulation,
            )
            for i, s in enumerate(batch_scenarios)
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
    seed: int = 42
    threshold_scope: str = "selective"
    economic_regulation: bool = True
    renderer_name: str | None = None
    opponent_renderer_name: str | None = None
    opponent_max_tokens: int = 300
    tinker_project_id: str | None = None

    async def __call__(self) -> tuple[RLDataset, RLDataset | None]:
        service_client = create_service_client("opponent", project_id=self.tinker_project_id)

        opp_renderer_name = (
            self.opponent_renderer_name
            or self.renderer_name
            or model_info.get_recommended_renderer_name(self.opponent_model)
        )
        opp_tokenizer = get_tokenizer(self.opponent_model)
        opp_renderer = get_renderer(opp_renderer_name, opp_tokenizer)
        opp_client = service_client.create_sampling_client(
            base_model=self.opponent_model
        )
        record_tinker_sessions()
        opponent = TinkerMessageCompleter(
            sampling_client=opp_client,
            renderer=opp_renderer,
            max_tokens=self.opponent_max_tokens,
            temperature=self.opponent_temperature,
        )

        learner_renderer_name = (
            self.renderer_name
            or model_info.get_recommended_renderer_name(self.model_name_for_tokenizer)
        )
        learner_tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        learner_renderer = get_renderer(learner_renderer_name, learner_tokenizer)

        ds_specs = _parse_datasets_str(self.datasets)
        scenarios = load_all_scenarios(ds_specs, seed=derive_seed(self.seed, "data"))
        random.Random(derive_seed(self.seed, "order")).shuffle(scenarios)
        scenarios = scenarios * self.num_epochs

        dataset = NegotiationDataset(
            opponent=opponent,
            scenarios=scenarios,
            renderer=learner_renderer,
            batch_size=self.batch_size,
            group_size=self.group_size,
            max_rounds=self.max_rounds,
            batna_threshold=self.batna_threshold,
            seed=self.seed,
            threshold_scope=self.threshold_scope,
            economic_regulation=self.economic_regulation,
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
    "amazon": "data/ahp/train",
    "craigslist": "data/craigslist/revision/train.json",
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

    seed: int = 42
    threshold_scope: str = "selective"
    economic_regulation: bool = True
    model_name: str = "Qwen/Qwen3.6-35B-A3B"
    renderer_name: str | None = None
    opponent_renderer_name: str | None = None
    opponent_max_tokens: int = 300
    recipe_name: str = "rlvr_batna"
    tinker_project_id: str | None = None
    run_label: str | None = None
    datasets: str = "casino:data/casino/ca.train.csv"
    group_size: int = 2
    batch_size: int = 2
    num_epochs: int = 2
    learning_rate: float = 3e-5
    lora_rank: int = 32
    max_tokens: int = 300
    max_rounds: int = 6

    train_temperature: float = 1.0
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

    validate_config(cli)
    renderer_name = cli.renderer_name or model_info.get_recommended_renderer_name(
        cli.model_name
    )

    if cli.resume_run:
        run_name = cli.resume_run
        log_path = f"logs/negotiation/{run_name}"
    else:
        model = re.sub(r"[^a-z0-9.-]+", "-", cli.model_name.rsplit("/", 1)[-1].lower()).strip("-")
        condition = "surplus" if cli.batna_threshold == 0 else (
            f"{cli.threshold_scope}-tau{cli.batna_threshold:g}"
        )
        datasets = {ds["name"] for ds in _parse_datasets_str(cli.datasets)}
        if datasets == {"casino", "dnd", "amazon", "craigslist"}:
            tasks = "all"
        elif datasets == {"casino", "dnd"}:
            tasks = "multiOnly"
        elif datasets == {"amazon", "craigslist"}:
            tasks = "priceOnly"
        else:
            tasks = "-".join(sorted(datasets))
        regulation = "regOn" if cli.economic_regulation else "regOff"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        run_name = f"{model}-{condition}-{tasks}-{regulation}-seed{cli.seed}-{timestamp}"
        if cli.run_label:
            run_name = f"{cli.run_label}-{run_name}"
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
        seed=cli.seed,
        threshold_scope=cli.threshold_scope,
        economic_regulation=cli.economic_regulation,
        renderer_name=cli.renderer_name,
        opponent_renderer_name=cli.opponent_renderer_name,
        opponent_max_tokens=cli.opponent_max_tokens,
        tinker_project_id=cli.tinker_project_id,
    )

    config = train.Config(
        model_name=cli.model_name,
        recipe_name=cli.recipe_name,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=dataset_builder,
        learning_rate=cli.learning_rate,
        lora_rank=cli.lora_rank,
        max_tokens=cli.max_tokens,
        temperature=cli.train_temperature,
        kl_penalty_coef=cli.kl_penalty_coef,
        max_steps=cli.max_steps,
        save_every=cli.save_every,
        eval_every=cli.eval_every,
        load_checkpoint_path=cli.load_checkpoint_path,
        wandb_project=cli.wandb_project,
        wandb_name=run_name if cli.wandb_project else None,
    )

    return config, run_name


def derive_seed(seed: int, *parts) -> int:
    return int(digest([seed, *parts])[:8], 16) & 0x7fffffff


def validate_config(cli):
    if cli.max_steps is not None and (type(cli.max_steps) is not int or cli.max_steps < 1):
        raise ValueError("max_steps must be a positive integer or null")
    if cli.tinker_project_id is not None:
        from uuid import UUID
        try:
            UUID(cli.tinker_project_id)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("tinker_project_id must be a UUID") from exc
    if cli.run_label is not None and (
        not isinstance(cli.run_label, str)
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", cli.run_label)
    ):
        raise ValueError("run_label must be 1-64 letters, digits, underscores or hyphens")
    if type(cli.seed) is not int or not 0 <= cli.seed <= 0x7fffffff:
        raise ValueError("seed must be an integer in [0, 2147483647]")
    if cli.threshold_scope not in {"selective", "uniform"}:
        raise ValueError("threshold_scope must be selective or uniform")
    if type(cli.economic_regulation) is not bool:
        raise ValueError("economic_regulation must be a boolean")
    if not math.isfinite(cli.batna_threshold) or not 0 <= cli.batna_threshold <= 1:
        raise ValueError("batna_threshold must be in [0, 1]")
    for name in ("batch_size", "group_size", "num_epochs", "max_rounds", "max_tokens", "opponent_max_tokens"):
        if type(getattr(cli, name)) is not int or getattr(cli, name) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cli.group_size < 2:
        raise ValueError("GRPO requires at least two trajectories per group")
    if cli.resume_run and (Path(cli.resume_run).name != cli.resume_run or cli.load_checkpoint_path):
        raise ValueError("resume_run must be a run name and cannot accompany load_checkpoint_path")
    # Fail before any service clients are opened.
    load_all_scenarios(_parse_datasets_str(cli.datasets), seed=derive_seed(cli.seed, "data"))


def load_config(config_path, *, seed=None):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise TypeError("Training config must be a mapping")
    unknown = set(cfg) - set(chz.chz_fields(CLIConfig))
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    if isinstance(cfg.get("datasets"), list):
        parts = []
        for ds in cfg["datasets"]:
            if isinstance(ds, dict):
                if set(ds) - {"name", "data_path", "weight"}:
                    raise ValueError("Unknown dataset setting")
                name = ds["name"]
                path = ds["data_path"] if "data_path" in ds else _default_path(name)
                parts.append(f"{name}:{path}:{ds.get('weight', 1.0)}")
            else:
                parts.append(ds)
        cfg["datasets"] = ",".join(parts)
    if seed is not None:
        cfg["seed"] = seed
    # Record the SDK's environment fallback in the manifest as a resolved setting.
    if cfg.get("tinker_project_id") is None:
        cfg["tinker_project_id"] = os.environ.get("TINKER_PROJECT_ID") or None
    cli = CLIConfig(**cfg)
    validate_config(cli)
    return cli


def save_run_manifest(cli, config):
    """Pin config and training pool; refuse incompatible or legacy resumes."""
    import importlib.metadata
    import platform
    resolved = {name: getattr(cli, name) for name in chz.chz_fields(CLIConfig)}
    resolved.pop("resume_run")
    resolved.pop("behavior_if_log_dir_exists")
    specs = _parse_datasets_str(cli.datasets)
    pools = {ds["name"]: digest([serialize_scenario(ds["name"], sc)
             for sc in get_env(ds["name"]).load_scenarios(ds["data_path"])]) for ds in specs}
    modules = ["src/train.py", "src/tinker_tracking.py", "src/envs.py", "src/protocol.py", "src/prompts.py", "src/scenarios.py", "src/craigslist_data.py"]
    source_hashes = {p: digest(Path(p).read_text()) for p in modules}
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": 1, "python_version": platform.python_version(), "config": resolved, "training_pool_hashes": pools,
        "source_hashes": source_hashes,
        "packages": {p: importlib.metadata.version(p) for p in ["tinker", "tinker-cookbook", "torch", "chz"]},
        "seed_controls": ["dataset_resampling", "dataset_order", "group_roles"],
        "uncontrolled_randomness": ["server_lora_initialization", "learner_sampling", "opponent_sampling"],
        "note": "Independent run seed, not a guarantee of bitwise deterministic remote training.",
    }
    directory = Path(config.log_path)
    path = directory / "training_manifest.json"
    if cli.resume_run:
        if not path.exists() or json.loads(path.read_text()) != manifest:
            raise ValueError("Resume requires an identical revision training manifest; use a fresh run otherwise")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Training manifest already exists")
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def plan_execution(cli, config, run_steps=None):
    """Resolve a launch boundary without changing the declared experiment budget."""
    if run_steps is not None and (type(run_steps) is not int or run_steps < 1):
        raise ValueError("run_steps must be a positive integer")
    scenarios = load_all_scenarios(
        _parse_datasets_str(cli.datasets), seed=derive_seed(cli.seed, "data")
    )
    available_steps = len(scenarios) * cli.num_epochs // cli.batch_size
    total_steps = min(cli.max_steps, available_steps) if cli.max_steps is not None else available_steps
    if total_steps < 1:
        raise ValueError("Training pool is too small for one batch")
    checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path) if cli.resume_run else None
    if cli.resume_run and (checkpoint is None or not checkpoint.state_path):
        raise ValueError("Resume requires a saved training-state checkpoint; none was found")
    start = checkpoint.batch if checkpoint else 0
    if type(start) is not int or not 0 <= start <= total_steps:
        raise ValueError("Saved checkpoint has an invalid batch position for this training budget")
    end = min(start + run_steps, total_steps) if run_steps is not None else total_steps
    return {
        "start_step": start,
        "end_step": end,
        "steps_this_launch": end - start,
        "total_steps": total_steps,
        "requested_run_steps": run_steps,
        "resume_checkpoint": checkpoint.state_path if checkpoint else None,
    }


def record_execution(log_path, plan, status, **extra):
    """Keep launch boundaries separate from the immutable training manifest."""
    from datetime import datetime

    record = {"recorded_at": datetime.now(UTC).isoformat(), **plan, "status": status, **extra}
    with (Path(log_path) / "executions.jsonl").open("a") as file:
        file.write(json.dumps(record) + "\n")


def main(config_path: str = "configs/train.yaml", *, seed: int | None = None,
         dry_run: bool = False, run_steps: int | None = None,
         resume_run: str | None = None):
    """CLI entry point for negotiation training.

    Args:
        config_path: Path to YAML training config.
    """
    cli = load_config(config_path, seed=seed)
    if resume_run is not None:
        cli = chz.replace(cli, resume_run=resume_run)
    config, run_name = build_config(cli)
    plan = plan_execution(cli, config, run_steps)
    if cli.resume_run:
        # Read-only for resumed runs: enforce the full-budget manifest even in
        # dry-run mode, before any clients, files, or remote jobs are created.
        save_run_manifest(cli, config)
    if dry_run:
        print(json.dumps({name: getattr(cli, name) for name in chz.chz_fields(CLIConfig)}, indent=2))
        print(json.dumps({"execution": plan}, indent=2))
        print("Configuration and training data validated; no API clients or training jobs created.")
        return
    if not plan["steps_this_launch"]:
        print(f"Run {run_name} is already complete at step {plan['total_steps']}; no training launched.")
        return
    if not cli.resume_run:
        cli_utils.check_log_dir(
            config.log_path, behavior_if_exists=cli.behavior_if_log_dir_exists
        )
        save_run_manifest(cli, config)
    # Cookbook saves weights AND optimizer state at its execution endpoint.
    # Only this invocation's config is capped; the manifest retains max_steps.
    runtime_config = chz.replace(config, max_steps=plan["end_step"])
    record_execution(config.log_path, plan, "started")
    print(f"Run: {run_name}; training steps {plan['start_step']} -> {plan['end_step']} "
          f"of {plan['total_steps']} total.", flush=True)
    # Cookbook constructs the learner ServiceClient internally. Its SDK reads
    # this variable; the opponent receives the same project explicitly above.
    previous_project = os.environ.get("TINKER_PROJECT_ID")
    try:
        if cli.tinker_project_id:
            os.environ["TINKER_PROJECT_ID"] = cli.tinker_project_id
        with track_tinker_sessions(
            train,
            log_path=config.log_path,
            project_id=cli.tinker_project_id,
            metadata={
                "run_name": run_name,
                "recipe_name": cli.recipe_name,
                "condition": "surplus" if cli.batna_threshold == 0 else cli.threshold_scope,
                "seed": str(cli.seed),
                "threshold_scope": cli.threshold_scope,
                "batna_threshold": str(cli.batna_threshold),
                "economic_regulation": str(cli.economic_regulation).lower(),
                "total_steps": str(plan["total_steps"]),
                "segment_start_step": str(plan["start_step"]),
                "segment_end_step": str(plan["end_step"]),
            },
            wandb_enabled=bool(cli.wandb_project),
        ):
            asyncio.run(train.main(runtime_config))
        checkpoint = checkpoint_utils.get_last_checkpoint(config.log_path)
        if (run_steps is not None or cli.resume_run) and (
            checkpoint is None or checkpoint.batch != plan["end_step"] or not checkpoint.state_path
        ):
            raise RuntimeError("Training returned without the expected resumable endpoint checkpoint")
        status = "paused" if plan["end_step"] < plan["total_steps"] else "completed"
        record_execution(config.log_path, plan, status,
                         checkpoint=checkpoint.to_dict() if checkpoint else None)
        print(f"Run {status} at step {plan['end_step']} of {plan['total_steps']}.")
        if status == "paused":
            print(f"Resume with: uv run python main.py train --config {config_path} "
                  f"--seed {cli.seed} --resume-run {run_name}")
    except BaseException as exc:
        record_execution(config.log_path, plan, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error_type=type(exc).__name__)
        raise
    finally:
        if previous_project is None:
            os.environ.pop("TINKER_PROJECT_ID", None)
        else:
            os.environ["TINKER_PROJECT_ID"] = previous_project
