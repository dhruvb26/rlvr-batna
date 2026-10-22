"""Read-only assertions on the actual datums passed to diagnostic training."""

import json
from contextlib import contextmanager
from pathlib import Path

from tinker_cookbook.rl import train


def check_training_data(groups, advantages, data, metadata):
    got = {}
    tokens = 0
    masked = 0
    failures = 0
    length_mass = []
    for datum, meta in zip(data, metadata, strict=True):
        f = {k: v.to_torch() for k, v in datum.loss_fn_inputs.items()}
        mask = f["mask"].bool()
        key = (meta["group_idx"], meta["traj_idx"])
        assert all(len(v) == datum.model_input.length for v in f.values())
        assert (f["advantages"][~mask] == 0).all(), "Nonlearner loss detected"
        expected = float(advantages[key[0]][key[1]])
        assert (f["advantages"][mask] == expected).all(), "Wrong advantage"
        entry = got.setdefault(key, ([], []))
        entry[0].extend(f["target_tokens"][mask].tolist())
        entry[1].extend(f["logprobs"][mask].tolist())
        tokens += int(mask.sum())
        masked += int((~mask).sum())
    for gi, group in enumerate(groups):
        rewards = group.get_total_rewards()
        mean = sum(rewards) / len(rewards)
        for ti, traj in enumerate(group.trajectories_G):
            a = float(advantages[gi][ti])
            assert abs(a - (rewards[ti] - mean)) < 1e-5
            expected_tokens = [v for t in traj.transitions for v in t.ac.tokens]
            expected_lp = [v for t in traj.transitions for v in t.ac.logprobs]
            actual_tokens, actual_lp = got[(gi, ti)]
            assert actual_tokens == expected_tokens, "Target/action mismatch"
            assert len(actual_lp) == len(expected_lp)
            assert all(abs(x - y) < 1e-4 for x, y in zip(actual_lp, expected_lp)), (
                "Logprob alignment mismatch"
            )
            if traj.transitions[-1].metrics.get("outcome_format_violation"):
                failures += 1
                assert a <= 1e-7, "Failed trajectory has positive advantage"
            length_mass.append((len(expected_tokens), abs(a) * len(expected_tokens)))
    total = sum(m for _, m in length_mass)
    top = sorted(length_mass, reverse=True)[: max(1, (len(length_mass) + 9) // 10)]
    return {
        "groups": len(groups),
        "trajectories": len(length_mass),
        "action_tokens": tokens,
        "masked_context_tokens": masked,
        "format_failures": failures,
        "checks_passed": True,
        "top_decile_abs_advantage_token_share": sum(m for _, m in top) / total
        if total
        else 0,
    }


@contextmanager
def audit_training_inputs(log_path):
    original = train.assemble_training_data

    def checked(groups, advantages):
        data, meta = original(groups, advantages)
        result = check_training_data(groups, advantages, data, meta)
        with (Path(log_path) / "training_input_audit.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        return data, meta

    train.assemble_training_data = checked
    try:
        yield
    finally:
        train.assemble_training_data = original
