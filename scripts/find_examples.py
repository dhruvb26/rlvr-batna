"""Extract full conversations for the best contrastive examples."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = str(ROOT / "logs/baseline_qwen/run_20260519_022034")
OUT_DIR = ROOT / "report" / "conversations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

files = {
    "dnd_batna": f"{BASE}/dnd/qwen3-30b-a3b-instruct-2507-batna_vs_qwen3-30b-a3b-instruct-2507/episodes.json",
    "dnd_surplus": f"{BASE}/dnd/qwen3-30b-a3b-instruct-2507-surplus_vs_qwen3-30b-a3b-instruct-2507/episodes.json",
    "casino_batna": f"{BASE}/casino/qwen3-30b-a3b-instruct-2507-batna_vs_qwen3-30b-a3b-instruct-2507/episodes.json",
    "casino_surplus": f"{BASE}/casino/qwen3-30b-a3b-instruct-2507-surplus_vs_qwen3-30b-a3b-instruct-2507/episodes.json",
}

data = {}
for name, path in files.items():
    with open(path) as f:
        data[name] = json.load(f)

TARGET_IDS = [99, 2, 47]


def format_episode(label, ep):
    lines = []
    lines.append("=" * 80)
    lines.append(label)
    lines.append("=" * 80)
    lines.append(f"episode_id: {ep['episode_id']}")
    lines.append(f"outcome: {ep['outcome']}")
    lines.append(f"bargained_ratio: {ep.get('bargained_ratio')}")
    lines.append(f"learner_points: {ep.get('learner_points')}")
    lines.append(f"opponent_points: {ep.get('opponent_points')}")
    lines.append(f"num_turns: {ep.get('num_turns')}")
    lines.append(f"who_terminated: {ep.get('who_terminated')}")
    lines.append(f"first_bid_ratio: {ep.get('first_bid_ratio')}")
    lines.append(f"pareto_efficient: {ep.get('pareto_efficient')}")
    lines.append(f"joint_surplus_norm: {ep.get('joint_surplus_norm')}")

    lines.append(
        f"\n--- LEARNER MESSAGES ({len(ep.get('learner_messages', []))} messages) ---"
    )
    for i, msg in enumerate(ep.get("learner_messages", [])):
        lines.append(f"\n[{i}] role={msg.get('role', 'unknown')}")
        lines.append(msg.get("content", ""))

    lines.append(
        f"\n--- OPPONENT MESSAGES ({len(ep.get('opponent_messages', []))} messages) ---"
    )
    for i, msg in enumerate(ep.get("opponent_messages", [])):
        lines.append(f"\n[{i}] role={msg.get('role', 'unknown')}")
        lines.append(msg.get("content", ""))
    return "\n".join(lines)


def save_episodes(filename, *blocks):
    text = "\n\n".join(blocks)
    path = OUT_DIR / filename
    path.write_text(text)
    print(f"  saved {path}")


dnd_batna_by_id = {e["episode_id"]: e for e in data["dnd_batna"]}
dnd_surplus_by_id = {e["episode_id"]: e for e in data["dnd_surplus"]}
casino_batna_by_id = {e["episode_id"]: e for e in data["casino_batna"]}
casino_surplus_by_id = {e["episode_id"]: e for e in data["casino_surplus"]}

# ── CaSiNo episode 64 (used in paper Appendix E.2) ──
save_episodes(
    "casino_episode_64.txt",
    format_episode(
        "THRESHOLD (BATNA) AGENT — CaSiNo Episode 64", casino_batna_by_id[64]
    ),
    format_episode("SURPLUS AGENT — CaSiNo Episode 64", casino_surplus_by_id[64]),
)

# ── JI episode 32 (used in paper Appendix E.3) ──
ji_files = {
    "ji_base": f"{BASE}/ji/qwen3-30b-a3b-instruct-2507_vs_qwen3-30b-a3b-instruct-2507/episodes.json",
    "ji_batna": f"{BASE}/ji/qwen3-30b-a3b-instruct-2507-batna_vs_qwen3-30b-a3b-instruct-2507/episodes.json",
}
ji_data = {}
for name, path in ji_files.items():
    with open(path) as f:
        ji_data[name] = {e["episode_id"]: e for e in json.load(f)}

save_episodes(
    "ji_episode_32.txt",
    format_episode("UNTRAINED (BASE) AGENT — JI Episode 32", ji_data["ji_base"][32]),
    format_episode(
        "TRAINED (THRESHOLD) AGENT — JI Episode 32", ji_data["ji_batna"][32]
    ),
)

# ── CRA episode 2 (used in paper Appendix E.1) ──
cra_files = {
    "cra_base": f"{BASE}/craigslist/qwen3-30b-a3b-instruct-2507_vs_qwen3-30b-a3b-instruct-2507/episodes.json",
    "cra_batna": f"{BASE}/craigslist/qwen3-30b-a3b-instruct-2507-batna_vs_qwen3-30b-a3b-instruct-2507/episodes.json",
}
cra_data = {}
for name, path in cra_files.items():
    with open(path) as f:
        cra_data[name] = {e["episode_id"]: e for e in json.load(f)}

save_episodes(
    "cra_episode_2.txt",
    format_episode("UNTRAINED (BASE) AGENT — CRA Episode 2", cra_data["cra_base"][2]),
    format_episode(
        "TRAINED (THRESHOLD) AGENT — CRA Episode 2", cra_data["cra_batna"][2]
    ),
)
