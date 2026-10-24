"""Verify the completed Qwen-opponent batch and compare with the original paper.

Run from the repository root with .venv/bin/python. No model calls.
"""
import hashlib
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LAUNCH = ROOT / 'files/off-the-shelf-qwen36-eval-2026-10-23'
DS = ['amazon', 'casino', 'craigslist', 'dnd', 'ji']
COUNTS = dict(zip(DS, [93, 124, 100, 100, 100]))
read = lambda p: json.loads(p.read_text())
clip = lambda x: max(-1., min(1., x))
cfg = yaml.safe_load((LAUNCH / 'eval-qwen36.yaml').read_text())
runs = sorted((ROOT / cfg['storage_dir']).glob('run_*'))
assert len(runs) == 1, 'Select the authorized run explicitly if multiple runs exist.'
run = runs[0]
expected = [(ds, m, run / ds / f"{m['learner']['label']}_vs_{m['opponent']['label']}")
            for m in cfg['matchups'] for ds in DS]
missing = [str(path.relative_to(ROOT)) for ds, m, path in expected if not (path / 'metrics.json').exists()]
if missing:
    raise SystemExit(f'INCOMPLETE: {45-len(missing)}/45 saved cells; no final report published.')
cells = []
for ds, matchup, path in expected:
    meta = read(path / 'metrics.json')
    assert meta['failed_episodes'] == 0, ('Retry only execution failures first', path)
    assert meta['protocol_version'] == 'explicit-allocation-v3-turn-limit-v1'
    assert meta['economic_regulation'] is True
    raw = (path / 'episodes.json').read_bytes()
    eps = json.loads(raw)
    panel = read(ROOT / f'data/evaluation_manifests/revision/main/{ds}.json')
    wanted = {(e['scenario_id'], e['repeat']): e for e in panel['episodes']}
    assert len(eps) == meta['scheduled_episodes'] == COUNTS[ds]
    assert len({(e['scenario_id'], e['repeat']) for e in eps}) == len(eps)
    for e in eps:
        entry = wanted[e['scenario_id'], e['repeat']]
        assert all(e[k] == entry[k] for k in ['learner_agent_id', 'opponent_agent_id', 'learner_goes_first', 'persona'])
        assert e['economic_regulation'] is True and e['protocol_version'] == meta['protocol_version']
    br = [clip(e['bargained_ratio']) for e in eps if e['bargained_ratio'] is not None]
    deals = sum(e['outcome'] == 'deal' for e in eps)
    assert len(br) == deals
    pareto = [e['pareto_efficient'] for e in eps if e['pareto_efficient'] is not None]
    bids = [e['first_bid_ratio'] for e in eps if e['first_bid_ratio'] is not None]
    quality = st.mean(br) if br else None
    pr = st.mean(pareto) if pareto else None
    fb = st.mean(bids) if bids else None
    saved = meta['overall']
    for value, key in [(quality, 'avg_bargained_ratio'), (pr, 'pareto_rate'), (fb, 'avg_first_bid_ratio')]:
        assert (value is None and saved[key] is None) or abs(value - saved[key]) <= .000050001
    assert saved['deal_count'] == deals and saved['total_episodes'] == len(eps)
    assert abs(saved['deal_rate'] - deals/len(eps)) < 1e-12
    cells.append(dict(learner=matchup['learner']['label'], dataset=ds,
                      quality=quality, utility=sum(br)/len(eps), deal_rate=deals/len(eps),
                      pareto=pr, first_bid=fb, episodes=len(eps), deals=deals,
                      source=str((path/'episodes.json').relative_to(ROOT)),
                      sha256=hashlib.sha256(raw).hexdigest()))

# Trained policies were already evaluated against this same native Qwen3.6 opponent.
trained = read(ROOT / 'ijpr-revision/original-metrics-2026-10-23/results.json')['cells']
for c in trained:
    if c['opponent'] == 'Qwen3.6 native' and c['condition'] in ['Selective τ=.4', 'Surplus', 'Uniform τ=.4']:
        assert c['protocols'] == ['explicit-allocation-v3-turn-limit-v1']
        cells.append(dict(learner=c['condition'], seed=c['seed'], dataset=c['dataset'],
                          quality=c['clipped_br'], utility=c['clipped_br']*c['deal_rate'],
                          deal_rate=c['deal_rate'], pareto=c['pareto_rate'], first_bid=c['first_bid_ratio'],
                          episodes=c['episodes'], deals=c['deals'], source=c['source'], sha256=c['source_sha256']))
groups = defaultdict(list)
for c in cells:
    groups[c['learner'], c['dataset']].append(c)
rows = {}
for (learner, ds), values in groups.items():
    rows[learner, ds] = {key: (st.mean(c[key] for c in values) if all(c[key] is not None for c in values) else None)
                         for key in ['quality', 'utility', 'deal_rate', 'pareto', 'first_bid']}

historical = (ROOT / 'ijpr-revision/original-metrics-2026-10-23/original-paper-tables.md').read_text()
def old_table(number):
    return historical.split(f'## Table {number}:', 1)[1].split('## Table ', 1)[0].split('\n\n', 1)[1].strip()

original_br = {}
for line in old_table(1).splitlines()[2:]:
    fields = [v.strip() for v in line.strip('|').split('|')]
    original_br[fields[0]] = st.mean(float(re.match(r'(-?[\d.]+)', v).group(1)) for v in fields[1:])
mapping = [('Threshold (Ours)', 'Selective τ=.4'), ('Surplus', 'Surplus'), ('Uniform (new)', 'Uniform τ=.4'),
           ('Qwen3-30B', 'qwen3-30b-a3b-instruct-2507'), ('Qwen3-30B-T', None),
           ('Qwen3-235B', 'qwen3-235b-a22b-2507'), ('Qwen3-235B-T', 'qwen3-235b-a22b-thinking-2507'),
           ('Qwen3.6-35B', 'qwen3.6-35b-a3b-untrained-nonthinking'), ('GPT-5.4', 'gpt-5.4-off-the-shelf-openai'),
           ('GPT-5.4-mini', 'gpt-5.4-mini-off-the-shelf-openai'), ('DeepSeek-V3.1', 'deepseek-chat-v3.1'),
           ('Kimi-K2-T', 'kimi-k2-thinking'), ('Llama-4-Maverick', 'llama-4-maverick')]
macros = []
for old, learner in mapping:
    if learner is None:
        continue
    seeds = sorted({c.get('seed', 0) for c in cells if c['learner'] == learner})
    per_seed = []
    for seed in seeds:
        selected = [c for c in cells if c['learner'] == learner and c.get('seed', 0) == seed]
        assert len(selected) == 5
        per_seed.append(dict(quality=st.mean(c['quality'] for c in selected) if all(c['quality'] is not None for c in selected) else None,
                             utility=st.mean(c['utility'] for c in selected),
                             deal_rate=sum(c['deals'] for c in selected)/sum(c['episodes'] for c in selected)))
    macros.append(dict(learner=learner, old_label=old, original_quality_rounded=original_br.get(old),
                       **{k: (st.mean(v[k] for v in per_seed) if all(v[k] is not None for v in per_seed) else None)
                          for k in ['quality', 'utility', 'deal_rate']}))
fmt = lambda x: '—' if x is None else f'{x:.4f}'
text = ['# Revised results against base Qwen3.6', '',
        'Revised trained-model rows (Threshold, Surplus and Uniform) average seeds 42, 43 and 44. Off-the-shelf models have one evaluation per scenario. BR cells show mean clipped bargained ratio (deal %); — indicates unavailable results.', '',
        '## Table 1 — mean clipped bargained ratio (deal %)', '',
        '| Learner | AHP | CA | CRA | DnD | JI |', '|---|---:|---:|---:|---:|---:|']
for old, learner in mapping:
    values = ['—']*5 if learner is None else [f"{rows[learner,ds]['quality']:.2f} ({100*rows[learner,ds]['deal_rate']:.0f})" if rows[learner,ds]['quality'] is not None else f"— ({100*rows[learner,ds]['deal_rate']:.0f})" for ds in DS]
    text.append('| '+old+' | '+' | '.join(values)+' |')
text += ['', '## Table 2 — Pareto efficiency and first-bid ratios', '',
         '| Learner | Pareto CA ↑ | Pareto DnD ↑ | First bid AHP ↓ | First bid CRA ↓ |', '|---|---:|---:|---:|---:|']
for old, learner in mapping:
    values = ['—']*4 if learner is None else [fmt(rows[learner,ds][key]) for ds,key in [('casino','pareto'),('dnd','pareto'),('amazon','first_bid'),('craigslist','first_bid')]]
    text.append('| '+old+' | '+' | '.join(values)+' |')
(OUT/'results.json').write_text(json.dumps(dict(run=str(run.relative_to(ROOT)), completed_new_episodes=4653, new_dataset_cells=45, original_source='ijpr-submission/paper.tex', summary=macros, cells=cells),indent=2,ensure_ascii=False)+'\n')
(OUT/'report.md').write_text('\n'.join(text)+'\n')
print('Verified 45 new cells / 4,653 episodes. Report:',OUT/'report.md')
