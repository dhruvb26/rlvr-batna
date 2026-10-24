"""Verify GPT-5.4 baseline evaluations and rebuild original Tables 3–4 offline."""
import hashlib
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
LAUNCH = ROOT/'files/off-the-shelf-gpt54-eval-2026-10-23'
DS = ['amazon','casino','craigslist','dnd','ji']
COUNTS = dict(zip(DS,[93,124,100,100,100]))
V3 = 'explicit-allocation-v3-turn-limit-v1'
read = lambda p: json.loads(p.read_text())
clip = lambda x: max(-1.,min(1.,x))
digest = lambda x: hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
cfg = yaml.safe_load((LAUNCH/'eval-gpt54.yaml').read_text())
runs = sorted((ROOT/cfg['storage_dir']).glob('run_*'))
assert len(runs)==1, 'Select the authorized run explicitly if multiple runs exist.'
run = runs[0]
assert len(cfg['matchups'])==8
group = digest(dict(opponent=cfg['matchups'][0]['opponent'],protocol_version=V3,
                    economic_regulation=True,max_turns=12,temperature=.7,top_p=1.))
specs = [(ds,m['learner']['label'],run/ds/f"{m['learner']['label']}_vs_{m['opponent']['label']}",False)
         for m in cfg['matchups'] for ds in DS]
missing = [str(p.relative_to(ROOT)) for ds,label,p,reused in specs if not (p/'metrics.json').exists()]
if missing:
    raise SystemExit(f'INCOMPLETE: {40-len(missing)}/40 new cells; no final report published.')
base = ROOT/'logs/eval/base-qwen36-gpt54-openai/run_20261004_202248_117114'
for ds in DS:
    paths = list((base/ds).glob('*/metrics.json'))
    assert len(paths)==1
    specs.append((ds,'qwen3.6-35b-a3b-untrained-nonthinking',paths[0].parent,True))


def measure(ds,label,path,*,reused=False,seed=None,require_group=True):
    meta = read(path/'metrics.json')
    assert meta['failed_episodes']==0, ('Retry outstanding execution failures first',path)
    assert meta['economic_regulation'] is True
    if (path/'failed.json').exists():
        assert not read(path/'failed.json'), ('Outstanding execution failures', path)
    if require_group:
        assert meta['protocol_version']==V3 and meta['comparison_group']==group
    assert meta['evaluation_settings']==dict(max_turns=12,temperature=.7,top_p=1.,economic_regulation=True)
    raw = (path/'episodes.json').read_bytes()
    eps = json.loads(raw)
    panel = read(ROOT/f'data/evaluation_manifests/revision/main/{ds}.json')
    assert meta['manifest_sha256']==digest(panel)
    wanted = {(e['scenario_id'],e['repeat']):e for e in panel['episodes']}
    assert len(eps)==meta['scheduled_episodes']==COUNTS[ds]
    assert len({(e['scenario_id'],e['repeat']) for e in eps})==len(eps)
    for e in eps:
        entry = wanted[e['scenario_id'],e['repeat']]
        assert all(e[k]==entry[k] for k in ['learner_agent_id','opponent_agent_id','learner_goes_first','persona'])
        assert e['economic_regulation'] is True and e['protocol_version']==meta['protocol_version']
    br = [clip(e['bargained_ratio']) for e in eps if e['bargained_ratio'] is not None]
    deals = sum(e['outcome']=='deal' for e in eps)
    assert len(br)==deals
    pareto = [e['pareto_efficient'] for e in eps if e['pareto_efficient'] is not None]
    bids = [e['first_bid_ratio'] for e in eps if e['first_bid_ratio'] is not None]
    if ds in ['casino','dnd']:
        assert len(pareto)==deals
    values=dict(quality=st.mean(br) if br else None, utility=sum(br)/len(eps),deal_rate=deals/len(eps),
                pareto=st.mean(pareto) if pareto else None, first_bid=st.mean(bids) if bids else None)
    saved=meta['overall']
    for key,saved_key in [('quality','avg_bargained_ratio'),('pareto','pareto_rate'),('first_bid','avg_first_bid_ratio')]:
        a,b=values[key],saved[saved_key]
        assert (a is None and b is None) or (a is not None and b is not None and abs(a-b)<=.000050001)
    assert saved['deal_count']==deals and saved['total_episodes']==len(eps)
    assert abs(saved['deal_rate']-values['deal_rate'])<1e-12
    return dict(learner=label,dataset=ds,seed=seed,reused=reused,episodes=len(eps),deals=deals,
                protocol=meta['protocol_version'],comparison_group=meta['comparison_group'],
                source=str((path/'episodes.json').relative_to(ROOT)),sha256=hashlib.sha256(raw).hexdigest(),**values)


cells=[measure(ds,label,path,reused=reused) for ds,label,path,reused in specs]
assert sum(c['episodes'] for c in cells if not c['reused'])==4136
assert sum(c['episodes'] for c in cells)==4653
trained_labels=['Selective τ=.2', 'Selective τ=.4', 'Selective τ=.6', 'Surplus', 'Uniform τ=.4']
refresh_launch=ROOT/'files/seed42-gpt54-v3-eval-2026-10-23'
refresh_cfg=yaml.safe_load((refresh_launch/'eval-seed42-v3.yaml').read_text())
refresh_runs=sorted((ROOT/refresh_cfg['storage_dir']).glob('run_*'))
assert len(refresh_runs)==1, 'Select the authorized seed42 re-evaluation run explicitly.'
refresh_run=refresh_runs[0]
refresh_labels={
    'Surplus':'qwen3.6-35b-a3b-surplus-temp1-seed42-step80',
    'Selective τ=.4':'qwen3.6-35b-a3b-selective-tau0.4-temp1-seed42-step80',
    'Uniform τ=.4':'qwen3.6-35b-a3b-uniform-tau0.4-temp1-seed42-step80'}
refresh_paths={(label,ds):refresh_run/ds/f"{name}_vs_gpt-5.4-openai"
               for label,name in refresh_labels.items() for ds in DS}
refresh_missing=[str(p.relative_to(ROOT)) for p in refresh_paths.values() if not (p/'metrics.json').exists()]
if refresh_missing:
    raise SystemExit(f'INCOMPLETE: {15-len(refresh_missing)}/15 seed42 refresh cells; prior published report retained.')
for c in read(ROOT/'ijpr-revision/original-metrics-2026-10-23/results.json')['cells']:
    if c['opponent']=='GPT-5.4' and c['condition'] in trained_labels:
        source=ROOT/c['source']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==c['source_sha256']
        if c['seed']==42 and c['condition'] in refresh_labels:
            row=measure(c['dataset'],c['condition'],refresh_paths[c['condition'],c['dataset']],seed=42,require_group=True)
        else:
            row=measure(c['dataset'],c['condition'],source.parent,reused=True,seed=c['seed'],require_group=True)
            assert abs(row['quality']-c['clipped_br'])<1e-12
        cells.append(row)
for label in trained_labels:
    records = [c for c in cells if c['learner'] == label]
    assert len(records) == 15, ('Expected 15 trained cells', label)
    assert {(c['seed'], c['dataset']) for c in records} == {(seed, ds) for seed in [42, 43, 44] for ds in DS}
    assert sum(c['episodes'] for c in records) == 1551
groups=defaultdict(list)
for c in cells:
    groups[c['learner'],c['dataset']].append(c)
rows={(label,ds):{k:st.mean(c[k] for c in records) if all(c[k] is not None for c in records) else None
                 for k in ['quality','utility','deal_rate','pareto','first_bid']}
      for (label,ds),records in groups.items()}
historical=(ROOT/'ijpr-revision/original-metrics-2026-10-23/original-paper-tables.md').read_text()
def old_table(n):
    return historical.split(f'## Table {n}:',1)[1].split('## Table ',1)[0].split('\n\n',1)[1].strip()
old_means={}
for line in old_table(3).splitlines()[2:]:
    fields=[v.strip() for v in line.strip('|').split('|')]
    old_means[fields[0]]=st.mean(float(re.match(r'(-?[\d.]+)',v).group(1)) for v in fields[1:])
mapping=[('Threshold (Ours, τ=.4)','Selective τ=.4'),('Selective τ=.2','Selective τ=.2'),
         ('Selective τ=.6','Selective τ=.6'),('Surplus','Surplus'),('Uniform (new)','Uniform τ=.4'),
         ('Qwen3-30B','qwen3-30b-a3b-instruct-2507'),('Qwen3-30B-T',None),
         ('Qwen3-235B','qwen3-235b-a22b-2507'),('Qwen3-235B-T','qwen3-235b-a22b-thinking-2507'),
         ('Qwen3.6-35B','qwen3.6-35b-a3b-untrained-nonthinking'),('GPT-5.4','gpt-5.4-off-the-shelf-openai'),
         ('GPT-5.4-mini','gpt-5.4-mini-off-the-shelf-openai'),('DeepSeek-V3.1','deepseek-chat-v3.1'),
         ('Kimi-K2-T','kimi-k2-thinking'),('Llama-4-Maverick','llama-4-maverick')]
def macro(label,seeds=None):
    selected=[c for c in cells if c['learner']==label and (seeds is None or c['seed'] in seeds)]
    by_seed=defaultdict(list)
    for c in selected:
        by_seed[c['seed']].append(c)
    statistics=[]
    for seed,records in by_seed.items():
        assert len(records)==5
        statistics.append(dict(seed=seed,quality=st.mean(c['quality'] for c in records) if all(c['quality'] is not None for c in records) else None,
                               utility=st.mean(c['utility'] for c in records),deal_rate=sum(c['deals'] for c in records)/sum(c['episodes'] for c in records)))
    return dict(learner=label,seeds=list(by_seed),per_seed=statistics,
                **{k:st.mean(s[k] for s in statistics) if all(s[k] is not None for s in statistics) else None for k in ['quality','utility','deal_rate']})
summary=[dict(old_label=old,original_quality_rounded=old_means.get('Threshold (Ours)' if label == 'Selective τ=.4' else old),**macro(label)) for old,label in mapping if label]
matched=[macro(label,[43,44]) for label in trained_labels]
best=max((s for s in summary if s['learner'] not in trained_labels),key=lambda s:s['utility'])
selective=next(s for s in summary if s['learner']=='Selective τ=.4')
surplus=next(s for s in summary if s['learner']=='Surplus')
fmt=lambda x:'—' if x is None else f'{x:.4f}'
text = ['# Revised results against GPT-5.4', '',
        'All trained-model rows (Selective τ=.2/.4/.6, Surplus and Uniform τ=.4) average seeds 42, 43 and 44. Off-the-shelf models have one evaluation per scenario. BR cells show mean clipped bargained ratio (deal %); — indicates unavailable results.', '',
        '## Table 3 — mean clipped bargained ratio (deal %)', '',
        '| Learner | AHP | CA | CRA | DnD | JI |', '|---|---:|---:|---:|---:|---:|']
for old, learner in mapping:
    values = ['—']*5 if learner is None else [f"{rows[learner,ds]['quality']:.2f} ({100*rows[learner,ds]['deal_rate']:.0f})" if rows[learner,ds]['quality'] is not None else f"— ({100*rows[learner,ds]['deal_rate']:.0f})" for ds in DS]
    text.append('| '+old+' | '+' | '.join(values)+' |')
text += ['', '## Table 4 — Pareto efficiency and first-bid ratios', '',
         '| Learner | Pareto CA ↑ | Pareto DnD ↑ | First bid AHP ↓ | First bid CRA ↓ |', '|---|---:|---:|---:|---:|']
for old, learner in mapping:
    values = ['—']*4 if learner is None else [fmt(rows[learner,ds][key]) for ds,key in [('casino','pareto'),('dnd','pareto'),('amazon','first_bid'),('craigslist','first_bid')]]
    text.append('| '+old+' | '+' | '.join(values)+' |')
results=dict(run=str(run.relative_to(ROOT)),completed_new_episodes=4136,new_dataset_cells=40,reused_baseline_episodes=517,
             off_the_shelf_total_episodes=4653,original_source='ijpr-submission/paper.tex',
             trained_seeds=[42,43,44],trained_dataset_cells=75,reused_threshold_dataset_cells=30,reused_threshold_episodes=3102,
             summary=summary,matched_protocol_summary=matched,cells=cells,
             seed42_refresh_run=str(refresh_run.relative_to(ROOT)),seed42_refresh_new_episodes=1551,seed42_refresh_dataset_cells=15,
             historical_handoff_summary=read(refresh_launch/'previous-comparison-results.json')['summary'])
(OUT/'results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False)+'\n')
(OUT/'report.md').write_text('\n'.join(text)+'\n')
print('Verified 45 baseline cells /4,653 episodes and 75 trained cells /7,755 episodes, including 30 reused threshold cells and 15 refreshed seed42 cells. Report:',OUT/'report.md')
print('Three-seed current-evaluation utility: Selective',selective['utility'],'Surplus',surplus['utility'],'best off-the-shelf',best['old_label'],best['utility'])
