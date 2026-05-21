"""Render charts of the active run from the local mirror. No GPU, no inference."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RUN_FILE = Path('/home/ermer/devproj/js/react/ablitMD/data/vastai_mirror/run_2026-05-17T19-57-17-480Z_8696f34a.json')
OUT = Path('/home/ermer/devproj/js/react/ablitMD/data/charts')
OUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = 'Qwen/Qwen3.6-27B'
MODE = 'non_thinking'
MODE_ORDER = ['hard', 'redirect', 'disclaimer', 'none']
MODE_COLORS = {'hard': '#c0392b', 'redirect': '#e67e22', 'disclaimer': '#f1c40f', 'none': '#27ae60'}

plt.rcParams.update({
  'figure.facecolor': '#1a1a1a', 'axes.facecolor': '#1a1a1a',
  'axes.edgecolor': '#666', 'axes.labelcolor': '#ddd',
  'xtick.color': '#ddd', 'ytick.color': '#ddd', 'text.color': '#eee',
  'axes.titlecolor': '#fff', 'font.size': 10,
})

run = json.loads(RUN_FILE.read_text())
prompts = run['prompts']

classified = []
for prompt in prompts:
  result = prompt.get('model_results', {}).get(MODEL_ID, {}).get(MODE)
  if result:
    classified.append({
      'category': prompt['category'],
      'category_group': prompt['category_group'],
      'type': prompt['type'],
      'refused': result['refused'],
      'refusal_mode': result.get('refusal_mode', 'none'),
    })

print(f'classified: {len(classified)} / {len(prompts)} prompts')

# ─── Chart 1: overall refusal mode distribution ─────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
counts = Counter(item['refusal_mode'] for item in classified)
values = [counts.get(mode, 0) for mode in MODE_ORDER]
colors = [MODE_COLORS[mode] for mode in MODE_ORDER]
bars = ax.bar(MODE_ORDER, values, color=colors, edgecolor='#333')
for bar, val in zip(bars, values):
  ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
          str(val), ha='center', color='#fff', fontweight='bold')
ax.set_title(f'Refusal Mode Distribution  ({len(classified)} of {len(prompts)} classified)')
ax.set_ylabel('count')
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / '01_refusal_modes.png', dpi=110, facecolor=plt.rcParams['figure.facecolor'])
plt.close()

# ─── Chart 2: harmful vs harmless (over-refusal indicator) ──────────────────
fig, ax = plt.subplots(figsize=(8, 5))
harmful = [item for item in classified if item['type'] == 'harmful']
harmless = [item for item in classified if item['type'] == 'harmless']
x = np.arange(len(MODE_ORDER))
width = 0.38
harmful_counts = [sum(1 for h in harmful if h['refusal_mode'] == mode) for mode in MODE_ORDER]
harmless_counts = [sum(1 for h in harmless if h['refusal_mode'] == mode) for mode in MODE_ORDER]
ax.bar(x - width/2, harmful_counts, width, label=f'harmful (n={len(harmful)})', color='#c0392b', edgecolor='#333')
ax.bar(x + width/2, harmless_counts, width, label=f'harmless (n={len(harmless)})', color='#3498db', edgecolor='#333')
ax.set_xticks(x); ax.set_xticklabels(MODE_ORDER)
ax.set_title('Refusal Mode by Prompt Type  (harmless+refused = over-refusal)')
ax.set_ylabel('count'); ax.legend()
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / '02_by_type.png', dpi=110, facecolor=plt.rcParams['figure.facecolor'])
plt.close()

# ─── Chart 3: stacked refusal mode per category_group ───────────────────────
group_mode = defaultdict(lambda: Counter())
for item in classified:
  group_mode[item['category_group']][item['refusal_mode']] += 1

groups_sorted = sorted(group_mode.keys(), key=lambda group: -sum(group_mode[group].values()))
fig, ax = plt.subplots(figsize=(11, 6))
bottoms = np.zeros(len(groups_sorted))
for mode in MODE_ORDER:
  values = np.array([group_mode[group].get(mode, 0) for group in groups_sorted])
  ax.bar(groups_sorted, values, bottom=bottoms, label=mode, color=MODE_COLORS[mode], edgecolor='#222')
  bottoms += values
ax.set_title('Refusal Mode by Category Group (stacked)')
ax.set_ylabel('count')
ax.legend(loc='upper right')
plt.xticks(rotation=35, ha='right')
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / '03_stacked_by_group.png', dpi=110, facecolor=plt.rcParams['figure.facecolor'])
plt.close()

# ─── Chart 4: refusal RATE per group (harmful only) ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
group_rate = {}
for group in groups_sorted:
  harmful_in_group = [item for item in classified if item['category_group'] == group and item['type'] == 'harmful']
  if not harmful_in_group:
    continue
  refused_count = sum(1 for h in harmful_in_group if h['refused'])
  group_rate[group] = (refused_count / len(harmful_in_group), len(harmful_in_group))

groups_for_rate = list(group_rate.keys())
rates = [group_rate[g][0] for g in groups_for_rate]
ns = [group_rate[g][1] for g in groups_for_rate]
bars = ax.bar(groups_for_rate, rates, color='#c0392b', edgecolor='#333')
for bar, n in zip(bars, ns):
  ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
          f'n={n}', ha='center', color='#ddd', fontsize=9)
ax.set_title('Refusal Rate by Category Group (harmful prompts only)')
ax.set_ylabel('refusal rate')
ax.set_ylim(0, 1.05)
plt.xticks(rotation=35, ha='right')
ax.grid(axis='y', alpha=0.2)
plt.tight_layout()
plt.savefig(OUT / '04_refusal_rate_by_group.png', dpi=110, facecolor=plt.rcParams['figure.facecolor'])
plt.close()

# ─── Chart 5: direction magnitudes per category, per refusal mode ───────────
dr = run.get('direction_results') or {}
if dr:
  cats_with_dirs = [c for c, v in dr.items() if v.get('by_mode')]
  n_cats = len(cats_with_dirs)
  cols = 4
  rows = (n_cats + cols - 1) // cols
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.5))
  axes = np.array(axes).reshape(rows, cols)
  for idx, cat in enumerate(cats_with_dirs):
    ax = axes[idx // cols, idx % cols]
    by_mode = dr[cat]['by_mode']
    for mode, payload in by_mode.items():
      mag = payload.get('magnitude_per_layer') or []
      if mag:
        ax.plot(mag, label=f'{mode} (n={payload.get("sample_count", 0)})',
                color=MODE_COLORS.get(mode, '#888'), linewidth=1.5)
    ax.set_title(cat, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(alpha=0.15)
  for idx in range(n_cats, rows * cols):
    axes[idx // cols, idx % cols].set_visible(False)
  fig.suptitle('Direction Magnitude per Layer (by refusal mode, per category)', fontsize=12, y=1.0)
  plt.tight_layout()
  plt.savefig(OUT / '05_direction_magnitudes.png', dpi=110, facecolor=plt.rcParams['figure.facecolor'])
  plt.close()
else:
  print('no direction_results in run; skipping chart 5')

print('charts written to', OUT)
for png in sorted(OUT.glob('*.png')):
  print(f'  {png.name}: {png.stat().st_size:,} bytes')
