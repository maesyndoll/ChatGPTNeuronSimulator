#!/usr/bin/env python3
"""Summarize V37 matched-seed Tetris agent results using only the standard library."""
import csv, sys, statistics
from pathlib import Path
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path.home()/"projects/neuromancer/results/v37-tetris/tetris_v37_games.csv"
if not p.exists(): raise SystemExit(f"CSV not found: {p}")
rows=list(csv.DictReader(p.open(newline="")))
if not rows: raise SystemExit("CSV contains no rows")
agents=sorted({r["agent"] for r in rows})
print(f"Games: {len(rows)}")
print("agent       n  mean_score  mean_lines  mean_pieces  topouts  active_decisions  action_spikes")
for a in agents:
 g=[r for r in rows if r["agent"]==a]
 print(f"{a:11s} {len(g):2d} {statistics.mean(int(x['score']) for x in g):10.1f} {statistics.mean(int(x['lines']) for x in g):11.2f} {statistics.mean(int(x['pieces']) for x in g):12.1f} {sum(int(x['topout']) for x in g):8d} {sum(int(x['neural_active_decisions']) for x in g):16d} {sum(int(x['neural_action_spikes']) for x in g):13d}")
print("\nMatched-seed comparison (Neuromancer minus each baseline):")
by={(r["agent"],r["seed"]):r for r in rows}
for base in ("random","heuristic"):
 dif=[]
 for seed in sorted({r["seed"] for r in rows}):
  if ("neuromancer",seed) in by and (base,seed) in by:
   dif.append(int(by[("neuromancer",seed)]["score"])-int(by[(base,seed)]["score"]))
 print(f"vs {base}: mean score difference {statistics.mean(dif):+.1f}; per-seed {dif}")
print("\nCaution: V37 uses fixed utility weights and does not yet learn from rewards.")
