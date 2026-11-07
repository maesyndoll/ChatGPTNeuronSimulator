#!/usr/bin/env python3
"""Analyze V38 training and held-out Tetris evaluation using only stdlib."""
import csv,sys,statistics
from pathlib import Path
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path.home()/"projects/neuromancer/results/v38-tetris/tetris_v38_games.csv"
if not p.exists(): raise SystemExit(f"CSV not found: {p}")
rows=list(csv.DictReader(p.open(newline="")))
if not rows: raise SystemExit("CSV contains no rows")
for stage in ("train","evaluation"):
 group=[r for r in rows if r["stage"]==stage]
 if not group: continue
 print(f"\n{stage.upper()} games: {len(group)}")
 agents=sorted({r["agent"] for r in group})
 print("agent         n  mean_score  mean_lines  mean_pieces  topouts  active_decisions  action_spikes")
 for a in agents:
  g=[r for r in group if r["agent"]==a]
  print(f"{a:13s} {len(g):2d} {statistics.mean(int(x['score']) for x in g):10.1f} {statistics.mean(int(x['lines']) for x in g):11.2f} {statistics.mean(int(x['pieces']) for x in g):12.1f} {sum(int(x['topout']) for x in g):8d} {sum(int(x['neural_active_decisions']) for x in g):16d} {sum(int(x['neural_action_spikes']) for x in g):13d}")
 if stage=="train":
  scores=[int(r["score"]) for r in group]
  chunks=[scores[i:i+6] for i in range(0,len(scores),6)]
  print("training score by 6-episode block:",[round(statistics.mean(x),1) for x in chunks])
 else:
  by={(r["agent"],r["seed"]):r for r in group}
  seeds=sorted({r["seed"] for r in group})
  for base in ("random","heuristic"):
   dif=[int(by[("neuromancer",s)]["score"])-int(by[(base,s)]["score"]) for s in seeds]
   wins=sum(d>0 for d in dif);ties=sum(d==0 for d in dif)
   print(f"Neuromancer vs {base}: mean paired score difference {statistics.mean(dif):+.1f}; wins={wins}, ties={ties}, losses={len(dif)-wins-ties}; per-seed={dif}")
print("\nInterpretation: only held-out evaluation with frozen learned weights measures generalization.")
