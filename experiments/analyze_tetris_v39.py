#!/usr/bin/env python3
"""V39 Tetris training and held-out evaluation analysis; standard library only."""
import csv,sys,statistics
from pathlib import Path
p=Path(sys.argv[1]) if len(sys.argv)>1 else Path.home()/"projects/neuromancer/results/v39-tetris/tetris_v39_games.csv"
if not p.exists(): raise SystemExit(f"CSV not found: {p}")
rows=list(csv.DictReader(p.open(newline="")))
for stage in ("train","evaluation"):
 g=[r for r in rows if r["stage"]==stage]
 if not g: continue
 print(f"\n{stage.upper()} games: {len(g)}")
 for a in sorted({r["agent"] for r in g}):
  z=[r for r in g if r["agent"]==a]
  print(f"{a:13s} n={len(z):2d} mean_score={statistics.mean(int(x['score']) for x in z):8.1f} mean_lines={statistics.mean(int(x['lines']) for x in z):6.2f} mean_pieces={statistics.mean(int(x['pieces']) for x in z):6.1f} topouts={sum(int(x['topout']) for x in z):2d} active_decisions={sum(int(x['neural_active_decisions']) for x in z):5d} output_spikes={sum(int(x['neural_action_spikes']) for x in z):7d}")
 if stage=="train":
  scores=[int(x["score"]) for x in g]
  print("training score by 10-episode block:",[round(statistics.mean(scores[i:i+10]),1) for i in range(0,len(scores),10)])
 else:
  d={(x["agent"],x["seed"]):x for x in g}; seeds=sorted({x["seed"] for x in g})
  for baseline in ("random","heuristic"):
   dif=[int(d[("neuromancer",s)]["score"])-int(d[(baseline,s)]["score"]) for s in seeds]
   print(f"Neuromancer vs {baseline}: mean paired difference={statistics.mean(dif):+.1f}; wins={sum(x>0 for x in dif)}, ties={sum(x==0 for x in dif)}, losses={sum(x<0 for x in dif)}")
print("\nCheck held-out results separately from training. Output spikes are counted only in the action-output population.")
