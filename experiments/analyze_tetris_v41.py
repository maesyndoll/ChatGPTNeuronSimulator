#!/usr/bin/env python3
"""V41 decision fidelity and cross-version Tetris analysis; standard library only."""
import csv,sys,statistics
from pathlib import Path
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path.home()/"projects/neuromancer/results"
files=[("V37",root/"v37-tetris"/"tetris_v37_games.csv"),("V38",root/"v38-tetris"/"tetris_v38_games.csv"),("V39",root/"v39-tetris"/"tetris_v38_games.csv"),("V40",root/"v40-tetris"/"tetris_v38_games.csv"),("V41",root/"v41-tetris"/"tetris_v41_games.csv")]
def val(r,k,d=0):
 try:return float(r.get(k,d) or d)
 except (ValueError,TypeError):return float(d)
print("CROSS-VERSION HELD-OUT PERFORMANCE")
for v,p in files:
 if not p.exists():print(f"{v}: missing {p}");continue
 with p.open(newline="") as f: rows=list(csv.DictReader(f))
 rows=[r for r in rows if r.get("stage","evaluation")=="evaluation"]
 agents={}
 for a in sorted({r.get("agent","?") for r in rows}):
  rr=[r for r in rows if r.get("agent")==a];agents[a]=rr
  if rr:print(f"{v:>3} {a:12s} n={len(rr):2d} score={statistics.mean(val(x,'score') for x in rr):8.1f} lines={statistics.mean(val(x,'lines') for x in rr):6.2f} pieces={statistics.mean(val(x,'pieces') for x in rr):6.1f} topouts={sum(val(x,'topout') for x in rr):.0f}")
 if "neuromancer" in agents and "heuristic" in agents:
  n={x.get("seed"):val(x,"score") for x in agents["neuromancer"]};h={x.get("seed"):val(x,"score") for x in agents["heuristic"]};keys=sorted(set(n)&set(h))
  if keys:
   d=[n[k]-h[k] for k in keys];print(f"    paired vs heuristic: n={len(d)} mean_delta={statistics.mean(d):+.1f} wins={sum(x>0 for x in d)} ties={sum(x==0 for x in d)} losses={sum(x<0 for x in d)}")
print("\nV41 DECISION FIDELITY")
p=root/"v41-tetris"/"tetris_v41_decisions.csv"
if not p.exists():print(f"Missing {p}")
else:
 with p.open(newline="") as f:d=list(csv.DictReader(f))
 if d:
  n=len(d);agree=sum(int(x["q_best_index"])==int(x["neural_selected_index"]) for x in d)
  regret=[val(x,"utility_regret") for x in d];rank=[val(x,"selected_q_rank") for x in d];margin=[val(x,"spike_margin") for x in d]
  silent=sum(val(x,"total_output_spikes")==0 for x in d)
  print(f"decisions={n}; exact neural/Q-best agreement={agree}/{n} ({100*agree/n:.1f}%)")
  print(f"selected Q-rank mean={statistics.mean(rank):.2f}, median={statistics.median(rank):.1f}")
  print(f"utility regret mean={statistics.mean(regret):.4f}, median={statistics.median(regret):.4f}, zero regret={sum(x==0 for x in regret)}/{n}")
  print(f"silent output decisions={silent}/{n}; mean top-vs-second spike margin={statistics.mean(margin):.2f}")
  groups={}
  for x in d:groups.setdefault(int(x["legal_candidates"]),[]).append(x)
  for k,rr in sorted(groups.items()):
   a=sum(int(x["q_best_index"])==int(x["neural_selected_index"]) for x in rr)
   print(f"  candidates={k:2d} n={len(rr):4d} exact={100*a/len(rr):5.1f}% mean_regret={statistics.mean(val(x,'utility_regret') for x in rr):.4f}")
 else:print("No decision rows.")
print("\nCross-version samples/seeds differ; these summaries are descriptive, not a paired causal comparison.")
