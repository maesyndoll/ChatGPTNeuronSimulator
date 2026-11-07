#!/usr/bin/env python3
"""Analyze Neuromancer V36 phase-gate robustness CSV using only the standard library."""
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path
path = Path(sys.argv[1]) if len(sys.argv)>1 else Path.home()/"projects/neuromancer/results/v36-phase-robustness/inhibitory_phase_robustness_v36.csv"
if not path.exists(): raise SystemExit(f"CSV not found: {path}")
with path.open(newline="") as f: rows=list(csv.DictReader(f))
if not rows: raise SystemExit("CSV contains no rows")
finite=[r for r in rows if int(r["finite"])==1]
def mean(xs): return statistics.mean(xs) if xs else float("nan")
def key(r): return (float(r["frequency_hz"]),float(r["gate_half_width_deg"]),int(float(r["gate_phase_shift_deg"])))
groups=defaultdict(list)
for r in finite:
    if r["gate"].startswith("inhibitory_phase_"): groups[key(r)].append(r)
print(f"Rows: {len(rows)}; finite: {len(finite)}; seeds: {len(set(r['seed'] for r in rows))}")
print("V36 phase-gate robustness: mean inhibitory-only minimum voltage delta and recruitment suppression")
print("freq_hz width_deg phase_deg mean_delta_mV suppressed/total")
for (freq,width,phase),g in sorted(groups.items()):
    delta=mean([float(r["minimum_i_only_delta_mv"]) for r in g])
    sup=sum(int(r["target_spikes_e_only"])>0 and int(r["target_spikes_both"])==0 for r in g)
    print(f"{freq:6.1f} {width:8.1f} {phase:9d} {delta:13.3f} {sup:4d}/{len(g):4d}")
print("\nControl summary")
for name in ("ungated","all_aligned","excitatory_shifted90"):
    g=[r for r in finite if r["gate"]==name]
    if g:
        e=sum(int(r["target_spikes_e_only"])>0 for r in g)
        both=sum(int(r["target_spikes_both"])>0 for r in g)
        print(f"{name}: E-only {e}/{len(g)}, E+I {both}/{len(g)}")
print("\nInterpretation limits: externally imposed oscillator; fixed tiny topology; initial-voltage seeds are not independent network topologies.")
