#!/usr/bin/env python3
"""Standard-library analysis for Neuromancer V35 inhibitory phase-response CSV."""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "projects/neuromancer/results/v35-phase-response/inhibitory_phase_response_v35.csv"
if not path.exists():
    raise SystemExit(f"CSV not found: {path}")

with path.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("CSV contains no data rows")

def num(row, key):
    return float(row[key])

def med(values):
    return statistics.median(values) if values else float("nan")

def mean(values):
    return statistics.mean(values) if values else float("nan")

finite = [r for r in rows if int(r["finite"]) == 1]
print(f"Rows: {len(rows)}")
print(f"Finite rows: {len(finite)}")
print(f"Seeds: {sorted({r['seed'] for r in rows})}")
print(f"Gate modes: {len({r['gate'] for r in rows})}")
print()

# Recruitment paired against the E-only ablation within each parameter/seed row.
by_gate = defaultdict(lambda: {"e": 0, "both": 0, "suppressed": 0, "facilitated": 0, "spike_free": 0})
for r in finite:
    g = by_gate[r["gate"]]
    e_fire = int(r["target_spikes_e_only"]) > 0
    both_fire = int(r["target_spikes_both"]) > 0
    g["e"] += e_fire
    g["both"] += both_fire
    g["suppressed"] += e_fire and not both_fire
    g["facilitated"] += (not e_fire) and both_fire
    g["spike_free"] += int(r["all_spike_free"]) == 1

print("1. RECRUITMENT SUMMARY")
print("gate                         E-only  E+I  suppressed  spike-free")
for gate in sorted(by_gate):
    x = by_gate[gate]
    print(f"{gate:28s} {x['e']:6d} {x['both']:4d} {x['suppressed']:11d} {x['spike_free']:11d}")
print()

# Phase-response curves: summarize inhibitory-only hyperpolarization and
# paired recruitment suppression over the full set of timing offsets/weights.
phase_rows = defaultdict(list)
for r in finite:
    if r["gate"].startswith("inhibitory_phase_"):
        phase_rows[int(float(r["gate_phase_shift_deg"]))].append(r)

print("2. INHIBITORY PHASE-RESPONSE CURVE")
print("phase  I-only min delta mean/median (mV)  E-only->E+I suppression  n")
for phase in sorted(phase_rows):
    group = phase_rows[phase]
    deltas = [num(r, "minimum_i_only_delta_mv") for r in group]
    suppressed = sum(int(r["target_spikes_e_only"]) > 0 and int(r["target_spikes_both"]) == 0 for r in group)
    print(f"{phase:3d}°   {mean(deltas):9.3f} / {med(deltas):9.3f}             {suppressed:3d}/{len(group):3d}             {len(group)}")
print()

print("3. PHASE RESPONSE BY STIMULUS OFFSET")
print("phase  offset_ms  mean I-only minimum delta (mV)  suppression/n")
for phase in sorted(phase_rows):
    group = phase_rows[phase]
    offsets = sorted({num(r, "i_offset_ms") for r in group})
    for offset in offsets:
        subset = [r for r in group if num(r, "i_offset_ms") == offset]
        deltas = [num(r, "minimum_i_only_delta_mv") for r in subset]
        suppressed = sum(int(r["target_spikes_e_only"]) > 0 and int(r["target_spikes_both"]) == 0 for r in subset)
        print(f"{phase:3d}°   {offset:+7.1f}       {mean(deltas):9.3f}                 {suppressed}/{len(subset)}")
print()

# Control checks: shifted E-only gate should block E recruitment; all-aligned
# and ungated should provide reference conditions.
for gate in ("ungated", "all_aligned", "excitatory_shifted90"):
    subset = [r for r in finite if r["gate"] == gate]
    if subset:
        e_fire = sum(int(r["target_spikes_e_only"]) > 0 for r in subset)
        both_fire = sum(int(r["target_spikes_both"]) > 0 for r in subset)
        print(f"CONTROL {gate}: E-only recruitment {e_fire}/{len(subset)}; E+I {both_fire}/{len(subset)}")

print()
print("Interpretation guardrails:")
print("- Inhibitory-only phase sweep shifts the gate on I-source events while leaving E transmission ungated.")
print("- A phase-dependent voltage or recruitment curve supports phase-selective control in this controlled target assay.")
print("- The minimum I-only voltage delta is a target-voltage contrast, not a direct measurement of synaptic current.")
print("- Rows share fixed topology/stimulus structure; four seeds vary initial voltage only, not independent network topologies.")
print("- This is not evidence yet for autonomous oscillations, learned phase coupling, or network-scale inhibition.")
