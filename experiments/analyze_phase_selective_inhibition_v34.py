import csv
from pathlib import Path
from collections import defaultdict
from statistics import mean, median

path = Path.home() / "projects/neuromancer/results/v34-phase-selective/phase_selective_inhibition_v34.csv"
with path.open(newline="") as f:
    rows = list(csv.DictReader(f))

def n(r, key):
    try:
        return float(r[key])
    except (KeyError, TypeError, ValueError):
        return 0.0

def yes(r, key):
    return n(r, key) > 0

print("NEUROMANCER V34 — PHASE-SELECTIVE INHIBITION ANALYSIS")
print("=" * 72)
print(f"Rows: {len(rows)}")
print(f"Seeds: {sorted(set(r['seed'] for r in rows))}")
print(f"Finite rows: {sum(r.get('finite','').lower() in ('1','true') for r in rows)}")

print("\n1. RECRUITMENT BY GATE")
print("Gate                   | E-only firing | E+I firing | Suppressed | Facilitated | Spike-free")
for gate in sorted(set(r["gate"] for r in rows)):
    g = [r for r in rows if r["gate"] == gate]
    e = sum(yes(r, "target_spikes_e_only") for r in g)
    both = sum(yes(r, "target_spikes_both") for r in g)
    suppressed = sum(yes(r, "target_spikes_e_only") and not yes(r, "target_spikes_both") for r in g)
    facilitated = sum(not yes(r, "target_spikes_e_only") and yes(r, "target_spikes_both") for r in g)
    sf = sum(r["all_spike_free"] == "1" for r in g)
    print(f"{gate:22s} | {e:13d} | {both:10d} | {suppressed:10d} | {facilitated:11d} | {sf:10d}")

print("\n2. INHIBITORY-ONLY GATE: SUPPRESSION BY I OFFSET")
print("Offset ms | ungated | I-aligned | I-shifted90 | E-shifted90")
for off in sorted(set(n(r, "i_offset_ms") for r in rows)):
    vals = []
    for gate in ("ungated", "inhibitory_aligned", "inhibitory_shifted90", "excitatory_shifted90"):
        g = [r for r in rows if r["gate"] == gate and n(r, "i_offset_ms") == off]
        sup = sum(yes(r, "target_spikes_e_only") and not yes(r, "target_spikes_both") for r in g)
        vals.append(f"{sup:5d}/{len(g):3d}")
    print(f"{off:8.1f} | " + " | ".join(vals))

print("\n3. SUPPRESSION BY INHIBITORY WEIGHT AT COINCIDENCE")
print("I weight | ungated | I-aligned | I-shifted90")
for iw in sorted(set(n(r, "i_weight") for r in rows)):
    vals = []
    for gate in ("ungated", "inhibitory_aligned", "inhibitory_shifted90"):
        g = [r for r in rows if r["gate"] == gate and n(r, "i_weight") == iw and n(r, "i_offset_ms") == 0]
        sup = sum(yes(r, "target_spikes_e_only") and not yes(r, "target_spikes_both") for r in g)
        vals.append(f"{sup:3d}/{len(g):2d}")
    print(f"{iw:8.1f} | " + " | ".join(vals))

print("\n4. STRICTLY SPIKE-FREE NONLINEAR INTERACTION")
print("Only rows where none, E-only, I-only, and E+I target traces spiked.")
print("Gate                   | n | mean mV | median mV | min mV | max mV")
for gate in sorted(set(r["gate"] for r in rows)):
    vals = [n(r, "peak_interaction_spike_free_mv") for r in rows
            if r["gate"] == gate and r["all_spike_free"] == "1"]
    if vals:
        print(f"{gate:22s} | {len(vals):4d} | {mean(vals):8.3f} | {median(vals):9.3f} | {min(vals):7.3f} | {max(vals):7.3f}")
    else:
        print(f"{gate:22s} |    0 | no spike-free cases")

print("\n5. SPIKE-FREE INTERACTION BY GATE AND I TIMING")
print("Gate                   | Offset ms | n | mean mV | median mV")
groups = defaultdict(list)
for r in rows:
    if r["all_spike_free"] == "1":
        groups[(r["gate"], n(r, "i_offset_ms"))].append(n(r, "peak_interaction_spike_free_mv"))
for (gate, off), vals in sorted(groups.items()):
    print(f"{gate:22s} | {off:9.1f} | {len(vals):3d} | {mean(vals):8.3f} | {median(vals):9.3f}")

print("\n6. RECRUITMENT BY E WEIGHT")
for gate in sorted(set(r["gate"] for r in rows)):
    print(f"\n{gate}")
    print("E weight | E-only | E+I | trials")
    for ew in sorted(set(n(r, "e_weight") for r in rows)):
        g = [r for r in rows if r["gate"] == gate and n(r, "e_weight") == ew]
        e = sum(yes(r, "target_spikes_e_only") for r in g)
        both = sum(yes(r, "target_spikes_both") for r in g)
        print(f"{ew:8.1f} | {e:6d} | {both:3d} | {len(g):6d}")
