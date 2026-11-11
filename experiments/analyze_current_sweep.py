import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

log_dir = Path("experiments/logs")
output_dir = Path("experiments/analysis")
output_dir.mkdir(parents=True, exist_ok=True)

records = []

for path in sorted(log_dir.glob("isolated_current_*.jsonl")):
    try:
        lines = path.read_text().splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        if not entries:
            continue

        entry = entries[-1]
        simulation = entry["configuration"]["simulation"]
        results = entry["results"]

        current_min = simulation["excitatory_current_min"]
        current_max = simulation["excitatory_current_max"]

        if current_min != current_max:
            print(f"Skipping non-constant current: {path}")
            continue

        records.append({
            "current": current_min,
            "excitatory_rate_hz":
                results["excitatory_mean_firing_rate_hz"],
            "inhibitory_rate_hz":
                results["inhibitory_mean_firing_rate_hz"],
            "excitatory_spikes": results["excitatory_spikes"],
            "inhibitory_spikes": results["inhibitory_spikes"],
            "steps": results["completed_steps"],
            "valid": results["numerical_state_valid"],
        })

    except (KeyError, ValueError, json.JSONDecodeError) as error:
        print(f"Could not process {path}: {error}")

records.sort(key=lambda item: item["current"])

if not records:
    raise SystemExit("No usable isolated current-sweep records found.")

csv_path = output_dir / "firing_rate_vs_current.csv"

with csv_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

print("\nFIRING RATE VS INPUT CURRENT")
print("-" * 76)
print(f"{'Current':>10} {'Excitatory Hz':>16} {'Inhibitory Hz':>16} {'Valid':>10}")

for record in records:
    print(
        f"{record['current']:>10.3f}"
        f"{record['excitatory_rate_hz']:>16.3f}"
        f"{record['inhibitory_rate_hz']:>16.3f}"
        f"{str(record['valid']):>10}"
    )

plt.figure(figsize=(9, 6))
plt.plot(
    [r["current"] for r in records],
    [r["excitatory_rate_hz"] for r in records],
    marker="o",
    label="Regular-spiking / excitatory",
)
plt.plot(
    [r["current"] for r in records],
    [r["inhibitory_rate_hz"] for r in records],
    marker="s",
    label="Fast-spiking / inhibitory",
)

plt.xlabel("Injected current")
plt.ylabel("Mean firing rate (Hz)")
plt.title("Izhikevich Neuron Firing Rate vs Input Current")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = output_dir / "firing_rate_vs_current.png"
plt.savefig(plot_path, dpi=180)

print(f"\nCSV:  {csv_path}")
print(f"Plot: {plot_path}")
print(f"Processed {len(records)} current levels.")
