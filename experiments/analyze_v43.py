#!/usr/bin/env python3
"""Standard-library analyzer for Neuromancer V43 CSV artifacts."""
import csv
import json
import pathlib
import statistics
import sys

def rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))

def numbers(data, key):
    return [float(row[key]) for row in data if row.get(key, "") not in ("", "nan")]

def mean(data, key):
    values = numbers(data, key)
    return statistics.mean(values) if values else float("nan")

def main():
    root = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("experiments/logs/v43_integrated_learning")
    required = ["v43_trials.csv", "v43_phase.csv", "v43_structure.csv",
                "v43_recall.csv", "v43_ablation.csv", "v43_summary.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SystemExit("Missing V43 artifacts: " + ", ".join(missing))

    trials = rows(root / "v43_trials.csv")
    full = [r for r in trials if r["condition"] == "full"]
    shuffled = [r for r in trials if r["condition"] == "shuffled"]
    ablation = rows(root / "v43_ablation.csv")
    structure = rows(root / "v43_structure.csv")
    recall = rows(root / "v43_recall.csv")
    summary = json.loads((root / "v43_summary.json").read_text())

    print("NEUROMANCER V43 ANALYSIS")
    print("trials:", len(trials), "full seeds:", len(full), "shuffled seeds:", len(shuffled))
    for label, data in (("full", full), ("shuffled", shuffled)):
        print(f"{label}: mean_AB_delta={mean(data, 'ab_weight_delta'):.6f}",
              f"mean_BC_delta={mean(data, 'bc_weight_delta'):.6f}",
              f"phase_confidence={mean(data, 'phase_confidence'):.4f}",
              f"B_recall={mean(data, 'b_recall'):.2f}",
              f"C_recall={mean(data, 'c_recall'):.2f}",
              f"precision={mean(data, 'precision'):.4f}",
              f"recall={mean(data, 'recall'):.4f}",
              f"jaccard={mean(data, 'jaccard'):.4f}")
    print("structure: births=", sum(int(r["births"]) for r in structure),
          "deaths=", sum(int(r["deaths"]) for r in structure),
          "correct_births=", sum(int(r["correct_births"]) for r in structure),
          "incorrect_births=", sum(int(r["incorrect_births"]) for r in structure))
    print("ablation:")
    for row in ablation:
        print(" ", row["condition"],
              "AB=", row["mean_ab_delta"], "BC=", row["mean_bc_delta"],
              "phase_conf=", row["mean_phase_confidence"],
              "births=", row["mean_births"], "deaths=", row["mean_deaths"],
              "mean_recall_units=", row["mean_recall"])
    print("CUDA event-driven phase feedback:", summary.get("cuda_event_driven_phase_feedback"))
    print("scope:", summary.get("scope"))

    if len(full) != 10:
        print("WARNING: expected 10 full-condition seeds")
    if any(r["csr_valid"] != "1" for r in full):
        print("WARNING: CSR validation failure in full condition")
    if not summary.get("cuda_event_driven_phase_feedback", False):
        print("WARNING: CUDA closed-loop phase test did not pass")
    print("Interpretation: this report summarizes the recorded experiment; it does not independently establish GPU associative recall or dendritic morphology.")

if __name__ == "__main__":
    main()
