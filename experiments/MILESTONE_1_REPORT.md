# Milestone 1 — Network Characterization Report

**Project:** ChatGPTNeuronSimulator
**Date:** 2026-11-05
**Status:** Experimental characterization complete

## 1. Objectives

Milestone 1 establishes a reproducible baseline for the spiking neural
network and measures how changing inhibitory-to-excitatory connectivity
affects population activity.

The experiments address four questions:

1. What are the baseline excitatory and inhibitory firing rates?
2. How does increasing inhibitory-to-excitatory connectivity affect those
   rates?
3. Is the observed effect consistent across different topology seeds?
4. How does the effect develop over simulated time?

## 2. Simulation environment

Recorded environment:

- GPU: NVIDIA GeForce RTX 5070 Ti
- Compute capability: 12.0
- CUDA: 13.4
- Operating system: Debian 13
- Compiler: GCC 14.2.0
- Excitatory neurons: 81,920
- Inhibitory neurons: 10,240
- Total neurons: 92,160
- Simulation timestep: 0.5 ms

The principal 5-second experiments used 10,000 simulation steps.

## 3. Connectivity dose-response experiment

### Method

Five topology seeds were tested:

- 1001
- 1002
- 1003
- 1004
- 1005

The neuron random seed and delay seed were held constant:

- Neuron seed: 12345
- Delay seed: 4919422011527349573

The inhibitory-to-excitatory connectivity probability (`p_ie`) was
varied across five levels. Other configuration parameters were held
constant.

### Results

Values are mean ± sample standard deviation across five topology seeds.

| p_ie | E firing rate (Hz) | I firing rate (Hz) | Mean voltage (mV) |
|---:|---:|---:|---:|
| 0.00500 | 21.84487 ± 0.00031 | 117.24249 ± 0.00651 | -65.24239 ± 0.03073 |
| 0.00625 | 21.75479 ± 0.00032 | 117.22902 ± 0.00566 | -65.24889 ± 0.04252 |
| 0.00750 | 21.66514 ± 0.00016 | 117.21223 ± 0.00186 | -65.23394 ± 0.00858 |
| 0.01000 | 21.48666 ± 0.00030 | 117.18830 ± 0.00630 | -65.20916 ± 0.00972 |
| 0.01250 | 21.30800 ± 0.00038 | 117.15634 ± 0.00361 | -65.21963 ± 0.01767 |

### Paired change relative to baseline

| p_ie | E rate change (Hz) | E rate change (%) | I rate change (Hz) | I rate change (%) |
|---:|---:|---:|---:|---:|
| 0.00625 | -0.09008 | -0.41238% | -0.01347 | -0.01149% |
| 0.00750 | -0.17973 | -0.82273% | -0.03026 | -0.02581% |
| 0.01000 | -0.35821 | -1.63979% | -0.05420 | -0.04622% |
| 0.01250 | -0.53687 | -2.45764% | -0.08615 | -0.07348% |

Excitatory firing decreased monotonically across all five tested
connectivity levels for every topology seed.

### Interpretation

Within the tested range, increasing inhibitory-to-excitatory connectivity
was consistently associated with lower excitatory population firing
rates. Inhibitory population firing rates also decreased slightly.

The measured excitatory response was approximately monotonic over the
tested range. These results do not establish that the relationship
remains linear outside the tested values.

The experiment used five topology seeds. The small variation across
those seeds is encouraging, but this sample size does not justify
broad claims about all possible network topologies.

## 4. Baseline temporal characterization

### Method

The baseline configuration was simulated for 10,000 steps at 0.5 ms
per step, corresponding to 5,000 ms of simulated time.

Telemetry was collected in 1,000-step windows, each representing
500 ms of simulated time.

Configuration:

- `p_ie`: 0.005
- Neuron seed: 12345
- Topology seed: 6075162723858990937
- Delay seed: 4919422011527349573

### Overall results

- Excitatory spikes: 8,947,755
- Inhibitory spikes: 6,002,513
- Total spikes: 14,950,268
- Excitatory mean firing rate: 21.8451 Hz
- Inhibitory mean firing rate: 117.237 Hz
- Mean membrane voltage: -65.23 mV

### Window results

| Window (ms) | E rate (Hz) | I rate (Hz) | Voltage snapshot (mV) |
|---:|---:|---:|---:|
| 0–500 | 23.4790 | 118.6470 | -65.6182 |
| 500–1000 | 21.6388 | 117.0640 | -65.6034 |
| 1000–1500 | 21.6499 | 117.0459 | -65.0397 |
| 1500–2000 | 21.6933 | 117.0834 | -65.6898 |
| 2000–2500 | 21.6616 | 117.0830 | -65.5169 |
| 2500–3000 | 21.6925 | 117.1154 | -65.8433 |
| 3000–3500 | 21.5964 | 117.0625 | -65.2107 |
| 3500–4000 | 21.7183 | 117.0791 | -65.0655 |
| 4000–4500 | 21.6216 | 117.1027 | -65.0962 |
| 4500–5000 | 21.6997 | 117.0832 | -65.2300 |

The first window had higher firing rates than the subsequent windows.

Across windows 2–10:

- E firing rate mean: 21.663561 Hz
- E firing rate sample SD: 0.040474 Hz
- I firing rate mean: 117.079905 Hz
- I firing rate sample SD: 0.021015 Hz

This suggests an initial transient followed by comparatively stable
population-level firing during the remainder of this run. It does not
establish mathematical stationarity.

Voltage values are snapshots at window boundaries, not averages over
the entire window.

## 5. Matched temporal treatment comparison

### Method

The baseline and treatment simulations used identical simulation
settings and random seeds. The only configuration parameter changed
was `p_ie`.

| Parameter | Control | Treatment |
|---|---:|---:|
| `p_ie` | 0.00500 | 0.00625 |
| Relative connectivity increase | — | 25% |
| Neuron seed | 12345 | 12345 |
| Topology seed | 6075162723858990937 | 6075162723858990937 |
| Delay seed | 4919422011527349573 | 4919422011527349573 |
| Simulation duration | 5,000 ms | 5,000 ms |

Both simulations used ten 500 ms telemetry windows.

### Overall results

| Measurement | Control | Treatment |
|---|---:|---:|
| Excitatory spikes | 8,947,755 | 8,910,817 |
| Inhibitory spikes | 6,002,513 | 6,002,219 |
| Total spikes | 14,950,268 | 14,913,036 |
| Excitatory firing rate | 21.8451 Hz | 21.7549 Hz |
| Inhibitory firing rate | 117.237 Hz | 117.231 Hz |
| Mean membrane voltage | -65.2300 mV | -65.2289 mV |

### Settled temporal comparison

For this comparison, the first 500 ms window is treated separately.
The following nine windows are summarized as the post-initial period.

| Measurement | Control mean | Treatment mean | Treatment minus control |
|---|---:|---:|---:|
| E firing rate | 21.663561 Hz | 21.575019 Hz | -0.088542 Hz |
| I firing rate | 117.079905 Hz | 117.074544 Hz | -0.005360 Hz |
| Voltage snapshot | -65.366175 mV | -65.361002 mV | +0.005173 mV |

Excitatory firing was lower in the treatment in all ten corresponding
windows.

The settled excitatory firing-rate reduction was approximately 0.409%
relative to the control's post-initial mean.

The inhibitory firing-rate difference and voltage snapshot difference
were comparatively small.

### Cross-experiment consistency

The five-topology-seed dose-response experiment measured an average
excitatory firing-rate change of -0.09008 Hz for the change from
`p_ie = 0.005` to `p_ie = 0.00625`.

The matched temporal experiment measured a post-initial mean difference
of -0.088542 Hz for the same connectivity change.

These values are close. This agreement supports consistency between
the two measurement approaches. The temporal comparison uses one
matched topology and is not an independent five-topology replication.

## 6. Main findings

1. Increasing `p_ie` reduced excitatory population firing across all
   five tested topology seeds.
2. The excitatory response was monotonic across the tested connectivity
   levels.
3. Inhibitory firing changed much less than excitatory firing.
4. The baseline simulation showed a higher initial firing rate followed
   by comparatively stable population-level rates.
5. The matched temporal treatment reduced excitatory firing in every
   measured window.
6. The temporal treatment result closely matched the corresponding
   dose-response result.

Taken together, these experiments support the conclusion that
increasing inhibitory-to-excitatory connectivity suppresses excitatory
population activity in this network configuration.

## 7. Limitations and cautions

- The topology dose-response used five topology seeds.
- The time-resolved treatment comparison used one matched topology.
- The temporal experiment lasted five seconds of simulated time.
- Nine post-initial windows are not sufficient to establish formal
  stationarity.
- Voltage telemetry records a boundary snapshot rather than a
  window-averaged voltage distribution.
- These experiments characterize population-level behavior; they do
  not establish individual-neuron response distributions or detailed
  causal mechanisms.
- Results apply to the tested neuron model, network size, input
  settings, connectivity range, and delay configuration.
- No claim of statistical significance is made here.

## 8. Reproducibility and data inventory

Primary configurations:

- `experiments/control_long_duration.json`
- `experiments/increased_inhibitory_connectivity.json`

Primary time-resolved logs:

- `experiments/logs/time_resolved_baseline.jsonl`
- `experiments/logs/time_resolved_increased_inhibition.jsonl`

Connectivity dose-response logs:

- `experiments/logs/topology_seed_sweep.jsonl`
- `experiments/logs/topology_dose_response.jsonl`

Additional experiment logs retained for future analysis:

- `experiments/logs/control_repeats.jsonl`
- `experiments/logs/increased_inhibitory_connectivity.jsonl`
- `experiments/logs/increased_inhibitory_connectivity_50pct.jsonl`
- `experiments/logs/increased_inhibitory_connectivity_100pct.jsonl`
- `experiments/logs/long_duration.jsonl`

Topology seed-sweep configurations are retained under:

- `experiments/topology_seed_sweep/`

The JSONL logs are append-oriented experimental records. Preserve the
original records when performing future analysis.

## 9. Validation

The CUDA build completed successfully.

All seven CTest tests passed:

1. `parameter_tests`
2. `ipc_parameter_json_tests`
3. `topology_tests`
4. `cuda_runtime_smoke`
5. `determinism_smoke`
6. `controller_smoke`
7. `ipc_server_integration`

Telemetry smoke testing also verified that the sum of per-window spike
counts matched the final spike totals and that the expected number of
windows was produced.

### Subsequent repository validation — November 5, 2026

After the original milestone validation, the test suite was expanded to
include CPU and CUDA STDP tests. The current CUDA build completed
successfully, and all nine CTest tests passed:

1. `parameter_tests`
2. `ipc_parameter_json_tests`
3. `topology_tests`
4. `stdp_tests`
5. `stdp_cuda_tests`
6. `cuda_runtime_smoke`
7. `determinism_smoke`
8. `controller_smoke`
9. `ipc_server_integration`

This later result records the current repository state; it does not
change the historical seven-test validation above.

## 10. Milestone conclusion

Milestone 1 — Network Characterization — is experimentally complete.

The simulator has demonstrated reproducible baseline activity,
a consistent connectivity-dependent reduction in excitatory firing,
and time-resolved telemetry capable of observing population behavior
throughout a simulation.

The next milestone is neuron-dynamics characterization, including
regular-spiking, fast-spiking, and bursting behavior. These experiments
should be designed and validated before moving on to synaptic
plasticity and SNN–ANN hybrid learning.
