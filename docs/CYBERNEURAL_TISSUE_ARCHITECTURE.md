# Cyberneural tissue architecture — foundation milestone

## Objective

Evolve the current GPU spiking simulator into an experimental computational-tissue
platform. The target is a self-developing substrate whose neural and non-neural
populations can be inspected, cloned, excised, grafted, transferred, and evaluated
while the simulation backend continues running.

This is a computational model. Cell classes and state variables are abstractions;
the implementation must not claim biological equivalence without validation.

## Architectural boundaries

1. **Simulation worker:** remains the exclusive owner of CUDA runtime state.
2. **Tissue model:** owns stable module identities, local CSR topology, population
   descriptors, interface ports, and cross-module binding adapters.
3. **Surgery layer:** performs clone, excise, graft, connect, and disconnect
   operations as validated changes. External adapters are never copied implicitly
   when a module is cloned.
4. **Physiology layer (next milestone):** adds explicit astrocyte-, microglia-, and
   oligodendrocyte-like population dynamics, extracellular fields, damage, and
   repair. Current population fields are data-model placeholders, not active cell
   simulations.
5. **Optimization layer:** evaluates isolated candidate copies against a documented
   objective and hard retention/stability/damage constraints. A candidate is not
   promoted merely because prediction error improves.
6. **Qt/QML operating table:** remains an asynchronous IPC client. It will issue
   surgery and experiment requests; it must never own simulation state or call
   CUDA directly.

## Implemented in this foundation

- Stable module and adapter IDs.
- Local CSR topology embedded in each neural module.
- Explicit excitatory, inhibitory, astrocyte-like, microglia-like, and
  oligodendrocyte-like population descriptors.
- Input/output ports with bounded adapter fan-in/fan-out.
- Structural cloning (copy graph/configuration, reset population runtime state).
- Stateful cloning (copy graph/configuration and population state).
- Excision removes dangling adapters and returns a detached module.
- Grafting restores a detached module; reconnection is explicit.
- A transparent candidate objective with hard acceptance gates.

## Deliberately not claimed yet

- No glial or immune CUDA kernels exist in this milestone.
- No live GPU topology surgery or VRAM residency migration exists yet.
- No biological histology is simulated; only fields and interfaces are reserved.
- Candidate scoring is an engineering selection policy, not a biological law.
- The frontend is not yet wired to the surgery API.
- No production STDP defaults are changed by the calibration archive.

## Safe implementation sequence

1. Land and CPU-test the tissue domain model.
2. Add controller-worker commands and IPC request/response types for snapshots
   and topology transactions.
3. Add candidate cloning/evaluation in an isolated workspace and commit accepted
   topology changes only at simulation batch boundaries.
4. Add GPU residency/checkpoint contracts preserving neuron state, traces, delay
   queues, and adapter state.
5. Add explicit glial/immune/environment state updates with independent update
   cadences.
6. Add Qt/QML operating-table panels for tissue inspection, topology surgery,
   candidate comparison, and virtual histology.
7. Validate functional adaptation, retention, recovery, and stability against
   fixed-topology and weight-only controls.

## Performance contract

- The Qt/QML thread only sends asynchronous IPC requests and renders cached data.
- The simulation worker remains the sole owner of CUDA objects.
- Expensive topology edits are prepared off the hot path and committed at safe
  simulation boundaries.
- Telemetry is coalesced; spike/event streams are bounded.
- Candidate evaluation must not stall the parent simulation.
