# Active Connectome and Meta-Topology Architecture

## Status

**Architecture proposal — not yet implemented.**

This document records the intended architectural direction for extending the existing
CSR-based spiking neural network simulator into a modular, parallel, multi-network
system. It is a design contract for incremental development, not a claim that
meta-topology, bridge connectivity, or active-connectome tracking already exists.

The guiding constraint is to preserve the validated single-network CSR topology,
CUDA simulation runtime, STDP implementation, deterministic experiment facilities,
and existing user interface. New capabilities are layered around those components.

## Motivation

The simulator is intended to serve two related purposes:

1. An experimental platform for studying neural dynamics, STDP, topology, and
   learning through controlled, reproducible experiments.
2. A foundation for exploring autonomous SNN–ANN hybrid systems and model
   cybernetic organisms.

A static connectome describes which neural elements are connected. An **active
connectome** additionally describes how that structural network is used and
changes over simulation time: activity, effective communication, plasticity,
and behaviorally relevant state.

A fly-brain-inspired model is a motivating use case for hierarchical organization
and whole-system analysis. It is an architectural inspiration, not a claim that
the simulator reproduces the biological fly brain. Anatomical datasets, region
definitions, neuron identities, and biological validation must be handled
explicitly when such data is introduced.

## Architectural layers

### 1. Local network modules

Each network module owns or references an independently defined neural network:

- Its local neuron and population identity space.
- Its local source-oriented CSR topology.
- Its simulation state and event queues.
- Its local plasticity state, including STDP traces when enabled.
- Its own parameters, random seeds, and experiment metadata.

The existing CSR representation remains the canonical representation for
within-network synapses. The existing runtime remains the execution engine for
a local network. A module should not require changes to the local CSR format
merely because it participates in a larger system.

### 2. Bridge connectivity

Connections between modules are represented separately from local CSR edges.
A bridge edge identifies a source module and source population/neuron, a target
module and target population/neuron, plus synaptic properties such as weight,
sign/type, and propagation delay.

Bridge connectivity may itself use a CSR-like sparse representation, grouped
by source module or source endpoint. Its exact storage layout should be selected
after the first two-module prototype and measured workloads are available.

Keeping bridge connectivity separate provides several benefits:

- Local network graphs can be reused or replaced independently.
- Inter-module wiring can be inspected and analyzed separately.
- A module can be connected to multiple other modules without flattening all
  neurons into one monolithic graph.
- Inter-module plasticity can be studied independently from local STDP.

Global neuron identifiers, module-local identifiers, and population identifiers
must have explicit, validated mappings. Identifier spaces must never be
implicitly conflated.

### 3. Meta-topology

The meta-topology is the system-level graph whose nodes are network modules and
whose edges describe permitted communication between them. It owns or references:

- Module identities and module types.
- Bridge connectivity definitions.
- Module and population mappings.
- Communication policies and routing metadata.
- Simulation-time and synchronization policy.
- Stable identities or hashes for module and bridge topology artifacts.

The meta-topology describes organization and communication. It should not own
the internal neuron state of every module or duplicate the local simulation
runtime's responsibilities.

### 4. Active-connectome state and analysis

The ActiveConnectome layer observes the evolving system and associates
time-indexed measurements with the structural meta-topology. It should
distinguish three related but non-interchangeable concepts:

**Structural connectivity**
- Which local and bridge edges exist.
- Their endpoints, types, delays, and stored weights.
- Topology provenance, version, and identity.

**Functional connectivity**
- Which pathways are participating in communication during a defined
  observation window.
- Event counts, rates, propagation latency, co-activity, and other explicitly
  defined functional measures.
- Functional connectivity is an observation or derived measure; it does not
  silently create or delete structural edges.

**Plastic connectivity**
- The time evolution of synaptic weights and plasticity state.
- Local and bridge plasticity are separately identified.
- Any future structural rewiring is an explicit operation with its own rules,
  provenance, and validation.

The ActiveConnectome should retain enough metadata to relate measurements to
simulation time, experiment configuration, module identities, topology hashes,
and random seeds. It should support both whole-system summaries and drill-down
to individual modules, populations, and pathways.

## Parallel execution and deterministic communication

Independent modules may be scheduled concurrently when hardware resources and
data dependencies permit. Parallelism is an execution strategy, not a change to
the mathematical definition of the network.

Inter-module communication requires an explicit event-delivery contract.
The initial implementation should use deterministic simulation boundaries:

1. Each participating module advances through a defined interval.
2. Outgoing events are collected with source identity and simulation timestamp.
3. Bridge routing applies the configured destination mapping and delay.
4. Events are delivered at the correct future simulation step.
5. Modules continue under a documented ordering and synchronization policy.

The event contract must define same-step visibility, delay semantics, ordering
for events targeting the same destination, and behavior when modules use
different time steps. The first prototype should require a shared time step and
a common simulation clock rather than attempting asynchronous multi-rate
execution.

CUDA streams and concurrent module execution can be introduced where safe.
The initial correctness baseline should prioritize deterministic event
semantics and reproducible tests. Cross-module transfers and synchronization
must not cause hidden per-step host readbacks.

## Learning boundaries

Initially, local STDP remains owned by each local network and continues to
operate according to its existing validated implementation.

Bridge plasticity should be a separately configurable and measurable mechanism.
It must not be silently enabled by local STDP or share trace state accidentally.
Experiments should be able to compare:

- Fixed local and fixed bridge connectivity.
- Local STDP with fixed bridge connectivity.
- Fixed local connectivity with independently enabled bridge plasticity.
- Both mechanisms enabled, after each has been validated independently.

Any future meta-level adaptation of module relationships is a third learning
mechanism. It should be introduced only after local and bridge learning can be
measured independently. Structural rewiring must be explicit, bounded, logged,
and reproducible.

## SNN–ANN interoperability

The module abstraction should eventually permit non-spiking computational
modules, including ANN components, without pretending that an ANN is a CSR
spiking network. Such modules need typed input/output adapters and an explicit
contract for representing values, events, timestamps, and update cadence.

The meta-topology can coordinate heterogeneous modules. The first
implementation should remain SNN-only; ANN integration is a future extension
once module lifecycle and event exchange are stable.

## Experimental and analytical requirements

The experiment layer should be able to vary and record:

- Local topology and bridge topology.
- Module count, population sizes, and connectivity density.
- STDP parameters and whether local or bridge plasticity is enabled.
- Input protocols, simulation duration, time step, and synchronization policy.
- Seeds, software revision, and topology artifact hashes.

Useful measurements include firing rates, spike counts, propagation latency,
event traffic, weight distributions and trajectories, saturation, module
activity, cross-module co-activity, numerical validity, execution time, and
memory use.

Analysis should separate structural, functional, and plastic connectivity.
Parameter sweeps and replicated runs can then support sensitivity analysis,
response surfaces, interaction analysis, and comparisons between candidate
network organizations. Performance optimization should be evaluated alongside
learning and behavioral outcomes rather than in isolation.

## Incremental implementation plan

1. **Preserve the baseline.** Keep the current single-network CSR and CUDA
   runtime behavior unchanged; retain its tests as regression coverage.
2. **Define module identity and ownership.** Introduce a lightweight module
   descriptor around an existing local network/runtime, without moving or
   duplicating its internal state.
3. **Prototype two modules.** Run two independently configured SNN modules
   under one coordinator, initially with no bridge edges.
4. **Add bridge connectivity.** Implement a small, validated bridge graph and
   explicit local/global identifier mappings.
5. **Implement event exchange.** Establish deterministic timestamp, delay,
   buffering, and synchronization semantics; test end-to-end spike delivery.
6. **Demonstrate concurrency.** Schedule independent modules concurrently
   where possible and compare results against the deterministic reference
   schedule.
7. **Add persistence and inspection.** Serialize module manifests, bridge
   topology, and stable topology identities; provide inspection and validation.
8. **Add ActiveConnectome observations.** Record structural, functional, and
   plastic state over time and expose analysis outputs.
9. **Study multi-module learning.** Validate local STDP, then bridge plasticity,
   then any explicitly defined meta-level adaptation.
10. **Scale and optimize.** Use measured workloads to evaluate module
    partitioning, CUDA streams, event batching, memory placement, and
    scheduling policies.
11. **Explore organism-scale models.** Import documented biological or
    synthetic connectome data with provenance, mapping, and validation. A
    fly-brain-inspired model is one possible research case study.

## First acceptance milestone

The first meaningful milestone is two independently operating SNN modules
connected by a bridge graph. Tests should demonstrate:

- Each module retains its own local CSR and simulation state.
- A spike from a selected source in module A reaches the intended target in
  module B through a bridge edge.
- Configured bridge delays are honored exactly.
- Local STDP behavior remains unchanged.
- Repeated runs under the same deterministic configuration produce the same
  event schedule and matching measured outcomes within the documented numeric
  tolerance.
- Existing single-network tests continue to pass.

Only after this milestone should the implementation expand to larger
meta-topologies or organism-scale datasets.

## Non-goals for the initial implementation

- Replacing the current CSR topology or CUDA runtime.
- Flattening all modules into one mandatory monolithic graph.
- Claiming biological equivalence to a real connectome.
- Enabling autonomous rewiring or meta-learning before its behavior is
  independently testable.
- Requiring ANN components in the first multi-network prototype.
- Assuming all modules can execute concurrently on one GPU without resource
  contention or synchronization costs.

The design is successful if it extends the validated simulator incrementally,
makes multi-network behavior measurable and reproducible, and provides a
clear route from local plasticity experiments to active-connectome research.
