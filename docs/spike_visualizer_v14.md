# V14 — Neural activity visualization

V14 replaces the desktop screenshot preview with a neural spike raster.

## Main display

- Horizontal axis: simulation time over the most recent 2,000 simulation steps.
- Vertical axis: neuron ID.
- Green marks: excitatory neuron spikes.
- Amber marks: inhibitory neuron spikes.
- Events are drawn from the backend's recorded spike-event stream.
- Raster redraws are coalesced by the existing bridge update timer.

The right panel shows the 8x8 luminance input map, simulation state, step count,
population spike totals and rates, and event-drop/generation telemetry.

## Input sampling

The desktop is sampled at 2 Hz in this milestone. Its frame is converted to an
8x8 luminance grid. The UI does not show the desktop image and does not apply a
green overlay. The 8x8 input is sent only when the user clicks **Encode current
desktop frame**, and only while the simulation is paused.

The raster is actual backend spike data. The 8x8 grid is measured visual input.
Neither is a learned attention map.

## Build

```bash
cd ~/projects/ChatGPTNeuronSimulator
git fetch origin
git switch --track origin/work/neuromancer-spike-visualizer-v14

cmake -S . -B build-desktop \
  -DSPIKINGMODEL_ENABLE_CUDA=ON \
  -DSPIKINGMODEL_ENABLE_QT_CLIENT=ON
cmake --build build-desktop \
  --target spiking_model neuromancer_desktop_agent -j$(nproc)
```

Restart the backend and launch `neuromancer_desktop_agent`.

## Reading the raster

The raster only shows neurons that actually emitted recorded spikes. A blank
raster can mean the network is silent, the simulation has not advanced, or
spike-event collection is disabled/empty. Check the event count and dropped
counter alongside the plot.

The next engineering step is a controlled visual stimulus suite (dark field,
bright field, left/right bars, checkerboard), plus per-cell input and per-neuron
response measurements. This tests whether the input encoder produces
reproducible spatially selective activity before adding learned visual features
or action selection.
